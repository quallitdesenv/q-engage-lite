import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from pathlib import Path
from typing import Dict, Tuple
import json
import os
from src.core import Task


class FeatureExtractor:
    """MobileNetV2-based feature extractor for person re-identification."""
    
    def __init__(self, model_path: str = "./pretrained/mobilenet_reid.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model_path = Path(model_path)
        
        # Load weights if available
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            # Use pretrained MobileNetV2 as feature extractor
            print(f"Warning: {model_path} not found, using pretrained MobileNetV2")
            # Save the initial model
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Saved initial model to {self.model_path}")
        
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard re-ID size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        """Build MobileNetV2 feature extractor."""
        base_model = mobilenet_v2(pretrained=True)
        # Use feature extractor + adaptive pooling
        model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return model.to(self.device)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract 1280-dim feature vector from image crop."""
        if image.size == 0:
            return np.zeros(1280)  # MobileNetV2 output dimension
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        # Ensure 1D vector
        if features.ndim != 1:
            features = features.flatten()
        
        return features
    
    def extract_features_batch(self, images: list) -> list:
        """Extract features from multiple images in batch for better GPU utilization."""
        if not images:
            return []
        
        # Filter out empty images
        valid_images = []
        valid_indices = []
        for idx, img in enumerate(images):
            if img.size > 0:
                valid_images.append(img)
                valid_indices.append(idx)
        
        if not valid_images:
            return [np.zeros(1280) for _ in images]
        
        # Batch preprocessing
        batch_tensors = []
        for img in valid_images:
            img_tensor = self.transform(img)
            batch_tensors.append(img_tensor)
        
        # Stack into single batch tensor
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            features = self.model(batch)
            features = features.cpu().numpy()
        
        # Map back to original order
        results = [np.zeros(1280) for _ in images]
        for idx, feat_idx in enumerate(valid_indices):
            feat = features[idx]
            if feat.ndim != 1:
                feat = feat.flatten()
            results[feat_idx] = feat
        
        return results


class DeepSORTTracker:
    """Simple DeepSORT tracker implementation with motion prediction."""
    
    def __init__(self, max_age: int = 70, min_hits: int = 3, iou_threshold: float = 0.25):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.max_age = max_age  # Increased from 30 to handle occlusions better
        self.min_hits = min_hits  # Increased from 1 to reduce false IDs
        self.iou_threshold = iou_threshold  # Slightly lowered for better matching
        self.feature_extractor = FeatureExtractor()
    
    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of bboxes [x1, y1, x2, y2]
            frame: Current frame for feature extraction
            
        Returns:
            List of (track_id, bbox) tuples
        """
        # Extract features for all detections in batch (GPU optimization)
        detection_crops = [self._crop_bbox(frame, bbox) for bbox in detections]
        detection_features = self.feature_extractor.extract_features_batch(detection_crops)
        
        # Predict new positions for existing tracks with motion estimation
        for track_id in self.tracks:
            if 'velocity' in self.tracks[track_id]:
                # Simple linear motion prediction
                bbox = self.tracks[track_id]['bbox']
                velocity = self.tracks[track_id]['velocity']
                predicted_bbox = bbox + velocity
                self.tracks[track_id]['predicted_bbox'] = predicted_bbox
            else:
                self.tracks[track_id]['predicted_bbox'] = self.tracks[track_id]['bbox']
        
        # Update existing tracks
        matched_tracks = []
        matched_det_indices = set()
        matched_track_ids = set()
        
        if self.tracks:
            # Build cost matrix for Hungarian algorithm (simplified greedy version)
            matches = []
            
            for det_idx in range(len(detections)):
                best_match = None
                best_score = 0
                
                for track_id, track in self.tracks.items():
                    # Use predicted position for IoU calculation
                    predicted_bbox = track.get('predicted_bbox', track['bbox'])
                    
                    # Calculate IoU with predicted position
                    iou = self._calculate_iou(detections[det_idx], predicted_bbox)
                    
                    # Calculate feature similarity (cosine similarity)
                    feat_sim = self._cosine_similarity(
                        detection_features[det_idx], 
                        track['features']
                    )
                    
                    # Combined score with higher weight on appearance for re-identification
                    # IoU 30%, Feature 70% for better re-ID after occlusion
                    score = 0.3 * iou + 0.7 * feat_sim
                    
                    # Require minimum IoU OR high feature similarity for long-lost tracks
                    if track['time_since_update'] > 5:
                        # For tracks lost for a while, rely more on features
                        min_feat_sim = 0.6
                        if feat_sim > min_feat_sim:
                            score = feat_sim
                    
                    if score > best_score and (iou > self.iou_threshold or feat_sim > 0.65):
                        best_score = score
                        best_match = track_id
                
                if best_match is not None:
                    matches.append((det_idx, best_match, best_score))
            
            # Sort by score and apply greedy matching
            matches.sort(key=lambda x: x[2], reverse=True)
            
            for det_idx, track_id, score in matches:
                if det_idx not in matched_det_indices and track_id not in matched_track_ids:
                    # Calculate velocity (motion estimation)
                    old_bbox = self.tracks[track_id]['bbox']
                    new_bbox = detections[det_idx]
                    velocity = new_bbox - old_bbox
                    
                    # Update matched track
                    self.tracks[track_id]['bbox'] = new_bbox
                    self.tracks[track_id]['velocity'] = velocity
                    # Exponential moving average for features (smoother updates)
                    alpha = 0.3  # Update weight
                    self.tracks[track_id]['features'] = (
                        alpha * detection_features[det_idx] + 
                        (1 - alpha) * self.tracks[track_id]['features']
                    )
                    self.tracks[track_id]['hits'] += 1
                    self.tracks[track_id]['time_since_update'] = 0
                    matched_det_indices.add(det_idx)
                    matched_track_ids.add(track_id)
        
        # Create new tracks for unmatched detections
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.tracks[track_id] = {
                'bbox': detections[det_idx],
                'features': detection_features[det_idx],
                'hits': 1,
                'time_since_update': 0,
                'age': 0
            }
            self.next_id += 1
            matched_track_ids.add(track_id)
        
        # Collect all valid tracks to return
        for track_id in matched_track_ids:
            if self.tracks[track_id]['hits'] >= self.min_hits:
                matched_tracks.append((track_id, self.tracks[track_id]['bbox']))
        
        # Age unmatched tracks
        to_delete = []
        for track_id in self.tracks:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['time_since_update'] += 1
                self.tracks[track_id]['age'] += 1
                
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    to_delete.append(track_id)
        
        # Delete old tracks
        for track_id in to_delete:
            del self.tracks[track_id]
        
        return matched_tracks
    
    def _crop_bbox(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop bounding box from frame."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within frame
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((64, 32, 3), dtype=np.uint8)
        
        crop = frame[y1:y2, x1:x2]
        return crop
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


class TrackTask(Task):
    """Object tracking task using DeepSORT with MobileNet re-identification."""
    
    # Class-level tracker to persist across frames
    _tracker = None
    
    def __init__(self):
        self.frame = None
        
        # Initialize tracker once with settings from config
        if TrackTask._tracker is None:
            # Load settings
            settings = self._load_settings()
            tracker_config = settings.get('app', {}).get('tracker', {})
            
            TrackTask._tracker = DeepSORTTracker(
                max_age=tracker_config.get('max_age', 70),
                min_hits=tracker_config.get('min_hits', 3),
                iou_threshold=tracker_config.get('iou_threshold', 0.25)
            )
            print(f"✓ Tracker initialized: max_age={TrackTask._tracker.max_age}, "
                  f"min_hits={TrackTask._tracker.min_hits}, "
                  f"iou_threshold={TrackTask._tracker.iou_threshold}")
    
    def _load_settings(self):
        """Load settings from JSON file."""
        # Load default settings first
        default_path = 'settings.default.json'
        if os.path.exists(default_path):
            with open(default_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Override with user settings if exists
        user_path = 'settings.json'
        if os.path.exists(user_path):
            with open(user_path, 'r') as f:
                user_settings = json.load(f)
                # Deep merge
                for key in ['app', 'camera', 'event', 'mqtt']:
                    if key in user_settings:
                        if key in settings:
                            if isinstance(settings[key], dict):
                                settings[key].update(user_settings[key])
                        else:
                            settings[key] = user_settings[key]
        
        return settings
    
    def run(self, bag=None):
        """
        Process detections and assign track IDs.
        
        Args:
            bag: List of YOLO detection results
            
        Returns:
            List of (track_id, bbox, gender) tuples
        """
        if not bag:
            return []
        
        # Extract bboxes from YOLO results
        detections = []
        for result in bag:
            bboxes = result.boxes.xyxy.cpu().numpy()
            for box in bboxes:
                detections.append(box)
        
        # Update tracker
        tracked_boxes = TrackTask._tracker.update(detections, self.frame)
        
        # Add gender as "unknown" since we removed gender classification
        tracked_boxes_with_gender = [(track_id, bbox, "unknown") for track_id, bbox in tracked_boxes]
        
        return tracked_boxes_with_gender