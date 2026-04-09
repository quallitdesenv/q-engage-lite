import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from pathlib import Path
from typing import Dict, Tuple
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


class DeepSORTTracker:
    """Simple DeepSORT tracker implementation."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
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
        # Extract features for all detections
        detection_features = []
        for bbox in detections:
            crop = self._crop_bbox(frame, bbox)
            features = self.feature_extractor.extract_features(crop)
            detection_features.append(features)
        
        # Update existing tracks
        matched_tracks = []
        matched_det_indices = set()
        matched_track_ids = set()
        
        if self.tracks:
            # Simple matching based on IoU and feature similarity
            for det_idx in range(len(detections)):
                best_match = None
                best_score = 0
                
                for track_id, track in self.tracks.items():
                    if track['time_since_update'] > 0:
                        continue
                    
                    # Calculate IoU
                    iou = self._calculate_iou(detections[det_idx], track['bbox'])
                    
                    # Calculate feature similarity (cosine similarity)
                    feat_sim = self._cosine_similarity(
                        detection_features[det_idx], 
                        track['features']
                    )
                    
                    # Combined score
                    score = 0.5 * iou + 0.5 * feat_sim
                    
                    if score > best_score and iou > self.iou_threshold:
                        best_score = score
                        best_match = track_id
                
                if best_match is not None:
                    # Update matched track
                    self.tracks[best_match]['bbox'] = detections[det_idx]
                    self.tracks[best_match]['features'] = detection_features[det_idx]
                    self.tracks[best_match]['hits'] += 1
                    self.tracks[best_match]['time_since_update'] = 0
                    matched_det_indices.add(det_idx)
                    matched_track_ids.add(best_match)
        
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
    
    def __init__(self, frame):
        self.frame = frame
        
        # Initialize tracker once
        if TrackTask._tracker is None:
            TrackTask._tracker = DeepSORTTracker(
                max_age=30,
                min_hits=1,
                iou_threshold=0.3
            )
    
    def run(self, bag=None):
        """
        Process detections and assign track IDs.
        
        Args:
            bag: List of YOLO detection results
            
        Returns:
            List of (track_id, bbox) tuples
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
        
        return tracked_boxes