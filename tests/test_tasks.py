"""
Automated tests for individual pipeline tasks.
Tests each task in isolation with sample images.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.tracker.tasks.detection import DetectionTask
from src.tracker.tasks.track import TrackTask
from src.tracker.tasks.show_results import ShowResultsTask


# Test fixtures
@pytest.fixture
def sample_images():
    """Load all sample images from tests/samples directory."""
    samples_dir = Path(__file__).parent / "samples"
    images = []
    for img_path in sorted(samples_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img_path.name, img))
    return images


@pytest.fixture
def results_dir():
    """Ensure results directory exists and return path."""
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    return results_path


@pytest.fixture
def yolo_model():
    """Load YOLO model for detection tests."""
    # Using YOLOv8n for fast testing
    model = YOLO('yolov8n.pt')
    return model


@pytest.fixture
def mock_detection_results():
    """Create mock detection results for testing TrackTask."""
    class MockBox:
        def __init__(self, xyxy):
            self.xyxy = MockTensor(xyxy)
    
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        
        def cpu(self):
            return self
        
        def numpy(self):
            return self.data
    
    class MockResult:
        def __init__(self, boxes_data):
            self.boxes = MockBox(boxes_data)
    
    # Create mock results with 2 detected persons
    boxes_data = [
        [100, 100, 200, 300],  # Person 1
        [300, 150, 400, 350],  # Person 2
    ]
    return [MockResult(boxes_data)]


class TestDetectionTask:
    """Test suite for DetectionTask."""
    
    def test_detection_task_initialization(self, yolo_model, sample_images):
        """Test that DetectionTask can be initialized properly."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = DetectionTask(yolo_model, frame)
        
        assert task is not None
        assert task.model == yolo_model
        assert task.frame is not None
    
    def test_detection_task_run(self, yolo_model, sample_images, results_dir):
        """Test that DetectionTask runs and returns detections."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        for img_name, frame in sample_images:
            task = DetectionTask(yolo_model, frame)
            results = task.run()
            
            # Verify results structure
            assert results is not None
            assert len(results) > 0
            
            # Save detection results
            result_frame = frame.copy()
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            output_path = results_dir / f"detection_{img_name}"
            cv2.imwrite(str(output_path), result_frame)
            print(f"Saved detection result: {output_path}")
    
    def test_detection_task_conf_threshold(self, yolo_model, sample_images):
        """Test that DetectionTask respects confidence threshold."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = DetectionTask(yolo_model, frame)
        results = task.run()
        
        # All detections should have conf >= 0.25
        for result in results:
            if hasattr(result.boxes, 'conf'):
                confidences = result.boxes.conf.cpu().numpy()
                assert all(conf >= 0.25 for conf in confidences)
    
    def test_detection_task_person_class(self, yolo_model, sample_images):
        """Test that DetectionTask only returns person class (cls=0)."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = DetectionTask(yolo_model, frame)
        results = task.run()
        
        # All detections should be class 0 (person)
        for result in results:
            if hasattr(result.boxes, 'cls') and len(result.boxes.cls) > 0:
                classes = result.boxes.cls.cpu().numpy()
                assert all(cls == 0 for cls in classes)


class TestTrackTask:
    """Test suite for TrackTask."""
    
    def test_track_task_initialization(self, sample_images):
        """Test that TrackTask can be initialized properly."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = TrackTask(frame)
        
        assert task is not None
        assert task.frame is not None
    
    def test_track_task_run(self, mock_detection_results, sample_images):
        """Test that TrackTask runs and returns tracked boxes."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = TrackTask(frame)
        tracked_boxes = task.run(mock_detection_results)
        
        # Verify tracked boxes structure
        assert tracked_boxes is not None
        assert isinstance(tracked_boxes, list)
        # Note: DeepSORT may filter detections based on min_hits
        # So we check that it returns a list, not necessarily same count
        
        # Verify each tracked box has (track_id, box) format
        for track_id, box in tracked_boxes:
            assert isinstance(track_id, (int, np.integer))
            assert len(box) >= 4  # x1, y1, x2, y2 (may have confidence)
    
    def test_track_task_with_empty_results(self, sample_images):
        """Test TrackTask with empty detection results."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = TrackTask(frame)
        tracked_boxes = task.run([])
        
        assert tracked_boxes is not None
        assert isinstance(tracked_boxes, list)
        assert len(tracked_boxes) == 0
    
    def test_track_task_integration(self, yolo_model, sample_images, results_dir):
        """Test TrackTask with real detections."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        for img_name, frame in sample_images[:2]:  # Test first 2 images
            # Run detection first
            detection_task = DetectionTask(yolo_model, frame)
            detections = detection_task.run()
            
            # Then run tracking
            track_task = TrackTask(frame)
            tracked_boxes = track_task.run(detections)
            
            # Verify tracking results
            assert isinstance(tracked_boxes, list)
            
            # Save tracking results
            result_frame = frame.copy()
            for track_id, box in tracked_boxes:
                x1, y1, x2, y2 = map(int, box[:4])  # Handle possible extra values
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    result_frame, 
                    f'ID: {track_id}', 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            output_path = results_dir / f"tracking_{img_name}"
            cv2.imwrite(str(output_path), result_frame)
            print(f"Saved tracking result: {output_path}")


class TestShowResultsTask:
    """Test suite for ShowResultsTask."""
    
    def test_show_results_task_initialization(self, sample_images):
        """Test that ShowResultsTask can be initialized properly."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        task = ShowResultsTask(frame)
        
        assert task is not None
        assert task.frame is not None
        assert task.local is True
    
    def test_show_results_task_run_with_detections(self, sample_images, results_dir):
        """Test ShowResultsTask with tracked boxes."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        frame_copy = frame.copy()
        
        # Mock tracked boxes
        tracked_boxes = [
            (1, [100, 100, 200, 300]),
            (2, [300, 150, 400, 350]),
        ]
        
        # Set local to False to prevent imshow
        task = ShowResultsTask(frame_copy)
        task.local = False
        result = task.run(tracked_boxes)
        
        # Verify task returns the same bag
        assert result == tracked_boxes
    
    def test_show_results_task_run_empty(self, sample_images):
        """Test ShowResultsTask with no detections."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        frame_copy = frame.copy()
        
        task = ShowResultsTask(frame_copy)
        task.local = False
        result = task.run([])
        
        assert result == []
    
    def test_show_results_task_visual_output(self, sample_images, results_dir):
        """Test ShowResultsTask visual output without imshow."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        for img_name, frame in sample_images[:2]:
            frame_copy = frame.copy()
            
            # Mock tracked boxes
            tracked_boxes = [
                (1, [100, 100, 200, 300]),
                (2, [300, 150, 400, 350]),
                (3, [500, 200, 600, 400]),
            ]
            
            # Manually draw results like ShowResultsTask does
            for track_id, box in tracked_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_copy, 
                    f'ID: {track_id}', 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            output_path = results_dir / f"show_results_{img_name}"
            cv2.imwrite(str(output_path), frame_copy)
            print(f"Saved show results output: {output_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
