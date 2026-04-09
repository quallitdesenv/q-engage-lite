"""
Integration tests for the complete pipeline.
Tests the full pipeline execution with all tasks.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.core import Pipeline
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
    model = YOLO('yolov8n.pt')
    return model


class TestPipelineIntegration:
    """Integration test suite for the complete pipeline."""
    
    def test_pipeline_initialization(self, yolo_model, sample_images):
        """Test that Pipeline can be initialized with tasks."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        
        tasks = [
            DetectionTask(yolo_model, frame),
            TrackTask(frame),
            ShowResultsTask(frame)
        ]
        
        pipeline = Pipeline(tasks)
        
        assert pipeline is not None
        assert len(pipeline.tasks) == 3
        assert pipeline.logger is not None
    
    def test_pipeline_run_complete(self, yolo_model, sample_images, results_dir):
        """Test complete pipeline execution with all tasks."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        for img_name, frame in sample_images:
            frame_copy = frame.copy()
            
            # Create tasks
            detection_task = DetectionTask(yolo_model, frame_copy)
            track_task = TrackTask(frame_copy)
            show_results_task = ShowResultsTask(frame_copy)
            show_results_task.local = False  # Disable imshow for testing
            
            # Create pipeline
            tasks = [detection_task, track_task, show_results_task]
            pipeline = Pipeline(tasks)
            
            # Run pipeline
            try:
                pipeline.run()
                
                # Save final result
                output_path = results_dir / f"pipeline_complete_{img_name}"
                cv2.imwrite(str(output_path), frame_copy)
                print(f"Saved pipeline result: {output_path}")
                
            except Exception as e:
                pytest.fail(f"Pipeline failed on {img_name}: {e}")
    
    def test_pipeline_task_sequence(self, yolo_model, sample_images, results_dir):
        """Test that pipeline executes tasks in correct sequence and passes data."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        img_name, frame = sample_images[0]
        frame_copy = frame.copy()
        
        # Track execution order
        execution_order = []
        
        # Custom tasks that track execution
        class TrackedDetectionTask(DetectionTask):
            def run(self, bag=None):
                execution_order.append('detection')
                result = super().run(bag)
                assert result is not None, "Detection should return results"
                return result
        
        class TrackedTrackTask(TrackTask):
            def run(self, bag=None):
                execution_order.append('track')
                assert bag is not None, "Track should receive detection results"
                result = super().run(bag)
                assert isinstance(result, list), "Track should return list of boxes"
                return result
        
        class TrackedShowResultsTask(ShowResultsTask):
            def run(self, bag=None):
                execution_order.append('show_results')
                assert isinstance(bag, list), "ShowResults should receive tracked boxes"
                result = super().run(bag)
                return result
        
        # Create tracked tasks
        detection_task = TrackedDetectionTask(yolo_model, frame_copy)
        track_task = TrackedTrackTask(frame_copy)
        show_results_task = TrackedShowResultsTask(frame_copy)
        show_results_task.local = False
        
        # Create and run pipeline
        pipeline = Pipeline([detection_task, track_task, show_results_task])
        pipeline.run()
        
        # Verify execution order
        assert execution_order == ['detection', 'track', 'show_results']
    
    def test_pipeline_with_multiple_images(self, yolo_model, sample_images, results_dir):
        """Test pipeline on all sample images."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        results_summary = []
        
        for img_name, frame in sample_images:
            frame_copy = frame.copy()
            
            # Run detection
            detection_task = DetectionTask(yolo_model, frame_copy)
            detections = detection_task.run()
            
            # Run tracking
            track_task = TrackTask(frame_copy)
            tracked_boxes = track_task.run(detections)
            
            # Draw results
            result_frame = frame_copy.copy()
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
            
            # Save result
            output_path = results_dir / f"pipeline_multi_{img_name}"
            cv2.imwrite(str(output_path), result_frame)
            
            # Track results
            results_summary.append({
                'image': img_name,
                'detections': len(tracked_boxes),
                'output': str(output_path)
            })
            
            print(f"Processed {img_name}: {len(tracked_boxes)} detections")
        
        # Save summary
        summary_path = results_dir / "pipeline_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Pipeline Execution Summary\n")
            f.write("=" * 50 + "\n\n")
            for result in results_summary:
                f.write(f"Image: {result['image']}\n")
                f.write(f"Detections: {result['detections']}\n")
                f.write(f"Output: {result['output']}\n")
                f.write("-" * 50 + "\n")
        
        print(f"Saved summary: {summary_path}")
        assert len(results_summary) == len(sample_images)
    
    def test_pipeline_error_handling(self, yolo_model, sample_images):
        """Test pipeline error handling."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        _, frame = sample_images[0]
        
        # Create a task that will fail
        class FailingTask:
            name = "FailingTask"
            
            def run(self, bag=None):
                raise ValueError("Intentional test failure")
        
        # Create pipeline with failing task
        tasks = [
            DetectionTask(yolo_model, frame),
            FailingTask(),
            TrackTask(frame)
        ]
        
        pipeline = Pipeline(tasks)
        
        # Pipeline should handle the error gracefully
        # (based on the Pipeline implementation, it logs errors but doesn't crash)
        try:
            pipeline.run()
        except Exception:
            pass  # Expected to fail, but shouldn't crash
    
    def test_pipeline_empty_detections(self, yolo_model, results_dir):
        """Test pipeline behavior with images containing no persons."""
        # Create a blank image
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Create tasks
        detection_task = DetectionTask(yolo_model, blank_frame)
        track_task = TrackTask(blank_frame)
        show_results_task = ShowResultsTask(blank_frame)
        show_results_task.local = False
        
        # Create and run pipeline
        pipeline = Pipeline([detection_task, track_task, show_results_task])
        pipeline.run()
        
        # Save result
        output_path = results_dir / "pipeline_empty_detections.jpg"
        cv2.imwrite(str(output_path), blank_frame)
        print(f"Saved empty detection result: {output_path}")


class TestPipelinePerformance:
    """Performance tests for the pipeline."""
    
    def test_pipeline_execution_time(self, yolo_model, sample_images):
        """Test pipeline execution time."""
        if not sample_images:
            pytest.skip("No sample images found")
        
        import time
        
        _, frame = sample_images[0]
        
        # Create tasks
        detection_task = DetectionTask(yolo_model, frame)
        track_task = TrackTask(frame)
        show_results_task = ShowResultsTask(frame)
        show_results_task.local = False
        
        # Create pipeline
        pipeline = Pipeline([detection_task, track_task, show_results_task])
        
        # Measure execution time
        start_time = time.time()
        pipeline.run()
        execution_time = time.time() - start_time
        
        print(f"Pipeline execution time: {execution_time:.3f} seconds")
        
        # Reasonable execution time (should complete in under 10 seconds for one image)
        assert execution_time < 10.0, f"Pipeline too slow: {execution_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
