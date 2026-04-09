"""
Q-Engage Lite - Tracker Module
AI-powered edge detection and object tracking using YOLO (Ultralytics)
"""
import os
import sys
import cv2
from cv2.typing import MatLike
from ultralytics import YOLO
import numpy as np
from src.core import Pipeline, Connector
from src.core.utils.memory_storage import MemoryStorage
from .tasks import DetectionTask, TrackTask, ShowResultsTask, StoreTask
import json
from datetime import datetime
import threading
import queue


def get_settings():
    filename = 'settings.json' if os.path.exists('settings.json') else 'settings.default.json'
    with open(filename, 'r') as f:
        settings = json.load(f)

    return {
        "app": {
            "name": settings.get("app",{}).get("name", "Q-Engage Lite - Tracker Module"),
            "version": settings.get("app",{}).get("version", "1.0.0"),
            "description": settings.get("app",{}).get("description", "AI-powered edge detection and object tracking using YOLO (Ultralytics)"),
            "author": settings.get("app",{}).get("author", "Quallity Solutions"),
            "license": settings.get("app",{}).get("license", "MIT"),
            "model": {
                "name": settings.get("app", {}).get("model", {}).get("name", "ssci_v2"),
                "version": settings.get("app", {}).get("model", {}).get("version", "2.0"),
                "description": settings.get("app", {}).get("model", {}).get("description", "State-of-the-art model for accurate people counting and demographic analysis."),
                "pretrained": settings.get("app", {}).get("model", {}).get("pretrained", True),
                "source": settings.get("app", {}).get("model", {}).get("source", "./pretrained/ssci_v2.pt")
            },
            "classifier": {
                "name": settings.get("app", {}).get("classifier", {}).get("name", "gender_classifier"),
                "version": settings.get("app", {}).get("classifier", {}).get("version", "1.0"),
                "description": settings.get("app", {}).get("classifier", {}).get("description", "A simple gender classification model based on a lightweight CNN architecture."),
                "pretrained": settings.get("app", {}).get("classifier", {}).get("pretrained", True),
                "source": settings.get("app", {}).get("classifier", {}).get("source", "./pretrained/gender_classifier_v1.pt")
            },
            "event": {
                "driver": settings.get("event",{}).get("driver", "local"),
                "path": settings.get("event",{}).get("path", "./tmp/events")
            },
            "store": {
                "timelapse_seconds": settings.get("app",{}).get("store",{}).get("timelapse_seconds", 10)
            },
        },
        "camera": {
            "id": settings.get("camera",{}).get("id", "1"),
            "resolution": settings.get("camera",{}).get("resolution", "1080p"),
            "frame_rate": settings.get("camera",{}).get("frame_rate", 30),
            "night_vision": settings.get("camera",{}).get("night_vision", True),
            "motion_detection": settings.get("camera",{}).get("motion_detection", True),
            "rtsp_url": settings.get("camera",{}).get("rtsp_url", ""),
            "gstream": settings.get("camera",{}).get("gstream", False)
        }
    }

# Global queue for async batch storage
batch_queue = queue.Queue(maxsize=100)
storage_thread = None

def batch_storage_worker():
    """Worker thread that handles async batch storage to disk."""
    while True:
        try:
            item = batch_queue.get()
            if item is None:  # Poison pill to stop the worker
                break
            
            camera_id, fnum, timestamp, batch = item
            
            # Write to disk (non-blocking for main thread)
            filepath = f'./tmp/batch_{camera_id}_{fnum}_{timestamp}.json'
            with open(filepath, 'w') as f:
                json.dump(batch, f, indent=4)
            
            batch_queue.task_done()
        except Exception as e:
            print(f"Error in storage worker: {e}", file=sys.stderr)

def store_batch(camera_id: str, fnum: int, timestamp: int):
    """Store batch asynchronously - non-blocking operation."""
    batch = {
        "camera": {
            "id": camera_id
        },
        "frames": [
            {
                "framenumber": fnum,
                "time": timestamp,
                "objects": [
                    {
                        "track_id": track_id,
                        "type": "PERSON",
                        "gender": gender.upper(),
                        "position": position
                    }
                ]
            } for track_id, position, gender in MemoryStorage.slots
        ]
    }
    MemoryStorage.slots.clear()

    MemoryStorage.save_batch(timestamp, batch)

    # Add to async queue instead of blocking write
    try:
        batch_queue.put_nowait((camera_id, fnum, timestamp, batch))
        print(f"Batch queued: camera={camera_id} frame={fnum} time={timestamp} objects={len(batch['frames'][0]['objects'])}")
    except queue.Full:
        print(f"Warning: Batch queue full, dropping batch {fnum}", file=sys.stderr)

def main():
    """
    Main entry point for the tracker module.
    Initializes AI model and processes video stream for edge detection and tracking.
    """
    global storage_thread
    
    settings = get_settings()
    print(f"{settings['app']['name']} v{settings['app']['version']}")
    print(settings['app']['description'])
    print("Initializing AI Edge Detector...")
    
    # Start async storage worker thread
    storage_thread = threading.Thread(target=batch_storage_worker, daemon=True)
    storage_thread.start()
    print("✓ Async storage worker started")

    try:
        resolution = settings['camera']['resolution'].split('x')
        width, height = int(resolution[0]), int(resolution[1])
        frame_rate = int(float(settings['camera']['frame_rate']) * 10)

        print(f"Loading YOLO model from {settings['app']['model']['source']}...")
        model = YOLO(settings['app']['model']['source'])

        print("✓ YOLO model loaded successfully")
        print("Tracker module ready. Waiting for video stream...")
        print("Press Ctrl+C to stop")

        connector = Connector(
            type = Connector.StreamType.GSTREAMER if settings['camera']['gstream'] else Connector.StreamType.FFMPEG
        )
        connector.connect(settings['camera']['rtsp_url'])
        i = 0
        timelapse_seconds = settings['camera']['frame_rate'] * 10
        dtime = datetime.now()

        pipeline = Pipeline([
            DetectionTask,
            TrackTask,
            StoreTask,
            ShowResultsTask
        ])

        while connector.isOpened():
            i += 1

            if i % frame_rate != 0:
                connector.read()
                continue

            ret, frame = connector.read()

            if not ret:
                break

            img = cv2.resize(frame, (width, height))

            pipeline(img, model)

            for track_id, position in MemoryStorage.all('tracks'):
                gender = MemoryStorage.load('genders', track_id)
                MemoryStorage.save_slot((track_id, position, gender))

            if (datetime.now() - dtime).total_seconds() >= timelapse_seconds:
                store_batch(settings['camera']["id"], i, int(datetime.now().timestamp()))
                dtime = datetime.now()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                connector.release()
                break

    except KeyboardInterrupt:
        print("\nShutting down tracker module...")
    except Exception as e:
        line = sys.exc_info()[-1].tb_lineno
        file = sys.exc_info()[-1].tb_frame.f_code.co_filename
        print(f"Error at file {file} line {line}: {e}", file=sys.stderr)
    finally:
        # Cleanup async storage thread
        print("Waiting for pending batches to be written...")
        batch_queue.join()  # Wait for all pending items to be processed
        batch_queue.put(None)  # Send poison pill
        if storage_thread:
            storage_thread.join(timeout=5)
        print("✓ Storage worker stopped")

if __name__ == "__main__":
    main()