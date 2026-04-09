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
from src.tracker.tasks.gender_classification import GenderClassificationTask
from .tasks import DetectionTask, TrackTask, ShowResultsTask, StoreTask
import json
from datetime import datetime

def process_detections(frame: MatLike, model: YOLO, classifier_path):
    pipeline = Pipeline([
        DetectionTask(model, frame),
        TrackTask(frame),
        GenderClassificationTask(frame, classifier_path),
        StoreTask(),
        ShowResultsTask(frame)
    ])

    pipeline.run()

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
            }
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

def store_batch(camera_id: str, fnum: int, timestamp: int):
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

    with open(f'./tmp/batch_{camera_id}_{fnum}_{timestamp}.json', 'w') as f:
        json.dump(batch, f, indent=4)
    print(f"Batch stored: camera={camera_id} frame={fnum} time={timestamp} objects={len(batch['frames'][0]['objects'])}")

def main():
    """
    Main entry point for the tracker module.
    Initializes AI model and processes video stream for edge detection and tracking.
    """
    settings = get_settings()
    print(f"{settings['app']['name']} v{settings['app']['version']}")
    print(settings['app']['description'])
    print("Initializing AI Edge Detector...")

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
        timelapse_seconds = 4
        dtime = datetime.now()  # Inicializar antes do loop

        while connector.isOpened():
            i += 1

            if i % frame_rate != 0:
                connector.read()
                continue

            ret, frame = connector.read()

            if not ret:
                break

            matrix = connector.to_matrix(frame)

            img = cv2.resize(matrix, (width, height))

            process_detections(img, model, settings['app']['classifier']['source'])
            
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
        sys.exit(1)

if __name__ == "__main__":
    main()