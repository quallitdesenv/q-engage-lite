"""
Q-Engage Lite - Tracker Module
AI-powered edge detection and object tracking using YOLO (Ultralytics)
"""
import os
import sys
import cv2
from cv2.typing import MatLike
import requests
from ultralytics import YOLO
import numpy as np
from src.core import Pipeline, Connector
from src.core.utils.memory_storage import MemoryStorage
from .tasks import DetectionTask, TrackTask, ShowResultsTask, StoreTask
import json
from datetime import datetime
import time
from .repositories import DetectionRepositoryInterface, container


def get_settings():
    # Load default settings first
    with open('settings.default.json', 'r') as f:
        default_settings = json.load(f)
    
    # Override with settings.json if exists
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as f:
            user_settings = json.load(f)
        # Deep merge
        for key in ['app', 'camera', 'event', 'mqtt']:
            if key in user_settings:
                if key in default_settings:
                    default_settings[key].update(user_settings[key])
                else:
                    default_settings[key] = user_settings[key]
    
    return default_settings


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
        
        rtsp_url = settings['camera']['rtsp_url']
        
        if not rtsp_url:
            print("✗ ERROR: rtsp_url is empty. Check settings.json or settings.default.json")
            return
        
        # Convert relative paths to absolute
        if not rtsp_url.startswith(('http://', 'https://', 'rtsp://', 'rtsps://')):
            rtsp_url = os.path.abspath(rtsp_url)
        
        print(f"Connecting to: {rtsp_url}")
        connector.connect(rtsp_url)
        
        if not connector.isOpened():
            print("✗ Failed to connect to video stream")
            return
        
        print("✓ Connected to video stream")
        
        i = 0
        timelapse_seconds = settings['camera']['frame_rate'] * 10
        dtime = datetime.now()
        frametimer = time.time()
        repo: DetectionRepositoryInterface = container.get(DetectionRepositoryInterface)

        pipeline = Pipeline([
            DetectionTask,
            TrackTask,
            StoreTask,
            ShowResultsTask
        ])
        
        desired_fps = 2
        frame_interval = 1 / desired_fps
        last_time = time.time()
        frame_count = 0  # Total frames read from source
        
        # Check if source is a file (not a stream)
        is_file_source = not rtsp_url.startswith(('rtsp://', 'rtsps://', 'http://', 'https://'))
        print(f"Source type: {'File' if is_file_source else 'Stream'}")

        while connector.isOpened():
            try:
                ret, frame = connector.read()
                frame_count += 1
                
                if not ret or frame is None:
                    print(f"✗ End of stream (total frames read={frame_count}, frames processed={i})")
                    break

                now = time.time()

                # Skip frames based on time interval ONLY for live streams
                # For video files, process every Nth frame instead
                if not is_file_source and (now - last_time) < frame_interval:
                    continue
                
                # For files, skip frames by counter to control processing rate
                if is_file_source and frame_count % 15 != 0:  # Process every 15th frame for files
                    continue

                last_time = now
                i += 1

                img = cv2.resize(frame, (width, height))

                pipeline(img, model)

                if (time.time() - frametimer) >= 0.5:
                    tracks = list(MemoryStorage.all('tracks'))
                    if tracks:
                        print(f"Frame {i}: Detected {len(tracks)} tracks")
                        for track_id, position in tracks:
                            repo.insert((track_id, position))

                    frametimer = time.time()

                # Check for 'q' key press (only if GUI is available)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    connector.release()
                    break
            except Exception as loop_error:
                print(f"✗ Error in main loop: {loop_error}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                break

    except KeyboardInterrupt:
        print("\nShutting down tracker module...")
    except Exception as e:
        line = sys.exc_info()[-1].tb_lineno
        file = sys.exc_info()[-1].tb_frame.f_code.co_filename
        print(f"Error at file {file} line {line}: {e}", file=sys.stderr)
    finally:
        print("✓ Tracker module stopped")


if __name__ == "__main__":
    main()