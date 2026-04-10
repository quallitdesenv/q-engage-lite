
import cv2

from src.core import Task
from ultralytics import YOLO
from cv2.typing import MatLike

class ShowResultsTask(Task):
    local: bool = True

    def __init__(self):
        self.frame = None
        self.display_enabled = True
        try:
            # Test if display is available
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
        except:
            self.display_enabled = False
            print("⚠ Display not available, running in headless mode")

    def run(self, bag=None):
        if bag and self.local and self.display_enabled:
            for item in bag:
                # Handle both (track_id, box) and (track_id, box, gender) formats
                if len(item) == 2:
                    track_id, box = item
                elif len(item) >= 3:
                    track_id, box, *rest = item
                else:
                    continue
                    
                x1, y1, x2, y2 = map(int, box[:4])  # Handle possible extra values
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Tracker Results', self.frame)

        return bag