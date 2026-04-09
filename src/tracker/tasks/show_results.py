
import cv2

from src.core import Task
from ultralytics import YOLO
from cv2.typing import MatLike

class ShowResultsTask(Task):
    local: bool = True

    def __init__(self, frame: MatLike):
        self.frame = frame

    def run(self, bag=None):
        if bag and self.local:
            for track_id, box, gender in bag:
                x1, y1, x2, y2 = map(int, box[:4])  # Handle possible extra values
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(self.frame, f'Gender: {gender}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Tracker Results', self.frame)
            cv2.waitKey(1)

        return bag