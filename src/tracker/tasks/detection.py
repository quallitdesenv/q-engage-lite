
from src.core import Task
from ultralytics import YOLO
from cv2.typing import MatLike

class DetectionTask(Task):
    def __init__(self, model: YOLO, frame: MatLike):
        self.model = model
        self.frame = frame

    def run(self, bag=None):
        results = self.model(self.frame, conf=0.25, cls=0)
        return results