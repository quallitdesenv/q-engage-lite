from cv2.typing import MatLike
from src.core import Task, GenderClassificator
from src.core.utils.memory_storage import MemoryStorage


class GenderClassificationTask(Task):
    def __init__(self, frame: MatLike, model_path: str):
        self.frame = frame
        self.classificator = GenderClassificator(model_path)
        self.genders = {}

    def run(self, bag=None):
        if bag:
            new_bag = []
            classificator = self.classificator
            for track_id, box in bag:
                x1, y1, x2, y2 = map(int, box[:4])  # Handle possible extra values
                person_img = self.frame[y1:y2, x1:x2]

                if track_id in self.genders:
                    gender = self.genders[track_id]
                else:
                    gender = classificator.predict(person_img)
                    self.genders[track_id] = gender

                new_bag.append((track_id, box, gender))

            return new_bag

        return bag