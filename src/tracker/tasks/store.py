from src.core import Task, MemoryStorage

class StoreTask(Task):
    def run(self, bag=None):
        for track_id, box, gender in bag:
            x1, y1, x2, y2 = map(int, box[:4])
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            MemoryStorage.save('tracks', track_id, (x_center, y_center))
            MemoryStorage.save('genders', track_id, gender)

        return bag