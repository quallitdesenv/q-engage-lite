from src.core import Task, MemoryStorage

class StoreTask(Task):
    def run(self, bag=None):
        if not bag:
            return bag
            
        for item in bag:
            # Handle both (track_id, box) and (track_id, box, gender) formats
            if len(item) == 2:
                track_id, box = item
            elif len(item) >= 3:
                track_id, box, *rest = item
            else:
                continue
                
            x1, y1, x2, y2 = map(int, box[:4])
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            MemoryStorage.save('tracks', track_id, (x_center, y_center))

        return bag