import sqlite3
import json
from pathlib import Path
from .detection_repository_interface import DetectionRepositoryInterface

class DetectionRepository(DetectionRepositoryInterface):
    def __init__(self, db_path="./db/detection_db.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                track_position TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def insert(self, detection):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        track_id, track_position = detection
        track_position_json = json.dumps(track_position) if not isinstance(track_position, str) else track_position
        cursor.execute('INSERT INTO detections (track_id, track_position) VALUES (?, ?)', (track_id, track_position_json))
        conn.commit()
        conn.close()

    def getall(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, track_id, track_position, created_at FROM detections')
        rows = cursor.fetchall()
        conn.close()
        return [{'id': row[0], 'track_id': row[1], 'track_position': json.loads(row[2]), 'created_at': row[3]} for row in rows]
