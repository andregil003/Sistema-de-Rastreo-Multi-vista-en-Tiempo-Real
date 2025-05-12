import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, path):
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'track_id','timestamp','x_bev','y_bev','cam_id'
        ])

    def log(self, track_id, pos, cam_id):
        ts = datetime.utcnow().isoformat()
        self.writer.writerow([
            track_id, ts, pos[0], pos[1], cam_id
        ])
        self.file.flush()
