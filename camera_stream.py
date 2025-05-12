import cv2
import threading
import json


class RTSPCamera:
    def __init__(self, cam_id, source):
        self.cam_id = cam_id
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()


def load_cameras_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    cameras = []
    for cam_cfg in config['cameras']:
        cam = RTSPCamera(cam_cfg['id'], cam_cfg['source']).start()
        cameras.append(cam)

    return cameras
