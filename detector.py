from ultralytics import YOLO

class YOLOv11Detector:
    def __init__(self, model_path='yolo11n.pt', threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0 and conf >= self.threshold:  # Clase 0 = 'person'
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'score': float(conf)
                })
        return detections