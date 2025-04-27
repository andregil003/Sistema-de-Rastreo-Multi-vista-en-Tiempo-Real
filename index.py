import cv2
import time
import csv
import numpy as np
from detector import YOLOv11Detector
from tracker import Tracker
from bev_projection import BEVProjector

cap = cv2.VideoCapture(0)

detector = YOLOv11Detector()
tracker = Tracker()
projector = BEVProjector('config.json')
CAM_ID = 1

with open('output_tracks.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['track_id', 'timestamp', 'x_bev', 'y_bev'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        projected_detections = []

        for d in detections:
            x, y, w, h = d['bbox']
            #base_point = (x + w // 2, y + h)  # punto base de la persona (pies)
            base_point = (x + w // 2, y + h // 2)  # Ahora usamos el centro de la persona

            bev_point = projector.image_to_bev(CAM_ID, base_point)
            projected_detections.append({'bev': bev_point, 'bbox': d['bbox']})

        tracks = tracker.update([p['bev'] for p in projected_detections])

        # Dibujo sobre el video original
        for d in detections:
            x, y, w, h = d['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for trk in tracks:
            if trk.history:
                x, y = trk.kf.x[:2]
                cv2.putText(frame, f"ID: {trk.id}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                writer.writerow([trk.id, time.time(), float(x), float(y)])

        # Crear vista cenital (BEV)
        bev_image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # fondo blanco
        for trk in tracks:
            if trk.history:
                pts = np.array([(int(x * 100 + 250), int(500 - y * 100)) for x, y in trk.history], dtype=np.int32)
                cv2.polylines(bev_image, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                x, y = pts[-1]
                cv2.circle(bev_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(bev_image, f"ID:{trk.id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        for trk in tracks:
            for pt in trk.history:
                x, y = int(pt[0]*100), int(pt[1]*100)
                cv2.circle(bev_image, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(bev_image, f"{x},{y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)


        cv2.imshow('Tracking with BEV', frame)
        cv2.imshow('Bird Eye View', bev_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
