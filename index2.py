import time
import json
import cv2
import numpy as np
from camera_stream import RTSPCamera
from detector import YOLOv11Detector
from bev_projection import BEVProjector
from tracker import MultiCameraTracker
from visualization import draw_detections, draw_trajectories
from logger import CSVLogger

# === CONFIGURACIÓN INICIAL ===
dir_cfg = 'config.json'
config = json.load(open(dir_cfg))

camera_cfg = [c for c in config['cameras'] if c['id'] == 3][0]
cam = RTSPCamera(camera_cfg['id'], camera_cfg['rtsp_url']).start()

detector = YOLOv11Detector()
projector = BEVProjector(dir_cfg)
tracker = MultiCameraTracker(
    config['tracker']['max_age'],
    config['tracker']['dist_threshold']
)
logger = CSVLogger('trajectories_cam3.csv')

# === VISTA BEV AMPLIADA ===
bev_h, bev_w = 800, 800
PX_PER_METER = 100  # puedes bajar a 80 para más cobertura
bev_canvas = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 255

def bev_to_canvas_coords(x, y):
    """Transforma BEV (x,y) en píxeles para el lienzo"""
    return (
        int(x * PX_PER_METER + bev_w / 2),
        int(bev_h / 2 - y * PX_PER_METER)
    )

try:
    while True:
        frame = cam.read()
        if frame is None:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # === DETECCIÓN UNA SOLA VEZ ===
        detections = detector.detect(frame)

        # === PROYECCIÓN A BEV ===
        all_dets = []
        for d in detections:
            x, y, w, h = d['bbox']
            base = (x + w // 2, y + h)
            bev_pt = projector.image_to_bev(cam.cam_id, base)
            all_dets.append((bev_pt[0], bev_pt[1], cam.cam_id))

        # === TRACKING ===
        tracks = tracker.update(all_dets)

        # === DIBUJAR EN VISTA BEV ===
        bev_canvas[:] = 255
        for trk in tracks:
            if trk.history:
                pts = np.array([
                    bev_to_canvas_coords(x, y)
                    for x, y in trk.history
                ], dtype=np.int32)

                cv2.polylines(bev_canvas, [pts], False, (0, 0, 255), 2)

                cx, cy = bev_to_canvas_coords(*trk.history[-1])
                cv2.circle(bev_canvas, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(bev_canvas, f"ID:{trk.id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # === VISUALIZACIÓN FRAME ORIGINAL ===
        vis_frame = draw_detections(frame.copy(), detections, tracks, projector, cam.cam_id)
        vis_frame = draw_trajectories(vis_frame, tracks, projector, cam.cam_id)
        vis_frame = cv2.resize(vis_frame, (640, 480))

        # === MOSTRAR VENTANAS ===
        cv2.imshow('Tracking Cam 3', vis_frame)
        cv2.imshow('BEV View', bev_canvas)

        # === LOGEO DE ÚLTIMAS POSICIONES ===
        for trk in tracks:
            if trk.history:
                logger.log(trk.id, trk.history[-1], cam.cam_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

finally:
    cam.stop()
    cv2.destroyAllWindows()
