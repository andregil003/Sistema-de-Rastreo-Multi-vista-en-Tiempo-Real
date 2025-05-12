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

# === Cargar configuración ===
dir_cfg = 'config_calibrated.json'
config = json.load(open(dir_cfg))

# === Inicializar cámaras ===
cams = [RTSPCamera(c['id'], c['rtsp_url']).start() for c in config['cameras']]

# === Inicializar componentes ===
detector = YOLOv11Detector()
projector = BEVProjector(dir_cfg)
tracker = MultiCameraTracker(
    max_age=config['tracker']['max_age'],
    dist_threshold=1.5,            # más permisivo que 1.0
    camera_overlap_threshold=1.2   # nuevo parámetro para control de penalización
)

logger = CSVLogger('trajectories_multi.csv')
# === Configuración BEV ===
bev_h, bev_w = 800, 800
PX_PER_METER = 100
bev_canvas = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 255

def bev_to_canvas_coords(x, y):
    return (
        int(x * PX_PER_METER + bev_w / 2),
        int(bev_h / 2 - y * PX_PER_METER)
    )

def get_color_by_id(track_id):
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))

try:
    while True:
        frames = []
        all_dets = []
        dets_by_cam = {}

        # === Leer y detectar en cada cámara ===
        for cam in cams:
            frame = cam.read()
            if frame is None:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frames.append((cam.cam_id, frame))

            detections = detector.detect(frame)
            dets_by_cam[cam.cam_id] = detections

            for d in detections:
                x, y, w, h = d['bbox']
                base = (x + w // 2, y + h)
                bev_pt = projector.image_to_bev(cam.cam_id, base)
                all_dets.append((bev_pt[0], bev_pt[1], cam.cam_id))

        # === Tracking global ===
        tracks = tracker.update(all_dets)

        # === Dibujar BEV solo con tracks confirmados ===
        bev_canvas[:] = 255
        for trk in tracks:
            if len(trk.history) > 5:
                color = get_color_by_id(trk.id)
                pts = np.array([
                    bev_to_canvas_coords(x, y)
                    for x, y in trk.history
                ], dtype=np.int32)
                cv2.polylines(bev_canvas, [pts], False, color, 2)
                cx, cy = bev_to_canvas_coords(*trk.history[-1])
                cv2.circle(bev_canvas, (cx, cy), 5, color, -1)
                cv2.putText(bev_canvas, f"ID:{trk.id}", (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # === Dibujar cada cámara con detecciones y trayectorias ===
        vis_frames = []
        for cam_id, frame in frames:
            dets = dets_by_cam[cam_id]
            vis = draw_detections(frame.copy(), dets, tracks, projector, cam_id)
            vis = draw_trajectories(vis, tracks, projector, cam_id)
            vis_frames.append(cv2.resize(vis, (320, 240)))

        # === Mostrar interfaz ===
        combined = cv2.hconcat(vis_frames)
        cv2.imshow('Multi-Camera Tracking', combined)
        cv2.imshow('Bird Eye View (BEV)', bev_canvas)

        # === Guardar última posición solo de tracks confirmados ===
        for trk in tracks:
            if len(trk.history) > 5:
                logger.log(trk.id, trk.history[-1], trk.cam_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

finally:
    for cam in cams:
        cam.stop()
    cv2.destroyAllWindows()
