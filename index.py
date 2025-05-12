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

# 1) Carga de configuración
dir_cfg = 'config.json'
config = json.load(open(dir_cfg))

# 2) Inicializar cámaras (asegúrate de que para la webcam local uses un entero 0, no "0")
cams = [
    RTSPCamera(c['id'], c['rtsp_url']).start()
    for c in config['cameras']
]

detector = YOLOv11Detector()
projector = BEVProjector(dir_cfg)
tracker = MultiCameraTracker(
    config['tracker']['max_age'],
    config['tracker']['dist_threshold']
)
logger = CSVLogger('trajectories.csv')

# 3) Crear lienzo BEV
bev_h, bev_w = 500, 500
bev_canvas = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 255

try:
    while True:
        frames = []
        all_dets = []

        # 4) Leer cada cámara y detectar
        for cam in cams:
            frame = cam.read()
            if frame is None:
                # placeholder negro si no llega frame
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frames.append((cam.cam_id, frame))

            # Detector + proyección BEV
            dets_img = detector.detect(frame)
            for d in dets_img:
                x, y, w, h = d['bbox']
                base = (x + w // 2, y + h)
                bev_pt = projector.image_to_bev(cam.cam_id, base)
                all_dets.append((bev_pt[0], bev_pt[1], cam.cam_id))

        # 5) Actualizar tracker global
        tracks = tracker.update(all_dets)

        # 6) Dibujar BEV unificado
        bev_canvas[:] = 255
        for trk in tracks:
            if trk.history:
                pts = np.array([
                    (int(x * 100 + bev_w/2), int(bev_h - y * 100))
                    for x, y in trk.history
                ], dtype=np.int32)
                cv2.polylines(bev_canvas, [pts], False, (0,0,255), 2)
                cx, cy = pts[-1]
                cv2.circle(bev_canvas, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(bev_canvas, f"ID:{trk.id}",
                            (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0,0,0), 1)

        # 7) Visualizar detecciones + trayectorias en cada frame
        vis_frames = []
        for cam_id, frame in frames:
            dets = detector.detect(frame)
            vis = draw_detections(frame.copy(), dets, tracks, projector, cam_id)
            vis = draw_trajectories(vis, tracks, projector, cam_id)
            vis_frames.append(cv2.resize(vis, (320, 240)))

        # 8) Combinar todos los frames en una sola imagen
        combined = cv2.hconcat(vis_frames)

        # 9) Mostrar ventanas
        cv2.imshow('Multi-Camera Tracking', combined)
        cv2.imshow('Bird Eye View (BEV)', bev_canvas)

        # 10) Registrar última posición de cada track
        for trk in tracks:
            if trk.history:
                logger.log(trk.id, trk.history[-1], cam_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

finally:
    for cam in cams:
        cam.stop()
    cv2.destroyAllWindows()
