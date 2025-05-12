import cv2
import numpy as np

def get_color_by_track(track):
    return track.color if hasattr(track, 'color') else (0, 255, 0)

def draw_detections(frame, detections, tracks, projector, cam_id):
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        base = (x + w // 2, y + h)
        bev_pt = projector.image_to_bev(cam_id, base)

        # Buscar el track más cercano en esta cámara
        closest_trk = None
        min_dist = float('inf')
        
        for trk in tracks:
            if cam_id in getattr(trk, 'last_cameras', set()):
                dist = np.linalg.norm(trk.kf.x[:2] - np.array(bev_pt))
                if dist < min_dist and dist < 1.0:  # 1 metro de umbral
                    min_dist = dist
                    closest_trk = trk

        if closest_trk:
            color = get_color_by_track(closest_trk)
            cv2.putText(
                frame, f"ID:{closest_trk.id} (Cam:{cam_id})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return frame

def draw_trajectories(frame, tracks, projector, cam_id):
    for trk in tracks:
        if cam_id in getattr(trk, 'last_cameras', set()):
            color = get_color_by_track(trk)
            pts = [
                projector.bev_to_image(cam_id, (p[0], p[1]))
                for p in trk.history
            ]
            for i in range(1, len(pts)):
                if pts[i-1] and pts[i]:
                    cv2.line(frame, pts[i-1], pts[i], color, 2)
    
    return frame