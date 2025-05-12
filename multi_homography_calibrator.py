import cv2
import numpy as np
import json
from ultralytics import YOLO

# === CONFIGURACIÓN ===
IP_CAMERAS = [
    {"id": 1, "url": "http://192.168.0.5:4747/video"},
    {"id": 2, "url": "http://192.168.0.6:4747/video"}
]

REAL_POINTS = [
    [0, 0],
    [2, 0],
    [2, 1],
    [0, 1]
]

MODEL_PATH = 'yolo11n.pt'

def get_detection_center(model, frame):
    results = model(frame)[0]
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0 and conf > 0.5:  # Clase persona
            x1, y1, x2, y2 = map(int, box.tolist())
            return ((x1 + x2) // 2, y2)  # base del cuerpo
    return None

def capture_sync_frames(cameras):
    frames = []
    for cam in cameras:
        cap = cv2.VideoCapture(cam["url"])
        if not cap.isOpened():
            print(f"Error abriendo cámara {cam['id']}")
            return None
        ret, frame = cap.read()
        if not ret:
            print(f"Error leyendo de cámara {cam['id']}")
            return None
        frames.append((cam["id"], frame))
        cap.release()
    return frames

def main():
    print("== CALIBRACIÓN AUTOMÁTICA SINCRÓNICA ==")
    print("Colócate en cada punto en orden. Presiona cualquier tecla para capturar.")

    model = YOLO(MODEL_PATH)
    cam_points = {1: [], 2: []}

    for idx, real_pt in enumerate(REAL_POINTS):
        print(f"\nPunto {idx+1} en el mundo real: {real_pt}")
        input("Presiona ENTER cuando estés en posición...")

        frames = capture_sync_frames(IP_CAMERAS)
        if not frames:
            print("Error en la captura sincrónica")
            return

        for cam_id, frame in frames:
            person_base = get_detection_center(model, frame)
            if person_base:
                cam_points[cam_id].append(person_base)
                cv2.circle(frame, person_base, 5, (0, 255, 0), -1)
            else:
                print(f"No se detectó persona en cámara {cam_id}")

            cv2.imshow(f"Camara {cam_id}", frame)

        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    updated_cams = []
    for cam in IP_CAMERAS:
        cam_id = cam["id"]
        if len(cam_points[cam_id]) == 4:
            image_pts = np.array(cam_points[cam_id], dtype=np.float32)
            world_pts = np.array(REAL_POINTS, dtype=np.float32)
            H, _ = cv2.findHomography(image_pts, world_pts)
            cam_cfg = {
                "id": cam_id,
                "rtsp_url": cam["url"],
                "homography": H.tolist()
            }
            updated_cams.append(cam_cfg)
        else:
            print(f"No se obtuvieron 4 puntos en cámara {cam_id}")

    config = {
        "cameras": updated_cams,
        "bev": {"grid_resolution": 0.05},
        "tracker": {"max_age": 30, "dist_threshold": 4.0}
    }

    with open("config_calibrated.json", "w") as f:
        json.dump(config, f, indent=2)
        print("\nHomografías guardadas en 'config_calibrated.json'")

if __name__ == "__main__":
    main()
