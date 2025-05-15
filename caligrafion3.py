import cv2
import numpy as np
import json

# === CONFIGURACIÓN ===
IP_CAMERAS = [
    {"id": 1, "url": "http://192.168.0.24:4747/video"},
    {"id": 2, "url": "http://192.168.0.4:4747/video"}
]

REAL_POINTS = [
    [0, 0],
    [2, 0],
    [2, 1],
    [0, 1]
]

def get_manual_points(frame, cam_id):
    points = []
    clone = frame.copy()
    window_name = f"Camara {cam_id} - Selecciona 4 puntos en el suelo"
    cv2.imshow(window_name, clone)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)

    cv2.setMouseCallback(window_name, click_event)

    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow(window_name)
    return points

def capture_frame(cam):
    cap = cv2.VideoCapture(cam["url"])
    if not cap.isOpened():
        print(f"Error abriendo cámara {cam['id']}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error leyendo de cámara {cam['id']}")
        return None
    return frame

def main():
    print("== CALIBRACIÓN MANUAL DE CÁMARAS ==")
    print("Selecciona 4 puntos en el suelo (en orden) para cada cámara.")

    cam_points = {}

    for cam in IP_CAMERAS:
        cam_id = cam["id"]
        print(f"\n--- Cámara {cam_id} ---")
        frame = capture_frame(cam)
        if frame is None:
            print(f"No se pudo capturar imagen de la cámara {cam_id}")
            continue

        points = get_manual_points(frame, cam_id)
        if len(points) == 4:
            cam_points[cam_id] = points
        else:
            print(f"Menos de 4 puntos seleccionados para cámara {cam_id}")

    updated_cams = []
    for cam in IP_CAMERAS:
        cam_id = cam["id"]
        if cam_id in cam_points:
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
            print(f"No homografía para cámara {cam_id}")

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
