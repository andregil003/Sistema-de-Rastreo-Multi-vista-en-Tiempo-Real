import cv2
import numpy as np
import json

# Define el tamaño de la imagen que usarás para la proyección BEV
OUTPUT_SHAPE = (500, 500)

# Coordenadas reales del mundo en metros o cm (elige tu unidad)
# Ejemplo: un rectángulo de 2m x 1m (o 200cm x 100cm)
real_points = [
    [0, 0],
    [2, 0],
    [2, 1],
    [0, 1]
]

clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Selecciona 4 puntos en el suelo", param)

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        return

    clone = frame.copy()
    cv2.imshow("Selecciona 4 puntos en el suelo", clone)
    cv2.setMouseCallback("Selecciona 4 puntos en el suelo", click_event, clone)

    while len(clicked_points) < 4:
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    image_pts = np.array(clicked_points, dtype=np.float32)
    world_pts = np.array(real_points, dtype=np.float32)

    H, status = cv2.findHomography(image_pts, world_pts)
    print("Homografía calculada:")
    print(H)

    # Guardar en formato JSON para usar en config.json
    homography_list = H.tolist()
    print("\nHomografía (para config.json):")
    print(json.dumps(homography_list, indent=2))

if __name__ == "__main__":
    main()
