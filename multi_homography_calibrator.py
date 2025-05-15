import cv2
import numpy as np
import json

# === CONFIGURACIÓN ===
IP_CAMERAS = [
    {"id": 1, "url": "http://192.168.0.24:4747/video"},
    {"id": 2, "url": "http://192.168.0.4:4747/video"}
]

# Puntos en el mundo real (metros)
REAL_POINTS = [
    [0, 0],
    [2, 0],
    [2, 1],
    [0, 1]
]

# Variables globales para almacenar puntos
image_points = []
current_image = None
current_camera_id = None

def click_event(event, x, y, flags, params):
    global image_points, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) < 4:
            image_points.append((x, y))
            print(f"Punto {len(image_points)} seleccionado en: ({x}, {y})")
            
            # Dibujar punto
            cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(current_image, f"{len(image_points)}", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f"Calibración de Cámara {current_camera_id}", current_image)
            
            # Si ya seleccionamos los 4 puntos, mostrar mensaje
            if len(image_points) == 4:
                print("\nSe han seleccionado los 4 puntos. Presiona 'q' para continuar.")

def main():
    global image_points, current_image, current_camera_id
    
    print("== CALIBRACIÓN MANUAL DE HOMOGRAFÍA ==")
    print("Instrucciones:")
    print("1. Selecciona 4 puntos en la imagen que correspondan a las esquinas del área de seguimiento")
    print("2. Los puntos deben corresponder a estas coordenadas en el mundo real:")
    for i, point in enumerate(REAL_POINTS):
        print(f"   Punto {i+1}: {point}")
    print("3. Presiona 'q' después de seleccionar los 4 puntos")
    
    all_camera_data = {}
    
    for camera in IP_CAMERAS:
        current_camera_id = camera["id"]
        image_points = []  # Resetear puntos para cada cámara
        
        cap = cv2.VideoCapture(camera["url"])
        if not cap.isOpened():
            print(f"Error: No se puede abrir la cámara {camera['id']}")
            continue
        
        print(f"\nCapturando imagen de cámara {camera['id']}...")
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: No se puede leer de la cámara {camera['id']}")
            continue
        
        current_image = frame.copy()
        cv2.imshow(f"Calibración de Cámara {camera['id']}", current_image)
        cv2.setMouseCallback(f"Calibración de Cámara {camera['id']}", click_event)
        
        print(f"\nPara cámara {camera['id']}, selecciona los 4 puntos en este orden:")
        for i, point in enumerate(REAL_POINTS):
            print(f"   Punto {i+1}: {point}")
        
        print("\nHaz click en los 4 puntos y luego presiona 'q'")
        
        # Esperar a que el usuario seleccione los puntos
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):  # Opción para reiniciar
                image_points = []
                current_image = frame.copy()
                cv2.imshow(f"Calibración de Cámara {camera['id']}", current_image)
                print("\nPuntos reiniciados. Selecciona nuevamente.")
        
        cv2.destroyAllWindows()
        
        # Verificar que se hayan seleccionado los 4 puntos
        if len(image_points) != 4:
            print(f"Error: Se requieren 4 puntos para la calibración. Solo se seleccionaron {len(image_points)}.")
            continue
        
        # Imprimir los puntos seleccionados
        print(f"\nPuntos seleccionados para cámara {camera['id']}:")
        for i, pt in enumerate(image_points):
            print(f"   Punto {i+1}: {pt} -> {REAL_POINTS[i]}")
        
        # Calcular homografía
        try:
            image_pts = np.array(image_points, dtype=np.float32)
            world_pts = np.array(REAL_POINTS, dtype=np.float32)
            
            H, status = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                all_camera_data[camera["id"]] = {
                    "id": camera["id"],
                    "rtsp_url": str(camera["url"]),
                    "homography": H.tolist()
                }
                print(f"\nHomografía calculada correctamente para cámara {camera['id']}")
                
                # Dibujamos los puntos transformados para verificar
                h, w = frame.shape[:2]
                test_image = frame.copy()
                
                # Dibujar cuadrícula
                grid_size = 0.5  # metros
                for x in np.arange(0, 3, grid_size):
                    for y in np.arange(0, 2, grid_size):
                        world_pt = np.array([[x, y, 1]], dtype=np.float32).T
                        image_pt = np.dot(np.linalg.inv(H), world_pt)
                        image_pt = (int(image_pt[0,0] / image_pt[2,0]), int(image_pt[1,0] / image_pt[2,0]))
                        
                        if 0 <= image_pt[0] < w and 0 <= image_pt[1] < h:
                            cv2.circle(test_image, image_pt, 3, (0, 0, 255), -1)
                
                cv2.imshow(f"Verificación de Homografía - Cámara {camera['id']}", test_image)
                print("Presiona cualquier tecla para continuar...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Error: No se pudo calcular la homografía para cámara {camera['id']}")
        except Exception as e:
            print(f"Error al calcular homografía para cámara {camera['id']}: {str(e)}")
    
    # Crear el archivo de configuración final
    if all_camera_data:
        cameras_list = list(all_camera_data.values())
        config = {
            "cameras": cameras_list,
            "bev": {"grid_resolution": 0.05},
            "tracker": {"max_age": 30, "dist_threshold": 4.0}
        }
        
        with open("config_calibrated.json", "w") as f:
            json.dump(config, f, indent=2)
            print("\nConfiguración guardada en 'config_calibrated.json'")
        
        print("\nCalibración finalizada con éxito.")
    else:
        print("\nNo se pudo generar la configuración - no hay cámaras calibradas.")

if __name__ == "__main__":
    main()