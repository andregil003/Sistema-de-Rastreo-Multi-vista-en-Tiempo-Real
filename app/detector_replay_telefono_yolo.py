import os

import cv2

RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos", "yolo11n.pt"
)
CLASE_TELEFONO = 67
ANCHO_INFERENCIA = 480


class DetectorReplayTelefono:
    def __init__(self, umbral_conf=0.35):
        from ultralytics import YOLO
        self.modelo = YOLO(RUTA_MODELO)
        self.umbral_conf = umbral_conf
        self.telefonos = []

    def detectar_telefonos(self, frame):
        h, w = frame.shape[:2]
        escala = 1.0
        if w > ANCHO_INFERENCIA:
            escala = w / ANCHO_INFERENCIA
            pequeno = cv2.resize(frame, (ANCHO_INFERENCIA, int(h / escala)))
        else:
            pequeno = frame
        resultados = self.modelo.predict(
            pequeno, classes=[CLASE_TELEFONO], conf=self.umbral_conf,
            imgsz=384, verbose=False,
        )
        telefonos = []
        for r in resultados:
            for caja in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = caja
                if escala != 1.0:
                    x1, y1, x2, y2 = x1 * escala, y1 * escala, x2 * escala, y2 * escala
                telefonos.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        self.telefonos = telefonos
        return telefonos

    @staticmethod
    def _expander(bbox, factor=0.5):
        x, y, w, h = bbox
        dx, dy = int(w * factor / 2), int(h * factor / 2)
        return (x - dx, y - dy, w + dx * 2, h + dy * 2)

    @staticmethod
    def _se_superponen(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah

    def hay_replay_con_telefonos(self, telefonos, rostros):
        if not telefonos or not rostros:
            return None
        for tel in telefonos:
            for rostro in rostros:
                zona = self._expander(rostro["bbox"], 0.6)
                if self._se_superponen(zona, tel):
                    return {
                        "telefono_bbox": tel,
                        "rostro_bbox": rostro["bbox"],
                        "confianza_rostro": rostro.get("score"),
                    }
        return None

    def hay_replay(self, frame, rostros):
        return self.hay_replay_con_telefonos(self.detectar_telefonos(frame), rostros)
