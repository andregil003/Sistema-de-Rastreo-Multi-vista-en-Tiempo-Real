import os
import threading
import time
import urllib.request

import cv2
import numpy as np

RUTA_MODELOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos")
URL_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
URL_SFACE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ARCHIVO_YUNET = os.path.join(RUTA_MODELOS, "face_detection_yunet_2023mar.onnx")
ARCHIVO_SFACE = os.path.join(RUTA_MODELOS, "face_recognition_sface_2021dec.onnx")

UMBRAL_COSINE = 0.363


def _descargar_si_falta(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 10000:
        return
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    tmp = destino + ".descargando"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, destino)


class MotorReconocimientoFacial:
    def __init__(self, umbral_similitud=UMBRAL_COSINE):
        _descargar_si_falta(URL_YUNET, ARCHIVO_YUNET)
        _descargar_si_falta(URL_SFACE, ARCHIVO_SFACE)
        self.umbral = umbral_similitud
        self.detector = None
        self.reconocedor = None
        self.tamano_actual = (0, 0)
        self.galeria = []
        self.galeria_actualizada_ts = 0
        self.candado = threading.Lock()
        self._preparar()

    def _preparar(self):
        img_prueba = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector = cv2.FaceDetectorYN.create(ARCHIVO_YUNET, "", (320, 240), score_threshold=0.6)
        self.detector.setInputSize((320, 240))
        self.detector.detect(img_prueba)
        self.reconocedor = cv2.FaceRecognizerSF.create(ARCHIVO_SFACE, "")

    def detectar_rostros(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.tamano_actual:
            self.detector.setInputSize((w, h))
            self.tamano_actual = (w, h)
        _, caras = self.detector.detect(frame)
        resultados = []
        if caras is not None:
            for cara in caras:
                x, y, ww, hh = cara[0:4]
                resultados.append({"fila": cara, "bbox": (int(x), int(y), int(ww), int(hh)), "score": float(cara[14])})
        return resultados

    def extraer_embedding(self, frame, fila_cara):
        alineado = self.reconocedor.alignCrop(frame, fila_cara)
        emb = self.reconocedor.feature(alineado)
        return emb[0]

    def refrescar_galeria(self, funcion_carga, segundos_validez=30):
        ahora = time.time()
        if ahora - self.galeria_actualizada_ts > segundos_validez:
            try:
                galeria_nueva = funcion_carga()
                if galeria_nueva is not None:
                    self.galeria = galeria_nueva
                    self.galeria_actualizada_ts = ahora
            except Exception:
                pass

    def identificar(self, embedding):
        mejor = None
        mejor_similitud = -2.0
        for entrada in self.galeria:
            similitud = float(
                self.reconocedor.match(
                    embedding.reshape(1, -1),
                    entrada["embedding"].reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor = entrada
        if mejor is not None and mejor_similitud >= self.umbral:
            return {
                "persona_id": mejor["persona_id"],
                "nombre": mejor["nombre"],
                "similitud": mejor_similitud,
                "conocido": True,
            }
        return {"persona_id": None, "nombre": None, "similitud": mejor_similitud, "conocido": False}

    def mejor_coincidencia(self, embedding):
        mejor = None
        mejor_similitud = -2.0
        for entrada in self.galeria:
            similitud = float(
                self.reconocedor.match(
                    embedding.reshape(1, -1),
                    entrada["embedding"].reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor = entrada
        if mejor is None:
            return None, -2.0
        return mejor["nombre"], mejor_similitud

    def analizar_frame(self, frame):
        with self.candado:
            rostros = self.detectar_rostros(frame)
            salida = []
            for r in rostros:
                emb = self.extraer_embedding(frame, r["fila"])
                ident = self.identificar(emb)
                salida.append({**r, "embedding": emb, **ident})
        return salida
