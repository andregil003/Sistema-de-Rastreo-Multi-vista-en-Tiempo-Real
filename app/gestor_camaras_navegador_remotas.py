import threading
import time
import uuid


class CamaraNavegador:
    def __init__(self, websocket, nombre=None):
        self.cam_id = f"nav_{uuid.uuid4().hex[:8]}"
        self.websocket = websocket
        self.nombre = nombre or "Cámara de navegador"
        self.frame = None
        self.bloqueo = threading.Lock()
        self.conectada = True
        self.ultimo_frame_ts = time.time()

    def recibir_frame(self, datos_jpeg):
        with self.bloqueo:
            self.frame = datos_jpeg
            self.ultimo_frame_ts = time.time()

    def leer_jpeg(self):
        with self.bloqueo:
            return self.frame

    def desconectar(self):
        self.conectada = False


class GestorCamarasNavegador:
    def __init__(self):
        self.camaras = {}

    def registrar(self, websocket, nombre):
        cam = CamaraNavegador(websocket, nombre)
        self.camaras[cam.cam_id] = cam
        return cam

    def dar_baja(self, cam_id):
        cam = self.camaras.pop(cam_id, None)
        if cam is not None:
            cam.desconectar()

    def todas(self):
        ahora = time.time()
        vivas = []
        for cam in list(self.camaras.values()):
            if cam.conectada and ahora - cam.ultimo_frame_ts < 4:
                vivas.append(cam)
            elif not cam.conectada:
                self.camaras.pop(cam.cam_id, None)
        return vivas
