import threading
import time

import cv2

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]
COOLDONW_AUSENTE_S = 60.0
COOLDONW_FALLO_S = 20.0


def listar_nombres_dispositivos_windows():
    try:
        from pygrabber.dshow_graph import FilterGraph
        dispositivos = FilterGraph().get_input_devices()
        nombres = []
        for d in dispositivos:
            nombres.append(d if isinstance(d, str) else getattr(d, "name", str(d)))
        return nombres
    except Exception:
        return []


class CamaraUSB:
    def __init__(self, indice, nombre=None):
        self.cam_id = f"usb_{indice}"
        self.indice = indice
        self.nombre = nombre or f"Cámara USB {indice}"
        self.cap = None
        self.frame = None
        self.bloqueo = threading.Lock()
        self.activo = False
        self.ultimo_frame_ts = 0
        self.ts_apertura = 0
        self.backend_usado = None

    def _abrir_una_vez(self):
        for nombre_backend, backend in BACKENDS:
            try:
                cap = cv2.VideoCapture(self.indice, backend)
            except Exception:
                continue
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.backend_usado = nombre_backend
                    return True
            try:
                cap.release()
            except Exception:
                pass
        return False

    def abrir(self):
        if self._abrir_una_vez():
            self.activo = True
            self.ts_apertura = time.time()
            print(f"[{self.cam_id}] abierta con backend {self.backend_usado}", flush=True)
            threading.Thread(target=self._bucle, daemon=True).start()
            return True
        print(f"[{self.cam_id}] no se pudo abrir con ningún backend", flush=True)
        return False

    def _bucle(self):
        fallos_consecutivos = 0
        while self.activo:
            ret, frame = None, None
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret = False
            if not ret:
                fallos_consecutivos += 1
                if fallos_consecutivos > 400:
                    break
                time.sleep(0.05)
                continue
            fallos_consecutivos = 0
            with self.bloqueo:
                self.frame = frame
                self.ultimo_frame_ts = time.time()

    def leer(self):
        with self.bloqueo:
            if self.frame is None:
                return None
            return self.frame.copy()

    def desactivar(self):
        self.activo = False

    def liberar_en_hilo(self):
        threading.Thread(target=self._liberar, daemon=True).start()

    def _liberar(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class GestorCamarasUSB:
    def __init__(self, max_indices=4, indices_permitidos=None):
        self.max_indices = max_indices
        self.indices_permitidos = set(indices_permitidos) if indices_permitidos else None
        self.camaras = {}
        self.ausente_desde = {}
        self.fallo_desde = {}
        self.apagados = set()

    def _indice_habilitado(self, i):
        if i in self.apagados:
            return False
        if self.indices_permitidos is not None and i not in self.indices_permitidos:
            return False
        return True

    def _probar_indice(self, i):
        abierta = False
        for _, backend in BACKENDS:
            try:
                cap = cv2.VideoCapture(i, backend)
            except Exception:
                continue
            if cap.isOpened():
                ret, _ = cap.read()
                abierta = bool(ret)
            try:
                cap.release()
            except Exception:
                pass
            if abierta:
                break
        return abierta

    def mantenimiento(self):
        ahora = time.time()
        for cam_id in list(self.camaras.keys()):
            cam = self.camaras[cam_id]
            apagada_manual = cam.indice in self.apagados
            sin_senal_inicial = (
                cam.ultimo_frame_ts == 0 and ahora - cam.ts_apertura > 25
            )
            senal_perdida = (
                cam.ultimo_frame_ts > 0 and ahora - cam.ultimo_frame_ts > 6
            )
            if apagada_manual or sin_senal_inicial or senal_perdida or not cam.activo:
                print(f"[{cam_id}] retirando (liberación en hilo aparte)", flush=True)
                cam.desactivar()
                del self.camaras[cam_id]
                cam.liberar_en_hilo()
                if not apagada_manual:
                    self.ausente_desde[cam.indice] = ahora

        nombres = listar_nombres_dispositivos_windows()
        encontradas = []
        for i in range(self.max_indices):
            if not self._indice_habilitado(i):
                continue
            cam_id = f"usb_{i}"
            if cam_id in self.camaras:
                encontradas.append(self.camaras[cam_id])
                continue
            if ahora - self.ausente_desde.get(i, -9999) < COOLDONW_AUSENTE_S:
                continue
            if ahora - self.fallo_desde.get(i, -9999) < COOLDONW_FALLO_S:
                continue
            if self._probar_indice(i):
                self.fallo_desde.pop(i, None)
                nombre = nombres[i] if i < len(nombres) else f"Cámara USB {i}"
                cam = CamaraUSB(i, nombre)
                if cam.abrir():
                    self.camaras[cam_id] = cam
                    encontradas.append(cam)
                else:
                    self.fallo_desde[i] = ahora
            else:
                self.ausente_desde[i] = ahora
        return encontradas

    def mantenimiento_forzado(self):
        self.ausente_desde.clear()
        self.fallo_desde.clear()
        return self.mantenimiento()

    def apagar_usb(self, indice):
        indice = int(indice)
        print(f"[usb_{indice}] apagada manualmente (liberando dispositivo)", flush=True)
        self.apagados.add(indice)
        cam = self.camaras.pop(f"usb_{indice}", None)
        if cam is not None:
            cam.desactivar()
            cam.liberar_en_hilo()
            return True
        return False

    def encender_usb(self, indice):
        indice = int(indice)
        if not self._indice_habilitado(indice):
            self.apagados.discard(indice)
            if self.indices_permitidos is not None and indice not in self.indices_permitidos:
                return False
        self.apagados.discard(indice)
        self.ausente_desde.pop(indice, None)
        self.fallo_desde.pop(indice, None)
        nombres = listar_nombres_dispositivos_windows()
        if self._probar_indice(indice):
            nombre = nombres[indice] if indice < len(nombres) else f"Cámara USB {indice}"
            cam = CamaraUSB(indice, nombre)
            if cam.abrir():
                self.camaras[f"usb_{indice}"] = cam
                print(f"[usb_{indice}] encendida manualmente", flush=True)
                return True
            self.fallo_desde[indice] = time.time()
        else:
            self.ausente_desde[indice] = time.time()
        return False

    def reiniciar_todo(self):
        print("Reiniciando todas las cámaras USB…", flush=True)
        self.cerrar_todas()
        self.ausente_desde.clear()
        self.fallo_desde.clear()
        time.sleep(1.0)
        return self.mantenimiento()

    def todas(self):
        return list(self.camaras.values())

    def cerrar_todas(self):
        for cam in self.camaras.values():
            cam.desactivar()
            cam.liberar_en_hilo()
        self.camaras.clear()
