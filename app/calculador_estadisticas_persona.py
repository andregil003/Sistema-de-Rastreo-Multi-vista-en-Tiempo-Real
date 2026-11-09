import time


class CalculadorEstadisticasPersona:
    def __init__(self, focal_px=550.0, altura_rostro_cm=23.0):
        self.focal_px = focal_px
        self.altura_rostro_cm = altura_rostro_cm
        self.personas = {}
        self.centroides_desconocidos = []
        self.segundos_validez = 8.0

    @staticmethod
    def _clave_camara_persona(cam_id, persona_id):
        return f"{cam_id}::{persona_id}"

    def estimar_distancia_m(self, bbox):
        _, _, _, h = bbox
        if h <= 2:
            return None
        return (self.focal_px * self.altura_rostro_cm) / (h * 100.0)

    def _identidad_desconocida(self, embedding):
        if embedding is None:
            return "desconocido"
        import numpy as np
        v = np.asarray(embedding, dtype=np.float32).ravel()
        mejor_i, mejor_sim = None, -2.0
        for i, c in enumerate(self.centroides_desconocidos):
            sim = float(np.dot(v, c) / ((np.linalg.norm(v) * np.linalg.norm(c)) + 1e-9))
            if sim > mejor_sim:
                mejor_sim, mejor_i = sim, i
        if mejor_i is not None and mejor_sim >= 0.40:
            c = self.centroides_desconocidos[mejor_i]
            self.centroides_desconocidos[mejor_i] = (c + v) / (np.linalg.norm((c + v)) + 1e-9)
            return f"Desconocido #{mejor_i + 1}"
        self.centroides_desconocidos.append(v)
        return f"Desconocido #{len(self.centroides_desconocidos)}"

    def registrar_avistamiento(self, cam_id, identificacion, bbox, score_deteccion):
        ahora = time.time()
        distancia = self.estimar_distancia_m(bbox)
        if identificacion.get("conocido"):
            clave_identidad = f"p_{identificacion['persona_id']}"
            nombre = identificacion["nombre"]
        else:
            clave_identidad = self._identidad_desconocida(identificacion.get("embedding"))
            nombre = clave_identidad
        clave = self._clave_camara_persona(cam_id, clave_identidad)
        p = self.personas.get(clave)
        if p is None or ahora - p["ultima_vez"] > self.segundos_validez:
            p = {
                "clave": clave,
                "identidad": clave_identidad,
                "nombre": nombre,
                "persona_id": identificacion.get("persona_id"),
                "camara": cam_id,
                "primera_vez_hoy": ahora,
                "ultima_vez": ahora,
                "tiempo_en_escena_s": 0.0,
                "veces_detectado": 0,
                "distancia_aprox_m": distancia,
                "confianza": score_deteccion,
                "similitud": identificacion.get("similitud"),
                "conocido": bool(identificacion.get("conocido")),
            }
        else:
            p["ultima_vez"] = ahora
            p["distancia_aprox_m"] = distancia
            p["confianza"] = score_deteccion
            p["similitud"] = identificacion.get("similitud")
            if p.get("persona_id") is None and identificacion.get("persona_id") is not None:
                p["persona_id"] = identificacion["persona_id"]
                p["nombre"] = identificacion["nombre"]
                p["conocido"] = True
        p["veces_detectado"] += 1
        p["tiempo_en_escena_s"] = round(p["ultima_vez"] - p["primera_vez_hoy"], 1)
        self.personas[clave] = p
        return p

    def limpiar_viejos(self):
        ahora = time.time()
        vivos = {}
        for clave, p in self.personas.items():
            if ahora - p["ultima_vez"] <= 90:
                vivos[clave] = p
        self.personas = vivos

    def resumen_actual(self):
        self.limpiar_viejos()
        lista = sorted(self.personas.values(), key=lambda p: p["ultima_vez"], reverse=True)
        return [
            {
                "nombre": p["nombre"],
                "conocido": p["conocido"],
                "camara": p["camara"],
                "distancia_aprox_m": None if p["distancia_aprox_m"] is None else round(p["distancia_aprox_m"], 2),
                "tiempo_en_escena_s": p["tiempo_en_escena_s"],
                "veces_detectado": p["veces_detectado"],
                "confianza": None if p["confianza"] is None else round(float(p["confianza"]), 3),
                "similitud": None if p.get("similitud") is None else round(float(p["similitud"]), 3),
            }
            for p in lista
        ]
