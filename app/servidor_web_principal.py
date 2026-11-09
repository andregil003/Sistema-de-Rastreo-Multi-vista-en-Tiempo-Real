import asyncio
import base64
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .calculador_estadisticas_persona import CalculadorEstadisticasPersona
    from .conexion_supabase_rostros import (
        cargar_galeria,
        contar_personas,
        detecciones_de_persona,
        eliminar_persona,
        guardar_alerta_replay,
        guardar_evento,
        listar_personas,
        registrar_persona,
        supabase_configurado,
        ultimos_reconocimientos,
    )
    from .detector_replay_telefono_yolo import DetectorReplayTelefono
    from .gestor_camaras_navegador_remotas import GestorCamarasNavegador
    from .gestor_camaras_usb_locales import GestorCamarasUSB
    from .motor_reconocimiento_facial_yunet_sface import MotorReconocimientoFacial
except ImportError:
    from calculador_estadisticas_persona import CalculadorEstadisticasPersona
    from conexion_supabase_rostros import (
        cargar_galeria,
        contar_personas,
        detecciones_de_persona,
        eliminar_persona,
        guardar_alerta_replay,
        guardar_evento,
        listar_personas,
        registrar_persona,
        supabase_configurado,
        ultimos_reconocimientos,
    )
    from detector_replay_telefono_yolo import DetectorReplayTelefono
    from gestor_camaras_navegador_remotas import GestorCamarasNavegador
    from gestor_camaras_usb_locales import GestorCamarasUSB
    from motor_reconocimiento_facial_yunet_sface import MotorReconocimientoFacial

load_dotenv()
RUTA_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CONFIG = os.path.join(RUTA_RAIZ, "config_sistema.json")
RUTA_PAGINA = os.path.join(RUTA_RAIZ, "pagina_web")

with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)


def guardar_config():
    with open(RUTA_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


app = FastAPI(title="Sistema Reconocimiento Facial Multi-cámara")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gestor_usb = GestorCamarasUSB(
    max_indices=config.get("max_cameras_usb_escaneo", 10),
    indices_permitidos=config.get("camaras_usb_indices"),
)
for _id_apagada in config.get("camaras_desactivadas", []):
    if isinstance(_id_apagada, str) and _id_apagada.startswith("usb_"):
        try:
            gestor_usb.apagados.add(int(_id_apagada.split("_")[1]))
        except ValueError:
            pass
gestor_nav = GestorCamarasNavegador()
motor_facial = None
detector_replay = None
estadisticas = CalculadorEstadisticasPersona(
    focal_px=config.get("focal_px_aproximada", 550),
    altura_rostro_cm=config.get("altura_rostro_cm", 23.0),
)

clientes_dashboard: set[WebSocket] = set()
cache_display: dict[str, dict] = {}
cams_nav_apagadas: set[str] = set()
resultados_analisis: dict[str, dict] = {}
ultimo_analisis_ts: dict[str, float] = {}
ciclos_analisis: dict[str, int] = {}
ultima_alerta_ts: dict[str, float] = {}
ultimo_log_evento: dict[str, float] = {}
alertas_recientes = deque(maxlen=50)
cola_eventos: deque[dict] = deque(maxlen=100)
estado_arranque = {"motor_listo": False}
calidad_jpeg = int(config.get("calidad_jpeg_stream", 70))

PALETA_BGR = [
    (60, 220, 60), (255, 160, 20), (50, 200, 255), (200, 120, 255),
    (80, 255, 220), (150, 130, 255), (70, 255, 120), (240, 90, 90),
    (100, 100, 240), (30, 180, 180),
]
COLOR_DESCONOCIDO_BGR = (40, 160, 255)


def _hex_a_bgr(hex_color):
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except Exception:
        return None


def _bgr_a_hex(bgr):
    b, g, r = [int(max(0, min(255, v))) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def color_para_persona(nombre, persona_id=None):
    override = config.get("colores_personas", {}).get(nombre)
    if override:
        bgr = _hex_a_bgr(override)
        if bgr:
            return bgr
    base = persona_id or nombre
    indice = int(hashlib.md5(str(base).encode()).hexdigest(), 16) % len(PALETA_BGR)
    return PALETA_BGR[indice]


def preparar_motores_pesados():
    global motor_facial, detector_replay
    if motor_facial is None:
        motor_facial = MotorReconocimientoFacial(umbral_similitud=config.get("umbral_similitud_rostro", 0.363))
    if detector_replay is None:
        detector_replay = DetectorReplayTelefono()
    estado_arranque["motor_listo"] = True


def decodificar_frame(cam):
    if hasattr(cam, "leer"):
        return cam.leer()
    jpeg = cam.leer_jpeg()
    if jpeg is None:
        return None
    arreglo = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arreglo, cv2.IMREAD_COLOR)


def refresco_rapido_cam(cam):
    frame = decodificar_frame(cam)
    if frame is None:
        return None
    res = resultados_analisis.get(cam.cam_id, {"rostros": [], "telefonos": [], "alerta": False})
    dibujar_overlay(frame, res["rostros"], res["telefonos"], res["alerta"])
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, calidad_jpeg])
    cache_display[cam.cam_id] = {
        "jpeg": buf.tobytes(),
        "ts": time.time(),
        "nombre": getattr(cam, "nombre", cam.cam_id),
    }
    return cam.cam_id


def analizar_camara_sync(cam_id, nombre_camara, frame):
    ahora = time.time()

    motor_facial.refrescar_galeria(cargar_galeria, segundos_validez=10)
    rostros = motor_facial.analizar_frame(frame) if motor_facial else []

    ciclo = ciclos_analisis.get(cam_id, 0) + 1
    ciclos_analisis[cam_id] = ciclo
    cada_n = max(1, int(config.get("frames_entre_chequeo_telefono", 10)))
    telefonos = []
    if ciclo % cada_n == 0 and detector_replay is not None:
        try:
            telefonos = detector_replay.detectar_telefonos(frame)
            resultado = detector_replay.hay_replay_con_telefonos(telefonos, rostros)
        except Exception:
            resultado = None
        if resultado is not None and ahora - ultima_alerta_ts.get(cam_id, 0) > 15:
            ultima_alerta_ts[cam_id] = ahora
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            captura = buf.tobytes()
            detalle = f"Telefono detectado junto a rostro en {nombre_camara}"
            registro = {
                "tipo": "alerta",
                "camara": nombre_camara,
                "cam_id": cam_id,
                "detalle": detalle,
                "captura_b64": base64.b64encode(captura).decode(),
                "creado_en": datetime.now(timezone.utc).isoformat(),
            }
            alertas_recientes.append(registro)
            cola_eventos.append(registro)
            guardar_alerta_replay(nombre_camara, captura, detalle)

    rostros_salida = []
    for r in rostros:
        distancia = estadisticas.estimar_distancia_m(r["bbox"])
        avistamiento = estadisticas.registrar_avistamiento(cam_id, r, r["bbox"], r["score"])
        clave_log = avistamiento["clave"]
        if avistamiento["conocido"] and ahora - ultimo_log_evento.get(clave_log, 0) > 30:
            ultimo_log_evento[clave_log] = ahora
            guardar_evento(
                avistamiento["persona_id"], avistamiento["nombre"], cam_id,
                r["score"], distancia, "reconocimiento",
            )
        rostros_salida.append({
            "bbox": r["bbox"],
            "nombre": avistamiento["nombre"],
            "conocido": bool(r["conocido"]),
            "similitud": float(r["similitud"]),
            "distancia_m": distancia if distancia is not None else -1.0,
            "color": color_para_persona(avistamiento["nombre"], avistamiento.get("persona_id"))
            if avistamiento["conocido"] else COLOR_DESCONOCIDO_BGR,
        })

    resultados_analisis[cam_id] = {
        "rostros": rostros_salida,
        "telefonos": telefonos,
        "alerta": ahora - ultima_alerta_ts.get(cam_id, 0) < 5,
    }


def dibujar_overlay(frame, rostros, telefonos, alerta_activa):
    alto_frame = frame.shape[0]
    for r in rostros:
        x, y, w, h = r["bbox"]
        if r.get("conocido"):
            color = r.get("color") or (60, 220, 60)
            etiqueta = r["nombre"]
        else:
            color = COLOR_DESCONOCIDO_BGR
            etiqueta = r.get("nombre") or "Desconocido"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        linea1 = etiqueta[:28]
        dist = r.get("distancia_m", -1)
        sim = r.get("similitud", 0.0)
        linea2 = f"dist ~{dist:.1f}m  sim {sim:.2f}"

        tamano_fuente = cv2.getTextSize(linea1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        ancho_barra = max(w, tamano_fuente + 70)
        alto_barra = 34
        y_barra = min(y + h + 2, alto_frame - alto_barra - 1)
        x_fin = min(frame.shape[1] - 1, x + ancho_barra)
        cv2.rectangle(frame, (x, y_barra), (x_fin, y_barra + alto_barra), color, -1)
        cv2.putText(frame, linea1, (x + 4, y_barra + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
        cv2.putText(frame, linea2, (x + 4, y_barra + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)

    for tel in telefonos:
        x, y, w, h = tel
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "TELEFONO", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if alerta_activa:
        cv2.putText(frame, "! POSIBLE REPLAY CON TELEFONO !", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def listar_camaras_json():
    principal = config.get("camara_principal_id")
    ahora = time.time()
    lista = []
    for cam in gestor_usb.todas():
        ts = getattr(cam, "ultimo_frame_ts", 0)
        senal = int((ahora - ts) * 1000) if ts > 0 else None
        lista.append({"id": cam.cam_id, "nombre": cam.nombre, "tipo": "usb", "activa": True, "senal_hace_ms": senal})
    for cam in gestor_nav.todas():
        ts = getattr(cam, "ultimo_frame_ts", 0)
        senal = int((ahora - ts) * 1000) if ts > 0 else None
        encendida = cam.cam_id not in cams_nav_apagadas
        lista.append({"id": cam.cam_id, "nombre": cam.nombre, "tipo": "navegador", "activa": encendida,
                      "senal_hace_ms": senal if encendida else None})
    nombres_apagadas = []
    if gestor_usb.apagados:
        try:
            from .gestor_camaras_usb_locales import listar_nombres_dispositivos_windows as _lnombres
        except ImportError:
            from gestor_camaras_usb_locales import listar_nombres_dispositivos_windows as _lnombres
        nombres_apagadas = _lnombres()
    for i in sorted(gestor_usb.apagados):
        nombre = nombres_apagadas[i] if i < len(nombres_apagadas) else f"Cámara USB {i}"
        lista.append({"id": f"usb_{i}", "nombre": nombre, "tipo": "usb", "activa": False, "senal_hace_ms": None})
    return sorted(lista, key=lambda c: (c["id"] != principal, c["id"]))


async def difundir(mensaje: dict):
    texto = json.dumps(mensaje)
    vivos = set()
    for ws in list(clientes_dashboard):
        try:
            await ws.send_text(texto)
            vivos.add(ws)
        except Exception:
            pass
    clientes_dashboard.clear()
    clientes_dashboard.update(vivos)


async def bucle_procesamiento():
    preparado = False
    escaneo_previo = 0.0
    loop = asyncio.get_running_loop()
    while True:
        try:
            if not preparado:
                await loop.run_in_executor(None, preparar_motores_pesados)
                preparado = True

            ahora = time.time()
            if ahora - escaneo_previo > 10:
                escaneo_previo = ahora
                await loop.run_in_executor(None, gestor_usb.mantenimiento)
                await difundir({"tipo": "camaras", "camaras": listar_camaras_json()})

            camaras = [
                c for c in gestor_usb.todas() + gestor_nav.todas()
                if c.cam_id not in cams_nav_apagadas
            ]
            ids_activos = {c.cam_id for c in camaras}
            for cid in list(cache_display):
                if cid not in ids_activos:
                    cache_display.pop(cid, None)
                    resultados_analisis.pop(cid, None)

            await asyncio.gather(
                *[loop.run_in_executor(None, refresco_rapido_cam, cam) for cam in camaras],
                return_exceptions=True,
            )

            intervalo_analisis = float(config.get("fps_reconocimiento_ms", 300)) / 1000.0
            tareas_heavy = []
            metadatos_heavy = []
            for cam in camaras:
                if ahora - ultimo_analisis_ts.get(cam.cam_id, 0) >= intervalo_analisis:
                    frame = decodificar_frame(cam)
                    if frame is None:
                        continue
                    ultimo_analisis_ts[cam.cam_id] = ahora
                    tareas_heavy.append(
                        loop.run_in_executor(
                            None, analizar_camara_sync,
                            cam.cam_id, getattr(cam, "nombre", cam.cam_id), frame,
                        )
                    )
                    metadatos_heavy.append(cam.cam_id)
            if tareas_heavy:
                resultados = await asyncio.gather(*tareas_heavy, return_exceptions=True)
                for cid, res in zip(metadatos_heavy, resultados):
                    if isinstance(res, Exception):
                        print(f"Error analizando {cid}: {res!r}", flush=True)

            while cola_eventos:
                evento = cola_eventos.popleft()
                await difundir(evento)

            await asyncio.sleep(0.1)
        except Exception as e:
            print("Error bucle procesamiento:", repr(e), flush=True)
            await asyncio.sleep(1.0)


async def generador_mjpeg(cam_id: str):
    ultimo_enviado = 0.0
    while True:
        datos = cache_display.get(cam_id)
        if datos is not None and datos["ts"] > ultimo_enviado:
            ultimo_enviado = datos["ts"]
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + datos["jpeg"]
                + b"\r\n"
            )
        await asyncio.sleep(0.08)


@app.on_event("startup")
async def al_arrancar():
    global _bucle_iniciado
    if _bucle_iniciado:
        return
    _bucle_iniciado = True
    asyncio.create_task(bucle_procesamiento())


_bucle_iniciado = False


class MiddlewareSinCache:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def enviar_con_cabeceras(mensaje):
            if mensaje["type"] == "http.response.start":
                cabeceras = list(mensaje.get("headers", []))
                cabeceras.append((b"cache-control", b"no-store"))
                mensaje = {**mensaje, "headers": cabeceras}
            await send(mensaje)

        await self.app(scope, receive, enviar_con_cabeceras)


app.add_middleware(MiddlewareSinCache)


@app.on_event("shutdown")
async def al_apagar():
    gestor_usb.cerrar_todas()


@app.get("/")
async def pagina_principal():
    return FileResponse(os.path.join(RUTA_PAGINA, "index.html"))


@app.get("/movil")
async def pagina_movil():
    return FileResponse(os.path.join(RUTA_PAGINA, "movil.html"))


@app.get("/api/stream/{cam_id}")
async def api_stream(cam_id: str):
    return StreamingResponse(
        generador_mjpeg(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/estado")
async def api_estado():
    return {
        "motor_listo": estado_arranque["motor_listo"],
        "supabase_configurado": supabase_configurado(),
        "personas_registradas": contar_personas(),
        "camaras_conectadas": sum(1 for c in listar_camaras_json() if c["activa"]),
    }


@app.get("/api/camaras")
async def api_camaras():
    return {"camaras": listar_camaras_json(), "camara_principal_id": config.get("camara_principal_id")}


class CuerpoPrincipal(BaseModel):
    cam_id: str


@app.post("/api/camara_principal")
async def api_camara_principal(cuerpo: CuerpoPrincipal):
    config["camara_principal_id"] = cuerpo.cam_id
    guardar_config()
    await difundir({"tipo": "camaras", "camaras": listar_camaras_json()})
    return {"ok": True}


class CuerpoEstadoCam(BaseModel):
    activa: bool


@app.post("/api/camara/{cam_id}/estado")
async def api_estado_camara(cam_id: str, cuerpo: CuerpoEstadoCam):
    loop = asyncio.get_running_loop()
    if cam_id.startswith("usb_"):
        try:
            indice = int(cam_id.split("_")[1])
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Identificador USB inválido")
        if cuerpo.activa:
            await loop.run_in_executor(None, gestor_usb.encender_usb, indice)
        else:
            await loop.run_in_executor(None, gestor_usb.apagar_usb, indice)
        config["camaras_desactivadas"] = [f"usb_{i}" for i in sorted(gestor_usb.apagados)]
        guardar_config()
    elif cam_id.startswith("nav_"):
        if cuerpo.activa:
            cams_nav_apagadas.discard(cam_id)
        else:
            cams_nav_apagadas.add(cam_id)
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Cámara no encontrada")

    principal = config.get("camara_principal_id")
    activas = [c for c in listar_camaras_json() if c["activa"]]
    if not principal and activas:
        config["camara_principal_id"] = activas[0]["id"]
        guardar_config()
    elif principal and not any(c["id"] == principal for c in activas):
        config["camara_principal_id"] = activas[0]["id"] if activas else ""
        guardar_config()
    camaras_json = listar_camaras_json()
    await difundir({"tipo": "camaras", "camaras": camaras_json})
    return {"ok": True, "camaras": camaras_json}


@app.get("/api/estadisticas")
async def api_estadisticas():
    resumen = estadisticas.resumen_actual()
    return {"personas": resumen, "total_en_escena": len(resumen)}


@app.get("/api/alertas")
async def api_alertas():
    return {"alertas": list(alertas_recientes)}


@app.get("/api/personas")
async def api_personas():
    personas = listar_personas()
    return {
        "personas": [
            {
                "nombre": p["nombre"],
                "color_hex": _bgr_a_hex(color_para_persona(p["nombre"], p["id"])),
                "creado_en": p.get("creado_en"),
            }
            for p in personas
        ]
    }


class CuerpoColor(BaseModel):
    color: str


@app.post("/api/color_persona/{nombre}")
async def api_color_persona(nombre: str, cuerpo: CuerpoColor):
    color = cuerpo.color.strip()
    if not (len(color) == 7 and color.startswith("#")):
        return {"ok": False, "error": "Color inválido, usa formato #rrggbb"}
    try:
        int(color[1:], 16)
    except ValueError:
        return {"ok": False, "error": "Color inválido"}
    config.setdefault("colores_personas", {})[nombre] = color
    guardar_config()
    await difundir({"tipo": "camaras", "camaras": listar_camaras_json()})
    return {"ok": True, "nombre": nombre, "color": color}


@app.get("/api/ultimos_reconocimientos")
async def api_ultimos_reconocimientos():
    return {"reconocimientos": ultimos_reconocimientos(15)}


@app.delete("/api/persona/{nombre}")
async def api_eliminar_persona(nombre: str):
    resultado = eliminar_persona(nombre)
    if not resultado.get("ok"):
        return resultado
    config.get("colores_personas", {}).pop(nombre, None)
    guardar_config()
    if motor_facial is not None:
        motor_facial.galeria_actualizada_ts = 0
    cola_eventos.append({
        "tipo": "registro",
        "nombre": nombre,
        "detalle": "Registro eliminado de la base de datos",
    })
    return {"ok": True, "eliminadas": resultado.get("eliminadas", 0)}


@app.get("/api/mis_detecciones/{nombre}")
async def api_mis_detecciones(nombre: str):
    return {"detecciones": detecciones_de_persona(nombre, 10)}


@app.get("/api/probar_reconocimiento/{cam_id}")
async def api_probar_reconocimiento(cam_id: str):
    loop = asyncio.get_running_loop()

    def analizar_ahora():
        cam = next((c for c in gestor_usb.todas() + gestor_nav.todas() if c.cam_id == cam_id), None)
        if cam is None:
            return {"ok": False, "error": "Esa cámara no está conectada"}
        frame = decodificar_frame(cam)
        if frame is None:
            return {"ok": False, "error": "Sin imagen de esa cámara"}
        motor_facial.refrescar_galeria(cargar_galeria, segundos_validez=0)
        rostros = motor_facial.analizar_frame(frame)
        salida = []
        for r in rostros:
            if r["conocido"]:
                salida.append({
                    "conocido": True,
                    "nombre": r["nombre"],
                    "similitud": float(r["similitud"]),
                })
            else:
                mejor_nombre, mejor_sim = motor_facial.mejor_coincidencia(r["embedding"])
                salida.append({
                    "conocido": False,
                    "mejor_nombre": mejor_nombre or "-",
                    "mejor_sim": float(mejor_sim),
                })
        return {
            "ok": True,
            "camara": getattr(cam, "nombre", cam_id),
            "umbral": config.get("umbral_similitud_rostro", 0.36),
            "galeria_rostros": len(motor_facial.galeria),
            "rostros": salida,
        }

    return await loop.run_in_executor(None, analizar_ahora)


@app.post("/api/camaras/reescanear")
async def api_reescanear():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, gestor_usb.mantenimiento_forzado)
    camaras = listar_camaras_json()
    await difundir({"tipo": "camaras", "camaras": camaras})
    return {"ok": True, "camaras": camaras}


@app.post("/api/camaras/reiniciar_usb")
async def api_reiniciar_usb():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, gestor_usb.reiniciar_todo)
    camaras = listar_camaras_json()
    await difundir({"tipo": "camaras", "camaras": camaras})
    return {"ok": True, "camaras": camaras}


class CuerpoRegistro(BaseModel):
    nombre: str
    codigo: str
    imagenes: list[str]


@app.post("/api/registrar")
async def api_registrar(cuerpo: CuerpoRegistro):
    codigo_configurado = os.getenv("CODIGO_SECRETO_REGISTRO", "").strip()
    if codigo_configurado and cuerpo.codigo.strip() != codigo_configurado:
        return {"ok": False, "error": "Código secreto incorrecto"}
    nombre = cuerpo.nombre.strip()[:60]
    if len(nombre) < 2:
        return {"ok": False, "error": "Nombre demasiado corto"}

    embeddings = []
    imagenes_jpeg = []
    for dataurl in cuerpo.imagenes[:12]:
        try:
            base64_data = dataurl.split(",", 1)[1]
            bytes_img = base64.b64decode(base64_data)
            arreglo = np.frombuffer(bytes_img, dtype=np.uint8)
            img = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)
            if img is None:
                continue
            rostros = motor_facial.detectar_rostros(img)
            if not rostros:
                continue
            mejor = max(rostros, key=lambda r: r["bbox"][2] * r["bbox"][3])
            emb = motor_facial.extraer_embedding(img, mejor["fila"])
            embeddings.append(np.asarray(emb).ravel())
            imagenes_jpeg.append(bytes_img)
        except Exception:
            continue

    if len(embeddings) < 3:
        return {"ok": False, "error": f"No se pudieron extraer rostros válidos ({len(embeddings)}). Acerca el rostro a la cámara con buena luz."}

    resultado = registrar_persona(nombre, codigo_configurado[:20], embeddings, imagenes_jpeg)
    if resultado.get("ok"):
        motor_facial.galeria_actualizada_ts = 0
        cola_eventos.append({
            "tipo": "registro",
            "nombre": nombre,
            "detalle": f"{len(embeddings)} muestras faciales guardadas",
        })
        return {"ok": True, "persona_id": resultado["persona_id"], "muestras": len(embeddings)}
    return resultado


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    clientes_dashboard.add(ws)
    try:
        await ws.send_text(json.dumps({"tipo": "camaras", "camaras": listar_camaras_json()}))
        await ws.send_text(json.dumps({"tipo": "alertas_historico", "alertas": list(alertas_recientes)}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clientes_dashboard.discard(ws)


@app.websocket("/ws/camara_navegador")
async def ws_camara_navegador(ws: WebSocket):
    await ws.accept()
    cam = None
    try:
        saludo = await asyncio.wait_for(ws.receive_text(), timeout=6)
        datos = json.loads(saludo)
        if datos.get("rol") != "camara":
            await ws.close(code=4001)
            return
        cam = gestor_nav.registrar(ws, datos.get("nombre"))
        await ws.send_text(json.dumps({"tipo": "registrada", "cam_id": cam.cam_id}))
        await difundir({"tipo": "camaras", "camaras": listar_camaras_json()})
        while True:
            mensaje = await ws.receive()
            if "bytes" in mensaje and mensaje["bytes"]:
                cam.recibir_frame(mensaje["bytes"])
            elif mensaje.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass
    finally:
        if cam is not None:
            gestor_nav.dar_baja(cam.cam_id)
            cache_display.pop(cam.cam_id, None)
            await difundir({"tipo": "camaras", "camaras": listar_camaras_json()})


app.mount("/web", StaticFiles(directory=RUTA_PAGINA), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PUERTO_SERVIDOR", "8000")))
