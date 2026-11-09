import json
import os
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

load_dotenv()

BUCKET_FOTOS = "rostros-registrados"

_cliente = None


def obtener_cliente():
    global _cliente
    if _cliente is not None:
        return _cliente
    url = os.getenv("SUPABASE_URL", "")
    clave = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url.startswith("http") or "PEGA_AQUI" in clave or len(clave) < 20:
        return None
    from supabase import create_client
    _cliente = create_client(url, clave)
    return _cliente


def supabase_configurado():
    return obtener_cliente() is not None


def registrar_persona(nombre, codigo_secreto, embeddings, imagenes_jpeg):
    cliente = obtener_cliente()
    if cliente is None:
        return {"ok": False, "error": "Supabase no configurado"}
    resp = (
        cliente.table("personas")
        .insert({"nombre": nombre, "codigo_registro": codigo_secreto})
        .execute()
    )
    persona_id = resp.data[0]["id"]
    filas = []
    for i, emb in enumerate(embeddings):
        url_imagen = None
        if i < len(imagenes_jpeg):
            ruta = f"{persona_id}/rostro_{i}.jpg"
            try:
                cliente.storage.from_(BUCKET_FOTOS).upload(
                    ruta,
                    imagenes_jpeg[i],
                    {"content-type": "image/jpeg", "upsert": "true"},
                )
                url_imagen = cliente.storage.from_(BUCKET_FOTOS).get_public_url(ruta)
            except Exception:
                url_imagen = None
        filas.append(
            {
                "persona_id": persona_id,
                "embedding": json.dumps([float(x) for x in np.asarray(emb).ravel()]),
                "origen": "celular",
                "imagen_url": url_imagen,
            }
        )
    cliente.table("rostros_entrenados").insert(filas).execute()
    return {"ok": True, "persona_id": persona_id}


def cargar_galeria():
    cliente = obtener_cliente()
    if cliente is None:
        return []
    personas = cliente.table("personas").select("id,nombre").execute().data
    nombres = {p["id"]: p["nombre"] for p in personas}
    rostros = (
        cliente.table("rostros_entrenados")
        .select("persona_id,embedding")
        .execute()
        .data
    )
    galeria = []
    for r in rostros:
        if r["persona_id"] not in nombres:
            continue
        galeria.append(
            {
                "persona_id": r["persona_id"],
                "nombre": nombres[r["persona_id"]],
                "embedding": np.array(json.loads(r["embedding"]), dtype=np.float32),
            }
        )
    return galeria


def guardar_evento(persona_id, nombre_detectado, camara, confianza, distancia_m, tipo):
    cliente = obtener_cliente()
    if cliente is None:
        return
    try:
        cliente.table("eventos_deteccion").insert(
            {
                "persona_id": persona_id,
                "nombre_detectado": nombre_detectado,
                "camara": camara,
                "confianza": None if confianza is None else round(float(confianza), 4),
                "distancia_aprox_m": None if distancia_m is None else round(float(distancia_m), 2),
                "tipo": tipo,
            }
        ).execute()
    except Exception:
        pass


def guardar_alerta_replay(camara, captura_jpeg, detalle):
    cliente = obtener_cliente()
    if cliente is None:
        return None
    captura_url = None
    try:
        ruta = f"alertas/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cliente.storage.from_(BUCKET_FOTOS).upload(
            ruta, captura_jpeg, {"content-type": "image/jpeg", "upsert": "true"}
        )
        captura_url = cliente.storage.from_(BUCKET_FOTOS).get_public_url(ruta)
    except Exception:
        pass
    try:
        resp = (
            cliente.table("alertas_replay")
            .insert({"camara": camara, "captura_url": captura_url, "detalle": detalle})
            .execute()
        )
        return resp.data[0]
    except Exception:
        return None


def contar_personas():
    cliente = obtener_cliente()
    if cliente is None:
        return 0
    try:
        return len(cliente.table("personas").select("id").execute().data)
    except Exception:
        return 0


def listar_personas():
    cliente = obtener_cliente()
    if cliente is None:
        return []
    try:
        return cliente.table("personas").select("id,nombre,creado_en").order("creado_en").execute().data
    except Exception:
        return []


def ultimos_reconocimientos(limite=15):
    cliente = obtener_cliente()
    if cliente is None:
        return []
    try:
        return (
            cliente.table("eventos_deteccion")
            .select("nombre_detectado,camara,confianza,distancia_aprox_m,creado_en")
            .eq("tipo", "reconocimiento")
            .order("creado_en", desc=True)
            .limit(limite)
            .execute()
            .data
        )
    except Exception:
        return []


def detecciones_de_persona(nombre, limite=10):
    cliente = obtener_cliente()
    if cliente is None:
        return []
    try:
        return (
            cliente.table("eventos_deteccion")
            .select("nombre_detectado,camara,confianza,distancia_aprox_m,creado_en,tipo")
            .eq("nombre_detectado", nombre)
            .order("creado_en", desc=True)
            .limit(limite)
            .execute()
            .data
        )
    except Exception:
        return []


def eliminar_persona(nombre):
    cliente = obtener_cliente()
    if cliente is None:
        return {"ok": False, "error": "Supabase no configurado"}
    try:
        filas = cliente.table("personas").select("id").eq("nombre", nombre).execute().data
        ids = [f["id"] for f in filas]
        if not ids:
            return {"ok": False, "error": "No existe ninguna persona con ese nombre"}

        for persona_id in ids:
            try:
                archivos = cliente.storage.from_(BUCKET_FOTOS).list(persona_id)
                if archivos:
                    rutas = [f"{persona_id}/{a['name']}" for a in archivos]
                    cliente.storage.from_(BUCKET_FOTOS).remove(rutas)
            except Exception:
                pass

        cliente.table("rostros_entrenados").delete().in_("persona_id", ids).execute()
        cliente.table("personas").delete().in_("id", ids).execute()
        return {"ok": True, "eliminadas": len(ids)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
