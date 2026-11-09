# Arquitectura y Código — Sistema de Reconocimiento Facial Multi-cámara

> Documentación técnica completa del proyecto. Complemento del [`MANUAL_DE_USO.md`](MANUAL_DE_USO.md) (operación) y el [`README.md`](README.md).

---

## 1. Visión general

Sistema de vigilancia con **reconocimiento facial en tiempo real** que combina:

- **Múltiples cámaras USB** locales (hasta `max_cameras_usb_escaneo`, por defecto 4) gestionadas automáticamente.
- **Detección de rostros** con YuNet (ONNX) y **reconocimiento** con SFace (embeddings de 128 dimensiones, comparación coseno).
- **Anti-spoofing / anti-replay**: detección de teléfonos celulares con YOLOv11n; si un celular aparece superpuesto a un rostro se genera una alerta (alguien podría estar mostrando una foto/video a la cámara).
- **Persistencia en Supabase** (Postgres + Storage): personas registradas, embeddings faciales, eventos de detección y alertas.
- **Dashboard web** en vivo (streams MJPEG con overlay dibujado en servidor) y **página móvil** para autoregistro.

### Diagrama general

```
┌──────────────────────────── CLIENTES (navegadores) ────────────────────────────┐
│  Dashboard PC: index.html · aplicacion.js · estilos_meet.css                    │
│    ├─ <img src="/api/stream/{id}">   ← video MJPEG de cada cámara               │
│    ├─ WebSocket /ws/dashboard        ← push de cámaras/alertas/registros        │
│    └─ fetch REST                     ← acciones (apagar cámara, registrar…)     │
│  Móvil: /movil → movil.html · aplicacion_movil.js                               │
└───────────────▲─────────────────────▲──────────────────────────▲────────────────┘
                │                      │                          │
════════════════╪══════════════════════╪══════════════════════════╪═══════════════
                │      FastAPI · servidor_web_principal.py                        │
                │      HTTP :8000  +  HTTPS :8443  (iniciar_servidores.py)        │
                │                                                                  │
                │  bucle_procesamiento() — tarea asyncio (~10 ciclos/seg):         │
                │   1. refresco_rapido_cam(cam) × N  → frame fresco + último       │
                │      overlay conocido → cache_display[cam_id] ──► generador_mjpeg│
                │   2. cada fps_reconocimiento_ms por cámara (en hilos):           │
                │      analizar_camara_sync():                                     │
                │        · MotorReconocimientoFacial.analizar_frame                │
                │          (YuNet detecta → SFace embedding → identificar)         │
                │        · DetectorReplayTelefono (cada N ciclos)                  │
                │        · CalculadorEstadisticasPersona.registrar_avistamiento    │
                │   3. difundir() eventos por WebSocket                            │
                │   4. persistir eventos/alertas                                   │
════════════════╪═══════════════════════════════════════════════════════════════════
                │                      │                          │
     ┌──────────┴─────────┐  ┌─────────┴──────────────┐  ┌────────┴────────────────┐
     │ GestorCamarasUSB   │  │ GestorCamarasNavegador │  │ conexion_supabase_      │
     │ (usb_0..usb_n,     │  │ (nav_xxxxxxxx, JPEGs   │  │ rostros.py              │
     │  hilo de captura   │  │  recibidos por WS)     │  │ personas · embeddings   │
     │  por cámara)       │  │                        │  │ eventos · alertas       │
     └────────────────────┘  └────────────────────────┘  │ Storage: fotos rostros  │
                                                          └────────┬────────────────┘
                                                                   ▼
                                                        Supabase (Postgres+Storage)
```

### Flujo de datos por frame (resumen)

1. Cada `CamaraUSB` tiene su **propio hilo daemon** que lee `cap.read()` continuamente y guarda el último frame bajo lock (`gestor_camaras_usb_locales.py`).
2. El bucle asíncrono principal toma ese frame, **dibuja el overlay del último análisis conocido** y publica el JPEG resultante en `cache_display` → los `<img>` MJPEG del navegador lo consumen (~12 fps efectivos).
3. En paralelo, **sin bloquear el video**, un análisis pesado (detección + embedding + identificación) corre en hilos del executor cada `fps_reconocimiento_ms` (300 ms por defecto). Su resultado alimenta tanto el overlay como las estadísticas y la base de datos.
4. Los cambios relevantes (nueva alerta, nuevo registro, lista de cámaras) se empujan a todos los dashboards abiertos vía `/ws/dashboard`.

---

## 2. Árbol de archivos

```
Sistema-de-Rastreo-Multi-vista-en-Tiempo-Real\
│
├── app\                              ← Backend Python (FastAPI)
│   ├── __init__.py                   (marca el paquete; vacío)
│   ├── iniciar_servidores.py         Punto de entrada: levanta HTTP+HTTPS y abre navegador
│   ├── servidor_web_principal.py     App FastAPI completa: API REST, WebSockets, bucles
│   ├── generador_certificado_https.py Certificado TLS autofirmado para el modo celular
│   ├── gestor_camaras_usb_locales.py Ciclo de vida de cámaras USB (abrir/apagar/reintentar)
│   ├── gestor_camaras_navegador_remotas.py Cámaras remotas enviadas por WebSocket
│   ├── motor_reconocimiento_facial_yunet_sface.py Detección YuNet + reconocimiento SFace
│   ├── detector_replay_telefono_yolo.py Detección de celulares (anti-replay)
│   ├── calculador_estadisticas_persona.py Estadísticas por persona + distancia estimada
│   └── conexion_supabase_rostros.py  Capa de acceso a Supabase (BD + Storage)
│
├── pagina_web\                       ← Frontend (vanilla JS, sin build)
│   ├── index.html                    Dashboard principal (pestañas + fondo Turing animado)
│   ├── aplicacion.js                 Toda la lógica del dashboard (~680 líneas)
│   ├── estilos_meet.css              Sistema de diseño (tokens CSS oscuros minimalistas)
│   ├── movil.html                    Página ligera para celular (registro + mi actividad)
│   └── aplicacion_movil.js           Lógica de la página móvil
│
├── modelos\                          ← Pesos descargados/guardados (~43 MB total)
│   ├── face_detection_yunet_2023mar.onnx      Detección facial (227 KB)
│   ├── face_recognition_sface_2021dec.onnx    Embeddings 128D (37 MB)
│   └── yolo11n.pt                             YOLO nano COCO (5.4 MB)
│
├── certs\                            ← cert.pem + key.pem (autogenerados, 10 años)
├── logs\                             ← servidor_out.log / servidor_err.log
│
├── sql_supabase.sql                  Esquema completo de BD (pegar en SQL Editor)
├── config_sistema.json               Configuración persistente del sistema
├── .env                              Secretos (Supabase URL/key, código registro)
├── requirements.txt                  Dependencias pip
├── README.md                         Descripción general
├── ARQUITECTURA_Y_CODIGO.md          Este documento
└── MANUAL_DE_USO.md                  Guía de operación
```

---

## 3. Backend — módulos uno por uno

Los imports usan el patrón *try relativo / except absoluto* (ver `servidor_web_principal.py:19-54`), por eso el sistema arranca igual con `python -m app.iniciar_servidores` o `python app\iniciar_servidores.py`.

### 3.1 `app/iniciar_servidores.py` — Punto de entrada

Bootstrap mínimo. No contiene lógica de negocio.

| Elemento | Función |
|---|---|
| `PUERTO_HTTP` / `PUERTO_HTTPS` | 8000 (configurable con env `PUERTO_SERVIDOR`) y 8443 fijos |
| `_correr_servidor(config)` | Ejecuta un `uvicorn.Server(config).run()` — usado como target de hilo |
| `main()` | Importa la app FastAPI, arma dos configs uvicorn (HTTP plano + HTTPS con `cert.pem`/`key.pem` si el certificado pudo generarse), lanza **cada servidor en un hilo daemon**, imprime las URLs (PC y celular), espera hasta que `/api/estado` responda (sondeo máx. 30 s), abre el navegador y queda dormido hasta `Ctrl+C`. |

Complementa a: todos. Es la única pieza que orquesta el arranque; al morir (Ctrl+C), los hilos daemon mueren con él y el evento `shutdown` de FastAPI libera las cámaras.

### 3.2 `app/servidor_web_principal.py` — El corazón (743 líneas)

App FastAPI única que concentra API REST, WebSockets, el bucle de procesamiento y el dibujo del overlay.

**Estado global clave** (líneas 73–108):

| Variable | Contenido |
|---|---|
| `gestor_usb` | Instancia de `GestorCamarasUSB`; precarga `apagados` desde `config["camaras_desactivadas"]` |
| `gestor_nav` | Instancia de `GestorCamarasNavegador` |
| `motor_facial`, `detector_replay` | Se crean **lazy** en `preparar_motores_pesados()` (primera iteración del bucle, en un executor para no congelar el event loop) |
| `estadisticas` | `CalculadorEstadisticasPersona` con focal y altura de config |
| `clientes_dashboard` | Set de WebSockets conectados al dashboard |
| `cache_display` | `{cam_id: {"jpeg", "ts", "nombre"}}` — último frame listo para transmitir |
| `resultados_analisis` | `{cam_id: {"rostros", "telefonos", "alerta"}}` — último análisis por cámara |
| `cams_nav_apagadas` | IDs de cámaras de navegador desactivadas en caliente (runtime only) |
| `alertas_recientes` | Deque maxlen=50 (historial de alertas en memoria) |
| `cola_eventos` | Deque maxlen=100 — eventos pendientes de difundir por WS |
| `PALETA_BGR` | 10 colores BGR asignados por hash MD5 del id/nombre de persona |

**Funciones internas:**

| Función | Qué hace | La llaman |
|---|---|---|
| `guardar_config()` | Escribe `config_sistema.json` con indent | Todos los endpoints que mutan config |
| `_hex_a_bgr` / `_bgr_a_hex` | Conversión entre `#rrggbb` y tupla BGR OpenCV | `color_para_persona`, `/api/personas` |
| `color_para_persona(nombre, persona_id)` | Color estable por persona: override manual de `colores_personas` o índice = `md5(id) % 10` | `analizar_camara_sync`, `/api/personas` |
| `preparar_motores_pesados()` | Inicializa motor facial + detector replay una sola vez; activa `motor_listo` | Bucle (1ª iteración) |
| `decodificar_frame(cam)` | Frame BGR uniforme: `cam.leer()` (USB) o decodifica JPEG (navegador) | Refresco y análisis |
| `refresco_rapido_cam(cam)` | Toma frame actual, dibuja el overlay del último análisis, codifica JPEG (calidad `calidad_jpeg_stream`) y lo deja en `cache_display`. Barato: sin inferencia | Bucle, todas las cámaras cada ciclo |
| `analizar_camara_sync(...)` | Análisis pesado completo: refresca galería (TTL 10 s), `motor_facial.analizar_frame`, chequeo YOLO de teléfonos cada `frames_entre_chequeo_telefono` ciclos, alerta replay con cooldown 15 s por cámara (captura JPEG → base64 → deque + cola WS + Supabase), registra avistamiento en estadísticas, loguea evento en Supabase con cooldown 30 s por persona-cámara | Bucle, throttled |
| `dibujar_overlay(frame, rostros, telefonos, alerta_activa)` | Dibujo OpenCV: caja color-persona + barra inferior con `nombre` y `dist ~X.Xm sim X.XX`; cajas rojas `TELEFONO`; banner rojo "¡ POSIBLE REPLAY CON TELEFONO !" durante 5 s | `refresco_rapido_cam`, `probar_reconocimiento` |
| `listar_camaras_json()` | Lista unificada USB+navegador con `{id, nombre, tipo, activa, senal_hace_ms}`; incluye USB apagadas (nombres vía pygrabber); ordena principal primero | `/api/camaras`, `/api/estado`, difusiones |
| `difundir(mensaje)` | Envía JSON a todos los WS del dashboard y poda los muertos | Cambios de cámaras, alertas, registros |
| `bucle_procesamiento()` | Corazón asíncrono (ver diagrama). Mantenimiento de USB cada 10 s; refresco rápido de todas las cámaras en paralelo (`asyncio.gather` sobre executors); análisis pesados lanzados solo si pasó `fps_reconocimiento_ms` por cámara; vacía `cola_eventos` hacia los WS | Startup event |
| `generador_mjpeg(cam_id)` | Generador async que emite cada JPEG nuevo de `cache_display` (poll 0.08 s) | `/api/stream/{cam_id}` |

**Endpoints HTTP (20):**

| Método | Ruta | Cuerpo | Respuesta / efecto |
|---|---|---|---|
| GET | `/` | — | `index.html` |
| GET | `/movil` | — | `movil.html` |
| GET | `/api/stream/{cam_id}` | — | Stream MJPEG con overlay |
| GET | `/api/estado` | — | `{motor_listo, supabase_configurado, personas_registradas, camaras_conectadas}` |
| GET | `/api/camaras` | — | `{camaras[], camara_principal_id}` |
| POST | `/api/camara_principal` | `{cam_id}` | Fija principal en config y difunde |
| POST | `/api/camara/{cam_id}/estado` | `{activa}` | **Apaga/enciende cámara.** USB: libera el dispositivo (`cap.release()`) o lo reabre; persiste en `camaras_desactivadas`. Nav: runtime-only. Reasigna principal si la apagada era la principal |
| GET | `/api/estadisticas` | — | Resumen de personas en escena (últimos 90 s) |
| GET | `/api/alertas` | — | Últimas 50 alertas en memoria (captura base64) |
| GET | `/api/personas` | — | Personas registradas + `color_hex` |
| POST | `/api/color_persona/{nombre}` | `{color}` | Guarda color override en config |
| GET | `/api/ultimos_reconocimientos` | — | Últimos 15 eventos tipo reconocimiento |
| DELETE | `/api/persona/{nombre}` | — | Borra persona + sus fotos Storage + embeddings; limpia color |
| GET | `/api/mis_detecciones/{nombre}` | — | Últimas 10 detecciones de esa persona |
| GET | `/api/probar_reconocimiento/{cam_id}` | — | Análisis instantáneo sin overlay: para depurar umbral/galería |
| POST | `/api/camaras/reescanear` | — | `mantenimiento_forzado()` (ignora cooldowns) |
| POST | `/api/camaras/reiniciar_usb` | — | Cierra todo y reescanea desde cero |
| POST | `/api/registrar` | `{nombre, codigo, imagenes[dataURL]}` | Valida `CODIGO_SECRETO_REGISTRO` si está definido; detecta rostro en cada imagen, extrae embedding; exige ≥3 válidas; guarda en Supabase e invalida caché de galería |
| WS | `/ws/dashboard` | — | Saluda con `{tipo:"camaras"}` + `{tipo:"alertas_historico"}`; luego recibe pushes `camaras`, `alerta`, `registro` |
| WS | `/ws/camara_navegador` | binario JPEG | Saludo JSON `{rol:"camara", nombre}` → registra `nav_xxxxxxxx`; recibe frames. *(Backend operativo; la UI que lo usaba fue retirada)* |

Además monta estáticos en `/web` y añade `MiddlewareSinCache` (`Cache-Control: no-store` a toda respuesta HTTP — evita ver JS/CSS viejos).

**Cierre limpio:** el evento `shutdown` llama `gestor_usb.cerrar_todas()` → todas las cámaras quedan libres para otras aplicaciones.

### 3.3 `app/gestor_camaras_usb_locales.py` — Vida y muerte de las USB

Diseño: **una cámara = un objeto + un hilo lector**. El gestor nunca toca hardware en su propio hilo.

| Elemento | Qué hace |
|---|---|
| `BACKENDS` | `[("DSHOW", CAP_DSHOW), ("MSMF", CAP_MSMF)]` — orden de apertura en Windows |
| `COOLDONW_AUSENTE_S=60` / `COOLDONW_FALLO_S=20` | Si un índice no existe o falló, no se reintenta antes de ese tiempo (evita martillar dispositivos medio colgados) |
| `listar_nombres_dispositivos_windows()` | Nombres amigables ("Integrated Webcam") vía `pygrabber` (DirectShow) |
| `class CamaraUSB` | `cam_id="usb_i"`. `_abrir_una_vez()` prueba backends, fija 640×480, exige un frame válido. `abrir()` lanza el **hilo lector** `_bucle()` (sale tras 400 fallos consecutivos ≈ 20 s sin señal). `leer()` devuelve copia bajo lock. `desactivar()` corta el hilo; `liberar_en_hilo()` hace `cap.release()` fuera de línea (nunca congela al llamador) |
| `class GestorCamarasUSB` | Ver abajo |

Métodos del gestor:

- `mantenimiento()` — llamado cada 10 s: retira cámaras apagadas manualmente, sin señal inicial (>25 s), señal perdida (>6 s) o inactivas (liberación en hilo aparte); luego escanea índices `0..max_indices` respetando `apagados`, `indices_permitidos` y cooldowns, y abre las nuevas encontradas.
- `mantenimiento_forzado()` — limpia cooldowns y escanea ya (botón "Detectar cámaras").
- `apagar_usb(i)` / `encender_usb(i)` — interruptores manuales; encender prueba y abre inmediatamente.
- `reiniciar_todo()`, `todas()`, `cerrar_todas()`.

El set público `apagados` es lo que el servidor persiste en `config["camaras_desactivadas"]`.

### 3.4 `app/motor_reconocimiento_facial_yunet_sface.py` — Cerebro visual

| Elemento | Detalle |
|---|---|
| `_descargar_si_falta(url, destino)` | Descarga ONNX desde opencv_zoo si no existe (>10 KB) — requiere internet **solo la primera vez** |
| `UMBRAL_COSINE = 0.363` | Similitud mínima para "es la misma persona" (0 = idéntico en la escala de cv2) |
| `__init__` | Descarga modelos, crea candado `threading.Lock`, `_preparar()` instancia `FaceDetectorYN` y `FaceRecognizerSF` |
| `detectar_rostros(frame)` | YuNet; reajusta `setInputSize` si cambia la resolución; devuelve `[{fila(YuNet), bbox(x,y,w,h), score}]` |
| `extraer_embedding(frame, fila)` | `alignCrop` + `feature` → vector 128D float32 |
| `refrescar_galeria(funcion_carga, segundos_validez)` | Caché de la galería de conocidos (el servidor usa TTL 10 s) para no golpear Supabase en cada frame |
| `identificar(embedding)` | Máxima similitud coseno contra toda la galería; ≥ umbral ⇒ `{persona_id, nombre, similitud, conocido:True}` |
| `mejor_coincidencia(embedding)` | Igual pero sin exigir umbral (para el panel de diagnóstico) |
| `analizar_frame(frame)` | **Punto de entrada thread-safe**: candado → detectar + extraer + identificar cada rostro |

### 3.5 `app/detector_replay_telefono_yolo.py` — Anti-replay

YOLOv8/v11-nano preentrenado en COCO filtrando **clase 67 = "cell phone"**.

- `detectar_telefonos(frame)`: reduce a ancho 480, `predict(imgsz=384, conf=0.35)`, reescala cajas → `[(x,y,w,h)]`.
- `_expander(bbox, factor)`: agranda una caja (se usa factor 0.6 alrededor del rostro).
- `hay_replay_con_telefonos(telefonos, rostros)`: si alguna caja de teléfono solapa una cara expandida ⇒ posible foto/pantalla → dict con ambas bboxes.
- `hay_replay(frame, rostros)`: atajo que detecta + cruza.

### 3.6 `app/calculador_estadisticas_persona.py` — Quién hay en escena

- `estimar_distancia_m(bbox)`: óptica pinhole simple → `(focal_px × altura_rostro_cm) / alto_px_del_rostro`.
- `_identidad_desconocida(embedding)`: agrupa desconocidos parecidos (coseno ≥ 0.40) en centroides → etiquetas "Desconocido #1", "#2"… consistentes en vez de infinitos nombres.
- `registrar_avistamiento(...)`: mantiene sesión por `cámara::identidad` (validez 8 s); acumula `tiempo_en_escena_s`, `veces_detectado`, última distancia/confianza/similitud; promueve a conocido si luego se identifica.
- `limpiar_viejos()` / `resumen_actual()`: purga >90 s y devuelve lista ordenada por actividad reciente.

### 3.7 `app/conexion_supabase_rostros.py` — Persistencia

Cliente singleton creado desde `.env` (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`). Si falta configuración, **todas las funciones devuelven vacío/None sin romper** (degradación elegante).

| Función | Tabla/Bucket |
|---|---|
| `registrar_persona(nombre, codigo, embeddings, imagenes_jpeg)` | Inserta en `personas`, sube fotos a Storage `rostros-registrados/{persona_id}/rostro_i.jpg`, inserta filas en `rostros_entrenados` (embedding como JSON) |
| `cargar_galeria()` | Join `personas`×`rostros_entrenados` → `[{persona_id, nombre, embedding(np.array)}]` |
| `guardar_evento(persona_id, nombre, camara, confianza, distancia, tipo)` | `eventos_deteccion` (errores tragados) |
| `guardar_alerta_replay(camara, captura_jpeg, detalle)` | Sube captura + inserta en `alertas_replay` |
| `contar_personas` / `listar_personas` | Lecturas simples |
| `ultimos_reconocimientos(limite)` / `detecciones_de_persona(nombre, limite)` | Historial |
| `eliminar_persona(nombre)` | Borra carpeta Storage de cada id, filas de `rostros_entrenados` y `personas` |

### 3.8 `app/generador_certificado_https.py` — HTTPS para el celular

- `ip_local_de_red()`: truco del socket UDP hacia 8.8.8.8 (no envía tráfico) para saber qué IP LAN usar.
- `generar_certificado_si_falta()`: RSA-2048 autofirmado válido **10 años** con SAN `localhost`, `127.0.0.1` y la IP local → `certs/cert.pem` + `certs/key.pem`. Necesario porque `getUserMedia` (cámara del celular) exige contexto seguro.

### 3.9 `app/gestor_camaras_navegador_remotas.py` — Cámaras remotas (opcional)

51 líneas: `CamaraNavegador` guarda el último JPEG recibido por WS bajo lock; `GestorCamarasNavegador.registrar/dar_baja/todas()` (vivas si frames hace <4 s). Hoy el backend sigue aceptando `/ws/camara_navegador`, pero ninguna página envía frames — quedó como infraestructura disponible.

---

## 4. Frontend — `pagina_web/`

Sin frameworks ni build. Dos apps independientes que comparten estilos base.

### 4.1 `index.html` + `aplicacion.js` — Dashboard

Estructura HTML: cabecera (logo + chips de estado motor/Supabase/personas), navegación por pestañas (**Reconocimiento**, **Cámaras**, **Registro**, **Alertas**, **Personas**), y una vista por pestaña. El video principal lleva cluster HUD (badge EN VIVO + reloj) y botón de pantalla completa.

Funciones principales de `aplicacion.js` (por área):

| Área | Funciones | Consumen |
|---|---|---|
| Conectividad | `conectarDashboard` (WS + reconexión), `refrescarEstado` | `/ws/dashboard`, `/api/estado` |
| Video | `renderizarVideos` (grid + badges + fullscreen), `pintarFilmstrip`, `pintarGrid` (tarjetas con stream, chip APAGADA, botones), `principalElegida`, `alternarFullscreen`, `camarasActivas` | `/api/stream/{id}`, mensajes WS `camaras` |
| Control de cámaras | `detectarCamaras`→`controlarCamaras` (reescanear), `reiniciarUSB`, `hacerPrincipal`, `cambiarEstadoCam` | `/api/camaras/reescanear`, `/api/camaras/reiniciar_usb`, `/api/camara_principal`, `/api/camara/{id}/estado` |
| Reconocimiento en vivo | `cargarStats`, `pintarStats` (barras de confianza, avatares con `coloresPersona`), `ticRelojHud`, `confPct`, `haceCuanto`, `estaSinSenal` | `/api/estadisticas` |
| Registro multi-persona | `activarCamaraRegistro`, `fuenteCapturaRegistro`, `actualizarSelectorRegistro`, `opcionesDeVideoLocal`, `pedirCamara`, `mensajeAmigableCamara`, `enviarRegistro` (lote de personas capturadas → una petición) | `/api/registrar` |
| Alertas | `recibirAlerta`, `crearToast`, `abrirModalAlerta`/`cerrarModalAlerta`, `agregarAlertaLista`, `pintarAlertas`, `notificarRegistro`, `sonarPitidoCorto` (WebAudio, sin archivos) | WS `alerta`/`registro`, `alertas_historico` |
| Personas/Historial | `cargarColores` (+ selector de color por persona), `eliminarPersona`, `cargarUltimosRecs` | `/api/personas`, `/api/color_persona/{nombre}`, `DELETE /api/persona/{nombre}`, `/api/ultimos_reconocimientos` |
| Diagnóstico | `probarReconocimiento` (muestra mejor coincidencia aunque esté bajo umbral) | `/api/probar_reconocimiento/{id}` |

### 4.2 `movil.html` + `aplicacion_movil.js` — Modo celular

Página pensada para el teléfono: selector de cámara (frontal/trasera via `enumerateDevices` agrupado), vista previa local, formulario de **autoregistro** (`registrarme`: captura varias muestras → `POST /api/registrar`) y sección "Mi actividad" (`verMiActividad` + `refrescarDetecciones` contra `/api/mis_detecciones/{nombre}` y `/api/ultimos_reconocimientos`). Requiere HTTPS (por eso el puerto 8443).

### 4.3 `estilos_meet.css` — Sistema de diseño

Tema oscuro minimalista con tokens:

```css
--fondo:#0f0f0f · --panel:#171717 · --tarjeta:#242424 · --input:#1a1a1a
--borde:#2e2e2e / #3a3a3a · --texto:#fafafa / --texto-suave:#a8a8a8 / #808080
--acento:#22d3ee (cyan) · --error:#ff5555 · --warning:#ffaa33
Tipografías: Inter (UI) + JetBrains Mono (datos numéricos)
```
Incluye alias legacy (`--borde`, `--verde`, `--rojo`, `--naranja`, `--azul`) que consume `movil.html`, scrollbars personalizados, estados `focus-visible` y `prefers-reduced-motion`.

El dashboard y la página móvil llevan además un **fondo animado** (`#bg-canvas` + `.bg-veil`, script inline antes de `</body>`): simulación Gray-Scott (patrón de Turing) a 112×112 celdas, renderizada en un canvas fijo con el color del acento; se elige un preset feed/kill al azar en cada carga, respeta `prefers-reduced-motion` y todo el contenido real usa `z-index ≥ 2` para quedar por encima.

---

## 5. Base de datos (`sql_supabase.sql`)

| Tabla | Columnas clave | Uso |
|---|---|---|
| `personas` | `id uuid pk`, `nombre`, `codigo_registro`, `creado_en` | Registro maestro |
| `rostros_entrenados` | `persona_id fk cascade`, `embedding jsonb` (128 floats), `origen`, `imagen_url` | Galería de comparación |
| `eventos_deteccion` | `persona_id fk set null`, `nombre_detectado`, `camara`, `confianza`, `distancia_aprox_m`, `tipo` (`reconocimiento`/`desconocido`), `creado_en` | Historial (índice por fecha desc) |
| `alertas_replay` | `camara`, `captura_url`, `detalle`, `atendida`, `creado_en` | Alertas anti-spoofing |
| Bucket `rostros-registrados` | público | Fotos de rostros y capturas de alertas |

RLS habilitado en las 4 tablas: solo el backend (service key) accede; el público no puede leer/escribir directamente.

---

## 6. Configuración

### `config_sistema.json`

| Clave | Defecto | Significado |
|---|---|---|
| `camara_principal_id` | `"usb_0"` | Cámara destacada en el dashboard (se auto-repara si desaparece) |
| `fps_reconocimiento_ms` | `300` | Intervalo de análisis pesado por cámara (menor = más CPU) |
| `frames_entre_chequeo_telefono` | `10` | Frecuencia relativa del chequeo YOLO (cada N análisis) |
| `umbral_similitud_rostro` | `0.36` | Umbral coseno para considerar "conocido" |
| `focal_px_aproximada` | `550` | Focal estimada para calcular distancia |
| `altura_rostro_cm` | `23.0` | Altura media de rostro para la misma cuenta |
| `resolucion_captura` | `[640,480]` | Referencia de resolución |
| `calidad_jpeg_stream` | `70` | Calidad JPEG del stream (menor = más fluido) |
| `max_cameras_usb_escaneo` | `4` | Índices 0..N-1 probados al escanear |
| `camaras_usb_indices` | `[]` | Vacío = escaneo libre; con ids = whitelist |
| `camaras_desactivadas` | `[]` | USB apagadas manualmente (persisten entre reinicios) |
| `colores_personas` | `{}` | Overrides `{"Nombre":"#22d3ee"}` |

### `.env`

```
SUPABASE_URL=https://xxxx.supabase.co      # obligatorio para persistencia
SUPABASE_SERVICE_KEY=eyJ...                # service role (SECRETO)
CODIGO_SECRETO_REGISTRO=miclave            # opcional: exige código al registrar
PUERTO_SERVIDOR=8000                       # opcional (solo afecta HTTP directo)
```

Sin Supabase configurado el sistema **arranca igual** (reconoce a nadie conocido, no guarda historial).

---

## 7. Cómo se complementan los módulos (capas)

```
Capa de presentación   pagina_web/*          → solo habla HTTP/WS con el servidor
Capa de orquestación   servidor_web_principal→ posee estado, bucles, overlay y API
Capa de dispositivos   gestor_camaras_*      → abstraen "de dónde sale un frame BGR/JPEG"
Capa de inteligencia   motor_facial + detector_replay + calculador_estadisticas
                       (puras: entran frames, salen resultados; no saben de red)
Capa de persistencia   conexion_supabase     → única que habla con la nube
Capa de arranque       iniciar_servidores + generador_certificado
```

Reglas de dependencia: `presentación → orquestación → {dispositivos, inteligencia, persistencia}`. Ningún módulo de inteligencia importa a otro de dispositivos, y solo el servidor conoce la config global — cambiar de cámara o de proveedor de BD toca una sola capa.
