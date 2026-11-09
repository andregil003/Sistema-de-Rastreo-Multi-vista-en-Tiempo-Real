# Sistema de Reconocimiento Facial Multi-cámara

Sistema web estilo Google Meet que:
- Conecta **todas las cámaras** de tu laptop (webcams USB) + **celulares vía navegador** como fuentes de video.
- **Registra rostros con nombre** desde el navegador del celular y los **reconoce en vivo** frente a cualquier cámara.
- Muestra **estadísticas por persona**: distancia aproximada, tiempo en escena, veces detectado, confianza.
- Detecta **posibles transmisiones de video con teléfono** (replay attack) usando YOLOv11 y lanza alertas automáticas con sonido.
- Guarda todo en **Supabase** (personas, embeddings faciales, eventos, capturas de alertas).

## Estructura del proyecto

```
├── app/
│   ├── iniciar_servidores.py                 ← PUNTO DE ENTRADA (HTTP 8000 + HTTPS 8443)
│   ├── servidor_web_principal.py             ← API FastAPI y WebSocket
│   ├── generador_certificado_https.py        ← Certificado autofirmado para el celular
│   ├── gestor_camaras_usb_locales.py        ← Enumera y lee webcams USB
│   ├── gestor_camaras_navegador_remotas.py  ← Recibe frames de celulares/navegadores
│   ├── motor_reconocimiento_facial_yunet_sface.py  ← Detección YuNet + embeddings SFace
│   ├── detector_replay_telefono_yolo.py     ← Alertas de replay con teléfono (YOLOv11n)
│   ├── calculador_estadisticas_persona.py   ← Distancia, tiempo en escena, conteos
│   └── conexion_supabase_rostros.py         ← Base de datos Supabase
├── pagina_web/
│   ├── index.html / estilos_meet.css / aplicacion.js
├── modelos/
│   ├── yolo11n.pt                           ← Detector de teléfonos
│   ├── face_detection_yunet_2023mar.onnx    ← Detector de rostros (auto-descargado)
│   └── face_recognition_sface_2021dec.onnx  ← Embeddings faciales (auto-descargado)
├── sql_supabase.sql                         ← Esquema a pegar en Supabase
├── config_sistema.json                      ← Umbrales y ajustes
├── ARQUITECTURA_Y_CODIGO.md                 ← Doc técnica: cada archivo, función y flujo
├── MANUAL_DE_USO.md                         ← Guía de operación completa
└── .env                                     ← Credenciales (NO subir a git)
```

> 📚 **Documentación detallada**: [`ARQUITECTURA_Y_CODIGO.md`](ARQUITECTURA_Y_CODIGO.md) (arquitectura, módulos, endpoints, base de datos) · [`MANUAL_DE_USO.md`](MANUAL_DE_USO.md) (instalación, inicio/apagado, uso diario, internet, problemas).

## Instalación

```powershell
python -m pip install -r requirements.txt
```

## Configurar Supabase (una sola vez)

1. Crea cuenta gratis en https://supabase.com → **New project**.
2. Abre **SQL Editor** → pega TODO el contenido de `sql_supabase.sql` → **Run**.
3. Ve a **Project Settings → API** y copia:
   - `Project URL` → al archivo `.env` en `SUPABASE_URL`
   - `service_role key` (secreta, no la anon) → `SUPABASE_SERVICE_KEY`
4. Registro abierto: `CODIGO_SECRETO_REGISTRO` puede quedar **vacío** (cualquiera se registra escribiendo su nombre). Si escribes un valor, ese será el PIN obligatorio.

## Ejecutar

```powershell
python app\iniciar_servidores.py
```

(también vale `python -m app.iniciar_servidores` desde la raíz)

Esto levanta dos servidores a la vez y **abre tu navegador automáticamente**:
- **PC**: http://localhost:8000
- **Celular (misma red WiFi)**: https://IP-DE-TU-LAPTOP:8443 → toca "Avanzado" → "Continuar al sitio" (aviso normal del certificado local). Sin HTTPS el navegador del celular bloquea la cámara.

> La primera vez Windows preguntará permiso de firewall para Python: aceptar para redes privadas. Si el celular no conecta, ejecuta en PowerShell como Administrador:
> ```powershell
> New-NetFirewallRule -DisplayName "Reconocimiento Facial 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private,Public
> New-NetFirewallRule -DisplayName "Reconocimiento Facial 8443" -Direction Inbound -LocalPort 8443 -Protocol TCP -Action Allow -Profile Private,Public
> ```

## Uso

| Sección | Qué hacer |
|---|---|
| **Reconocimiento** | Escenario grande = cámara principal; las demás en la tira inferior (clic para intercambiar). Botón de **pantalla completa** arriba a la derecha (el panel de personas queda flotando sobre el video). Panel derecho: estadísticas en vivo + últimos reconocimientos (historial desde Supabase). Las cajas sobre el video llevan el nombre y una barra de color propia de cada persona (editable en Registro → Colores). Cámaras sin señal por más de 5s se marcan "SIN SEÑAL". |
| **Cámaras** | Lista de todas las cámaras USB detectadas. Botones "Detectar cámaras" y "Reiniciar cámaras USB". Cada tarjeta tiene su interruptor **Apagar/Activar**: apagar libera el dispositivo (queda libre para otras apps), se recuerda entre reinicios y las apagadas aparecen atenuadas con botón Activar. La cámara principal se elige con "Hacer principal". |
| **Registro** | Desde el celular o PC: escribe tu nombre → elige cámara en el selector (**la webcam local o cualquier cámara del sistema**, sin error de "ocupada") → Registrarme. Es **multi-persona**: tras guardar puedes registrar a la siguiente sin reactivar nada. En "Colores de reconocimiento" cambias el color de cada persona y la eliminas (basurita). |
| **Modo celular** (/movil) | Página simple para teléfonos: registrarse (con cámara propia o del sistema), registrar varias personas seguidas, y ver tu actividad (cuándo te vieron y desde qué cámara). |
| **Alertas** | Si alguien muestra un teléfono transmitiendo video frente a una cámara: mini-toast discreta abajo a la derecha (con botón "Ver captura"), 1 pitido corto y registro en Supabase. |

## Ajustes útiles (`config_sistema.json`)

| Clave | Significado |
|---|---|
| `fps_reconocimiento_ms` | Cada cuánto se analiza cada cámara (300 ms por defecto) |
| `umbral_similitud_rostro` | 0.363 estricto; súbelo si hay falsos reconocimientos |
| `focal_px_aproximada` | Calibra la distancia estimada (550 típico a 640x480) |
| `frames_entre_chequeo_telefono` | Cada cuántos análisis corre YOLO de teléfonos |
