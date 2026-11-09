# Manual de Uso — Sistema de Reconocimiento Facial Multi-cámara

> Guía práctica de operación. Para detalles técnicos ver [`ARQUITECTURA_Y_CODIGO.md`](ARQUITECTURA_Y_CODIGO.md).

---

## 1. ¿Qué es este sistema?

Un servidor que conecta varias **cámaras USB** a tu PC, detecta rostros en tiempo real, reconoce a las personas registradas (nombre, distancia aproximada, confianza) y avisa cuando alguien intenta **engañar a una cámara mostrando un celular** con una foto o video. Todo se controla desde un panel web en tu navegador, y las personas pueden registrarse a sí mismas desde su celular.

---

## 2. Requisitos

| Requisito | Detalle |
|---|---|
| Sistema | Windows 10/11 |
| Python | 3.9 – 3.12 |
| Cámaras | Webcams USB integradas o externas (probadas: Integrated Webcam, cámaras USB genéricas). Mientras más cámaras activas, más CPU consume |
| Hardware | ~45 MB para modelos + lo que ocupe el entorno Python (PyTorch/ultralytics son lo más pesado) |
| Internet | Solo para instalación inicial, Supabase y primera descarga de modelos (ver [sección 7](#7-¿necesita-internet)) |

## 3. Instalación (solo la primera vez)

1. **Dependencias Python** (en la carpeta del proyecto):
   ```
   pip install -r requirements.txt
   ```
   Instala: fastapi, uvicorn, opencv-python, numpy, ultralytics (YOLO), supabase, python-dotenv, pygrabber, websockets. Puede tardar varios minutos por PyTorch.

2. **Base de datos Supabase** (gratis en supabase.com):
   - Crea un proyecto → **SQL Editor → New query**, pega TODO el contenido de `sql_supabase.sql` y ejecuta. Crea 4 tablas + bucket de fotos.

3. **Secretos**: crea el archivo `.env` en la raíz:
   ```
   SUPABASE_URL=https://tuproyecto.supabase.co
   SUPABASE_SERVICE_KEY=eyJ...   (Project Settings → API → service_role)
   CODIGO_SECRETO_REGISTRO=tuclave     ← opcional, protege el registro
   ```
   ⚠️ La `service key` es secreta: no la subas a Git ni la compartas.

4. **Modelos**: ya están en `modelos\`. Si faltan, el sistema los descarga solo al primer arranque.

5. **Firewall** (si vas a usarlo desde el celular): permite los puertos TCP 8000 y 8443 para Python:
   ```
   netsh advfirewall firewall add rule name="Sistema Facial HTTP" dir=in action=allow protocol=TCP localport=8000
   netsh advfirewall firewall add rule name="Sistema Facial HTTPS" dir=in action=allow protocol=TCP localport=8443
   ```

## 4. Cómo INICIAR

```
cd "C:\Users\andre\Documents\GitHub\Sistema-de-Rastreo-Multi-vista-en-Tiempo-Real"
python app\iniciar_servidores.py
```

Verás:

```
==============================================================
  PC:        http://localhost:8000
  CELULAR:   https://192.168.X.XX:8443   (acepta el aviso de certificado)
  Ctrl+C para detener
==============================================================
Navegador abierto en http://localhost:8000
```

- El navegador se abre solo tras unos segundos (cuando `/api/estado` responde).
- **En tu PC**: entra a `http://localhost:8000`.
- **Desde el celular** (misma red WiFi): entra a la URL `https://...:8443` que imprime la consola. El navegador mostrará un aviso de certificado — es normal (certificado autofirmado): pulsa **Avanzado → Proceder al sitio**.
- Los primeros ~10 segundos el sistema escanea cámaras USB y carga modelos; los videos aparecen solos.
- Si cierras la ventana de consola, el sistema se apaga.

### Iniciar en segundo plano (sin ventana)
```powershell
Start-Process python -ArgumentList "app\iniciar_servidores.py" -WorkingDirectory "<ruta del proyecto>" -WindowStyle Hidden
```
Los logs quedan en `logs\servidor_out.log` y `logs\servidor_err.log`.

## 5. Cómo APAGAR

| Situación | Acción |
|---|---|
| Ventana de consola visible | `Ctrl+C` en ella (o cerrar la ventana) |
| Iniciado oculto (PowerShell) | `Get-CimInstance Win32_Process -Filter "Name='python.exe'" \| Where-Object { $_.CommandLine -match 'iniciar' } \| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }` |

Al apagarse, el servidor **libera todas las cámaras automáticamente** (quedan disponibles para Zoom, OBS, etc.). Las cámaras que apagaste manualmente siguen apagadas al reiniciar (queda guardado en `config_sistema.json`). Las personas registradas e historial viven en Supabase: nada se pierde.

## 6. Uso diario (pestaña por pestaña)

### 🎥 Reconocimiento
- Cada cámara activa se ve como tarjeta de video; la **principal** va grande arriba (cámbiala haciendo clic en otra miniatura).
- Sobre cada rostro: caja con el color asignado a esa persona + barra con `Nombre`, `dist ~X.Xm` y `sim X.XX`. Desconocidos = caja naranja.
- Badge **EN VIVO** (parpadea si la señal se congela), reloj, botón de pantalla completa.
- Panel derecho: personas en escena ahora mismo (tiempo en escena, detecciones, confianza).

### 📷 Cámaras
- **Detectar cámaras**: escaneo inmediato (útil al conectar una USB nueva).
- **Reiniciar cámaras USB**: cierra todo y reescanea desde cero (arregla cámaras colgadas).
- Cada tarjeta tiene interruptor **Apagar / Activar**: apagar libera el dispositivo de verdad (puedes usarlo en otra app) y queda recordado entre reinicios. Las apagadas se ven atenuadas con botón Activar.
- **Hacer principal**: destaca esa cámara en Reconocimiento.

### 👤 Registro
1. Elige cámara (USB o webcam del propio navegador).
2. Escribe nombre (+ código secreto si configuraste uno).
3. Captura **al menos 3 muestras** con buena luz y rostro centrado (acepta varias personas antes de enviar: usa "Agregar otra persona").
4. Enviar → guarda embeddings + fotos en Supabase. En segundos la persona ya es reconocida.

### 🔔 Alertas
Cuando YOLO ve un celular superpuesto a un rostro: suena pitido, aparece toast + modal con captura de pantalla, y queda en el historial (también persistido en Supabase). Cooldown: máximo 1 alerta cada 15 s por cámara.

### 👥 Personas / Historial
Lista de registrados con su color (personalizable con selector de color), eliminar registro completo, últimos reconocimientos y actividad por persona.

### 📱 Modo celular (`https://TU_IP:8443/movil`)
Página ligera para que cada quien se registre a sí mismo con la cámara frontal y luego consulte "Mi actividad".

## 7. ¿Necesita internet?

| Función | Sin internet |
|---|---|
| Ver streams, overlay, cambiar/apagar cámaras, estadísticas, alertas en pantalla, fullscreen | ✅ Funciona (red local no requiere internet) |
| Registro de personas nuevas | ❌ Requiere internet (Supabase) |
| **Reconocer conocidos** | ❌ Requiere internet: la galería facial vive en Supabase; sin conexión todo aparece como desconocido |
| Historial de eventos/alertas persistentes | ❌ Requiere internet (las alertas igual saltan en pantalla) |
| Primer arranque sin carpeta `modelos\` | ❌ Necesita internet para descargar los 3 modelos (~43 MB) |
| Fuentes/iconos de la página | Caen a fuentes del sistema (solo estética) |

> Resumen: **el video y el control funcionan siempre; el reconocimiento de conocidos y la persistencia dependen de Supabase**.

## 8. Problemas comunes

| Síntoma | Solución |
|---|---|
| No abre `http://localhost:8000` | Mira `logs\servidor_err.log`; puerto ocupado → cambia `PUERTO_SERVIDOR` en `.env`; revisa que Python esté instalado |
| Una cámara no aparece | Pulsa "Detectar cámaras". Si otra app la está usando no podrá abrirla (ciérrala). Tras un fallo espera 20–60 s (cooldown anti-bucle) |
| Apareció y desaparece una cámara | Señal perdida >6 s → la retira y reintenta solo; revisa cable/puerto USB |
| El celular no conecta | Misma WiFi, usa la URL `https://IP:8443` exacta que imprime la consola, firewall abierto (paso 5). IP cambia si cambias de red |
| Aviso de certificado peligroso en celular | Normal: certificado autofirmado local. Avanzado → Proceder |
| Todos salen "Desconocido #n" | Sin internet (galería vacía), mala luz/distancia, o umbral muy alto (`umbral_similitud_rostro`) — usa "Probar reconocimiento" en Registro para ver la mejor coincidencia y su similitud real |
| Va lento | Sube `fps_reconocimiento_ms` (ej. 400–500), baja calidad del stream, o desactiva cámaras que no uses |
| La webcam del navegador no pide permisos | Usa Chrome/Edge actualizado; desde celular solo funciona en HTTPS (puerto 8443) |
| Registré y no me reconoce | Espera ~10 s (la galería se refresca sola) o reinicia el servidor |

## 9. Notas de seguridad

- Certificado HTTPS **autofirmado**: cifrado real, pero solo tú podéis verificarlo. Úsalo en tu red local.
- `.env` contiene la clave maestra de Supabase: nunca la publiques ni hagas commit (está en `.gitignore`).
- RLS de Supabase bloquea acceso público directo a las tablas; solo el backend (service key) lee/escribe.
- Si defines `CODIGO_SECRETO_REGISTRO`, solo quien tenga el código puede registrar rostros.
- Estás gestionando datos biométricos: úsalos conforme a las normas que te apliquen y elimina registros que ya no necesites (botón Eliminar borra también sus fotos).
