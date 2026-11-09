import os
import sys
import threading
import time
import webbrowser

import uvicorn

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.generador_certificado_https import (
        ARCHIVO_CERT,
        ARCHIVO_KEY,
        generar_certificado_si_falta,
        ip_local_de_red,
    )
else:
    from .generador_certificado_https import (
        ARCHIVO_CERT,
        ARCHIVO_KEY,
        generar_certificado_si_falta,
        ip_local_de_red,
    )

PUERTO_HTTP = int(os.getenv("PUERTO_SERVIDOR", "8000"))
PUERTO_HTTPS = 8443


def _correr_servidor(config):
    uvicorn.Server(config).run()


def main():
    from app.servidor_web_principal import app  # noqa: E402

    config_http = uvicorn.Config(
        app, host="0.0.0.0", port=PUERTO_HTTP, log_level="warning"
    )
    hilo_http = threading.Thread(target=_correr_servidor, args=(config_http,), daemon=True)

    hilo_https = None
    if generar_certificado_si_falta():
        config_https = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=PUERTO_HTTPS,
            ssl_certfile=ARCHIVO_CERT,
            ssl_keyfile=ARCHIVO_KEY,
            log_level="warning",
        )
        hilo_https = threading.Thread(target=_correr_servidor, args=(config_https,), daemon=True)

    hilo_http.start()
    if hilo_https is not None:
        hilo_https.start()

    ip = ip_local_de_red()
    print("=" * 62, flush=True)
    print(f"  PC:        http://localhost:{PUERTO_HTTP}", flush=True)
    if hilo_https is not None:
        print(f"  CELULAR:   https://{ip}:{PUERTO_HTTPS}   (acepta el aviso de certificado)", flush=True)
    else:
        print(f"  CELULAR:   http://{ip}:{PUERTO_HTTP} (sin HTTPS: la cámara del celular no funcionará)", flush=True)
    print("  Ctrl+C para detener", flush=True)
    print("=" * 62, flush=True)

    for _ in range(60):
        time.sleep(0.5)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{PUERTO_HTTP}/api/estado", timeout=1)
            break
        except Exception:
            continue
    try:
        webbrowser.open(f"http://localhost:{PUERTO_HTTP}")
        print("Navegador abierto en http://localhost:%d" % PUERTO_HTTP, flush=True)
    except Exception:
        pass

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nDeteniendo servidores…", flush=True)


if __name__ == "__main__":
    main()
