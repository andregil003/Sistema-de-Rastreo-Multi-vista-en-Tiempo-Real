import datetime
import ipaddress
import os
import socket

RUTA_CERTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs")
ARCHIVO_CERT = os.path.join(RUTA_CERTS, "cert.pem")
ARCHIVO_KEY = os.path.join(RUTA_CERTS, "key.pem")


def ip_local_de_red():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generar_certificado_si_falta():
    if os.path.exists(ARCHIVO_CERT) and os.path.exists(ARCHIVO_KEY):
        return True
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        print("Falta 'cryptography' para HTTPS: python -m pip install cryptography", flush=True)
        return False

    ip = ip_local_de_red()
    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    sujetos = [
        x509.NameAttribute(NameOID.COMMON_NAME, "Reconocimiento Facial Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Sistema Facial"),
    ]
    nombre = x509.Name(sujetos)
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(ip)),
    ])

    ahora = datetime.datetime.now(datetime.timezone.utc)
    certificado = (
        x509.CertificateBuilder()
        .subject_name(nombre)
        .issuer_name(nombre)
        .public_key(clave.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(ahora - datetime.timedelta(days=1))
        .not_valid_after(ahora + datetime.timedelta(days=3650))
        .add_extension(san, critical=False)
        .sign(clave, hashes.SHA256())
    )

    os.makedirs(RUTA_CERTS, exist_ok=True)
    with open(ARCHIVO_KEY, "wb") as f:
        f.write(clave.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(ARCHIVO_CERT, "wb") as f:
        f.write(certificado.public_bytes(serialization.Encoding.PEM))

    print(f"Certificado HTTPS generado para IP {ip} (válido 10 años)", flush=True)
    return True
