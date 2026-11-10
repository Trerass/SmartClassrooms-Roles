"""Casos de error de la API, sin cargar modelos. Ejecutar: python test_api_errores.py"""
import io
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

import roles_salon as rs

# Se reemplazan los modelos para que la prueba tarde segundos y no necesite TensorFlow ni PyTorch.
sys.modules["keras"] = MagicMock()
rs.configurar_tf = lambda: None
rs.get_yolo = lambda pesos: None
rutas_reales = rs.rutas
rs.rutas = lambda raiz: {**rutas_reales(raiz), "yolo_roles": Path("no-existe.pt")}
import app as api  # noqa: E402  (debe importarse con los modelos reemplazados)

rs.rutas = rutas_reales
client = TestClient(api.app, raise_server_exceptions=False)


def sin_personas(fuente):
    return rs.cargar_imagen(fuente), [], Counter({r: 0 for r in [*rs.CLASSES, "indeterminado"]})


def enviar(datos):
    return client.post("/analizar", files={"foto": ("foto.jpg", datos, "image/jpeg")})


def imagen(formato):
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 40).convert("RGB").save(buf, format=formato)
    return buf.getvalue()


api.analizar_foto = sin_personas
jpg = imagen("JPEG")

r = enviar(jpg)
assert r.status_code == 200, r.text
assert r.json()["total"] == 0
assert [x["rol"] for x in r.json()["roles"]] == ["Estudiante", "Aseo", "Seguridad"]  # sin Indeterminado si es 0

# Archivos que no son una imagen válida: 400 con un mensaje que la página pueda mostrar.
ihdr_truncado = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x02IHDR\x00\x00\x00\x00\x00\x00"
for nombre, datos in {"vacío": b"", "texto": b"hola", "cortado": jpg[:40], "ihdr": ihdr_truncado}.items():
    r = enviar(datos)
    assert r.status_code == 400, (nombre, r.status_code, r.text)
    assert isinstance(r.json()["detail"], str), nombre

r = enviar(b"x" * (api.MAX_BYTES + 1))
assert r.status_code == 413, r.text

r = client.post("/analizar")
assert r.status_code == 422, r.text

# Un archivo dañado nunca debe convertirse en un error del servidor.
rng = random.Random(1)
for base in (jpg, imagen("PNG"), imagen("WEBP")):
    for _ in range(300):
        datos = bytearray(base)
        for _ in range(rng.randint(1, 12)):
            datos[rng.randrange(min(300, len(datos)))] = rng.randrange(256)
        r = enviar(bytes(datos))
        assert r.status_code in (200, 400), (r.status_code, r.text)


# Si falla el modelo, no se le echa la culpa a la foto.
def falla(error):
    def analizar(fuente):
        raise error

    return analizar


for error in (RuntimeError("modelo"), OSError("disco")):
    api.analizar_foto = falla(error)
    r = enviar(jpg)
    assert r.status_code == 500, (r.status_code, r.text)
    assert r.json()["detail"] == "No se pudo analizar la foto. Inténtalo de nuevo."

print("OK")
