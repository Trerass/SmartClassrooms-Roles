"""Casos de error de la API, sin cargar modelos. Ejecutar: python test_api_errores.py"""
import io
import random
import sys
from collections import Counter
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

import roles_salon as rs

# Se reemplazan los modelos para que la prueba tarde segundos y no necesite TensorFlow ni PyTorch.
sys.modules["keras"] = MagicMock()
rs.configurar_tf = lambda: None
rs.get_yolo = lambda pesos: None
import app as api  # noqa: E402  (debe importarse con los modelos reemplazados)

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
assert r.json()["detecciones"] == [] and (r.json()["ancho"], r.json()["alto"]) == (64, 64)


def dos_personas(fuente):
    det = lambda x, rol, conf: dict(box=[x, 5.4, x + 10.6, 30.0], role=rol, conf=conf, objects=["laptop"])
    dets = [det(40.0, "persona_aseo", 0.9876), det(2.0, "indeterminado", 0.5)]
    return rs.cargar_imagen(fuente), dets, Counter(d["role"] for d in dets)


# Cada persona llega con su caja en enteros, numerada de izquierda a derecha.
api.analizar_foto = dos_personas
d = enviar(jpg).json()["detecciones"]
assert [x["rol"] for x in d] == ["Indeterminado", "Aseo"], d
assert d[1] == {"rol": "Aseo", "color": rs.ROLE_COLORS["persona_aseo"], "confianza": 0.988,
                "caja": [40, 5, 51, 30], "objetos": ["laptop"]}, d[1]
api.analizar_foto = sin_personas

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

# Con el salón, el conteo queda disponible para el agente (solo números, no la foto).
api.agente.ANALISIS.clear()
api.analizar_foto = dos_personas
r = client.post("/analizar", files={"foto": ("foto.jpg", jpg, "image/jpeg")}, data={"salon": "38-102"})
assert r.status_code == 200, r.text
assert api.agente.ANALISIS["38-102"]["total"] == 2
assert api.agente.ANALISIS["38-102"]["por_rol"]["Aseo"] == 1

# Quitar la foto en la página borra el conteo; repetirlo o hacerlo sin conteo no falla.
assert client.delete("/analisis/38-102").status_code == 204
assert "38-102" not in api.agente.ANALISIS
assert client.delete("/analisis/38-102").status_code == 204

# /preguntar: sin clave, 503 con un mensaje claro; si el LLM falla, 502; pregunta vacía, 400.
def sin_clave():
    raise RuntimeError("Falta la clave del modelo.")


api.agente.crear_cliente = sin_clave
r = client.post("/preguntar", json={"pregunta": "¿Cuántos hay?"})
assert r.status_code == 503 and "clave" in r.json()["detail"], r.text
api.agente.crear_cliente = lambda: object()
api.agente.preguntar = lambda pregunta, cliente: (_ for _ in ()).throw(ConnectionError("red"))
r = client.post("/preguntar", json={"pregunta": "¿Cuántos hay?"})
assert r.status_code == 502, r.text
api.agente.preguntar = lambda pregunta, cliente: {"respuesta": "Hay 2.", "pasos": [], "limite_alcanzado": False}
assert client.post("/preguntar", json={"pregunta": "¿Cuántos hay?"}).json()["respuesta"] == "Hay 2."
assert client.post("/preguntar", json={"pregunta": "  "}).status_code == 400

print("OK")
