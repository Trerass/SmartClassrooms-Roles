"""Prueba de la API. Ejecutar: python test_app.py"""
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)

foto = (Path(__file__).parent / "foto_salon.jpg").read_bytes()
r = client.post("/analizar", files={"foto": ("foto.jpg", foto, "image/jpeg")})
assert r.status_code == 200, r.text
body = r.json()
assert body["total"] > 0
assert sum(x["personas"] for x in body["roles"]) == body["total"]
assert len(body["detecciones"]) == body["total"]
for d in body["detecciones"]:
    x1, y1, x2, y2 = d["caja"]
    assert 0 <= x1 < x2 <= body["ancho"] and 0 <= y1 < y2 <= body["alto"], d

# Las personas pequeñas de una foto de poca resolución también cuentan.
buf = io.BytesIO()
Image.open(io.BytesIO(foto)).resize((277, 277)).save(buf, format="JPEG", quality=92)
r = client.post("/analizar", files={"foto": ("chica.jpg", buf.getvalue(), "image/jpeg")})
assert r.json()["total"] == body["total"], r.text

# Salón con 15 estudiantes (varios tapados al fondo) y 1 persona de aseo con su balde.
aseo = (Path(__file__).parent / "fotos_prueba" / "salon_aseo.webp").read_bytes()
r = client.post("/analizar", files={"foto": ("aseo.webp", aseo, "image/webp")})
roles = {x["rol"]: x["personas"] for x in r.json()["roles"]}
assert r.json()["total"] == 16, roles
assert roles["Aseo"] == 1, roles

r = client.post("/analizar", files={"foto": ("x.jpg", b"esto no es una imagen", "image/jpeg")})
assert r.status_code == 400, r.text

print("OK", body["total"], [(x["rol"], x["personas"]) for x in body["roles"]])
