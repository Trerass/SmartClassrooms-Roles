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
assert body["imagen"].startswith("data:image/jpeg;base64,")

# Las personas pequeñas de una foto de poca resolución también cuentan.
buf = io.BytesIO()
Image.open(io.BytesIO(foto)).resize((277, 277)).save(buf, format="JPEG", quality=92)
r = client.post("/analizar", files={"foto": ("chica.jpg", buf.getvalue(), "image/jpeg")})
assert r.json()["total"] == body["total"], r.text

r = client.post("/analizar", files={"foto": ("x.jpg", b"esto no es una imagen", "image/jpeg")})
assert r.status_code == 400, r.text

print("OK", body["total"], [(x["rol"], x["personas"]) for x in body["roles"]])
