"""API: analiza la foto de un salón (personas por rol) y responde preguntas con el agente."""
import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

import agente
import roles_salon as rs

MAX_BYTES = 15 * 1024 * 1024
P = rs.rutas(Path(__file__).parent / "data_roles")

yolo = rs.get_yolo("yolov8m.pt")  # antes que TensorFlow: si no, en Linux chocan sus librerías
rs.configurar_tf()
import keras  # noqa: E402  (configurar_tf debe correr antes)

model = keras.saving.load_model(P["modelo"])
class_names = json.loads(P["clases"].read_text(encoding="utf-8"))


def analizar_foto(fuente):
    return rs.analizar_salon(fuente, yolo, model, class_names)


app = FastAPI(title="Roles en salón")


# async a propósito: la inferencia corre en el event loop, una foto a la vez,
# porque YOLO no es seguro entre hilos. La foto no se conserva al terminar la petición.
@app.post("/analizar")
async def analizar(foto: UploadFile = File(...), salon: str | None = Form(None)):
    data = await foto.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "La foto supera los 15 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Exception:  # Pillow lanza varios tipos según cómo esté dañado el archivo
        raise HTTPException(400, "El archivo no es una imagen válida (usa JPG, PNG o WebP).")

    try:
        img, dets, conteo = analizar_foto(io.BytesIO(data))
    except Exception:
        logging.exception("Falló el análisis de una foto")
        raise HTTPException(500, "No se pudo analizar la foto. Inténtalo de nuevo.")
    if salon:  # el agente responde con el último conteo de cada salón (solo números, no la foto)
        agente.registrar_analisis(salon, len(dets), {rs.ROLE_LABELS[r]: n for r, n in conteo.items()})
    # La página dibuja los recuadros sobre la foto original; ancho y alto ya traen la rotación EXIF aplicada.
    return {
        "total": len(dets),
        "ancho": img.width,
        "alto": img.height,
        "roles": [
            {"rol": rs.ROLE_LABELS[r], "color": rs.ROLE_COLORS[r], "personas": n}
            for r, n in conteo.items()
            if n > 0 or r != "indeterminado"
        ],
        # De izquierda a derecha, para que la numeración se lea en el mismo orden que la foto.
        "detecciones": [
            {"rol": rs.ROLE_LABELS[d["role"]], "color": rs.ROLE_COLORS[d["role"]],
             "confianza": round(d["conf"], 3), "caja": [round(v) for v in d["box"]], "objetos": d["objects"]}
            for d in sorted(dets, key=lambda d: d["box"][0])
        ],
    }


@app.delete("/analisis/{salon}", status_code=204)
def borrar_analisis(salon: str):
    """La página llama a esto al quitar la foto de un salón: el agente ya no debe usar ese conteo."""
    agente.borrar_analisis(salon)


class Pregunta(BaseModel):
    pregunta: str


@app.post("/preguntar")
def preguntar(p: Pregunta):
    """Agente: responde sobre la ocupación de los salones usando herramientas. Devuelve también los pasos."""
    if not p.pregunta.strip():
        raise HTTPException(400, "Escribe una pregunta.")
    try:
        cliente = agente.crear_cliente()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    try:
        return agente.preguntar(p.pregunta[:500], cliente)
    except Exception:
        logging.exception("Falló el agente")
        raise HTTPException(502, "El modelo de lenguaje no respondió. Revisa la clave o inténtalo de nuevo.")
