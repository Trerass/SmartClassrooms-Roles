"""API que recibe una foto y devuelve las personas contadas por rol y la imagen anotada."""
import base64
import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import roles_salon as rs

MAX_BYTES = 15 * 1024 * 1024
P = rs.rutas(Path(__file__).parent / "data_roles")

if P["yolo_roles"].exists():  # YOLOv8m entrenado con roles reemplaza a las dos etapas
    from ultralytics import YOLO

    roles = YOLO(P["yolo_roles"])
    rs.validar_clases(roles.names)

    def analizar_foto(fuente):
        return rs.analizar_salon_yolo(fuente, roles)
else:
    rs.configurar_tf()
    import keras  # noqa: E402  (configurar_tf debe correr antes)

    model = keras.saving.load_model(P["modelo"])
    class_names = json.loads(P["clases"].read_text(encoding="utf-8"))
    yolo = rs.get_yolo("yolov8m.pt")

    def analizar_foto(fuente):
        return rs.analizar_salon(fuente, yolo, model, class_names)

app = FastAPI(title="Roles en salón")


# async a propósito: la inferencia corre en el event loop, una foto a la vez,
# porque YOLO no es seguro entre hilos. La foto no se conserva al terminar la petición.
@app.post("/analizar")
async def analizar(foto: UploadFile = File(...)):
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
        buf = io.BytesIO()
        rs.dibujar_resultados(img, dets, conteo).save(buf, format="JPEG", quality=90)
    except Exception:
        logging.exception("Falló el análisis de una foto")
        raise HTTPException(500, "No se pudo analizar la foto. Inténtalo de nuevo.")
    return {
        "total": len(dets),
        "roles": [
            {"rol": rs.ROLE_LABELS[r], "color": rs.ROLE_COLORS[r], "personas": n}
            for r, n in conteo.items()
            if n > 0 or r != "indeterminado"
        ],
        "imagen": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    }
