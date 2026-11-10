"""API que recibe una foto y devuelve las personas contadas por rol y la imagen anotada."""
import base64
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import roles_salon as rs

MAX_BYTES = 15 * 1024 * 1024
P = rs.rutas(Path(__file__).parent / "data_roles")

if P["yolo_roles"].exists():  # YOLOv8m entrenado con roles reemplaza a las dos etapas
    from ultralytics import YOLO

    roles = YOLO(P["yolo_roles"])

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
        img, dets, conteo = analizar_foto(io.BytesIO(data))
    except (OSError, Image.DecompressionBombError):  # archivo que no es imagen, truncado o enorme
        raise HTTPException(400, "El archivo no es una imagen válida (usa JPG, PNG o WebP).")

    buf = io.BytesIO()
    rs.dibujar_resultados(img, dets, conteo).save(buf, format="JPEG", quality=90)
    return {
        "total": len(dets),
        "roles": [
            {"rol": rs.ROLE_LABELS[r], "color": rs.ROLE_COLORS[r], "personas": n}
            for r, n in conteo.items()
            if n > 0 or r != "indeterminado"
        ],
        "imagen": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    }
