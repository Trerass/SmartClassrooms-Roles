# SmartClassrooms Roles

Sube una foto de un salón y la plataforma devuelve la imagen con cada persona remarcada y su rol: estudiante, aseo o seguridad.

```text
Foto → YOLOv8m (detecta personas) → recorte → EfficientNetB0 (clasifica el rol) → imagen anotada y conteo
```

Variante para clase, separada de [SmartClassrooms](https://github.com/Trerass/SmartClassrooms). Solo usa tres salones, definidos en `SALONES` de `frontend/src/app/App.tsx`.

## Ejecutar

Requiere Python 3.12, Node 20 o superior y pnpm.

Terminal 1, la API (la primera vez descarga `yolov8m.pt`; cargar los modelos tarda alrededor de un minuto):

```powershell
cd vision-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Terminal 2, la página:

```powershell
cd frontend
pnpm install
pnpm dev
```

Abre `http://localhost:5173`. Las fotos se procesan en memoria y no se guardan.

Prueba de la API: `python test_app.py` dentro de `vision-service`.

## Modelo

| Etapa | Modelo | Estado |
|---|---|---|
| Detección de personas | YOLOv8m preentrenado en COCO (clase `person`) | Se descarga solo |
| Clasificación del rol | EfficientNetB0 con transfer learning | `vision-service/data_roles/modelo/clasificador_roles.keras` |

Resultado del clasificador en el conjunto de prueba (74 recortes): accuracy 0,797 y F1 macro 0,787.

## Reentrenar

`vision-service/roles_salon.py` tiene los subcomandos `descargar`, `recortar`, `dividir`, `entrenar` y `contar`. El dataset no está en el repositorio porque las imágenes vienen de la web y no se pueden redistribuir; va en `vision-service/data_roles/`.
