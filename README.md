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

Abre `http://localhost:5173`. Para abrirla desde un celular en la misma red WiFi, usa `pnpm dev --host` y entra con la IP del PC. Las fotos no se conservan al terminar el análisis.

Prueba de la API: `python test_app.py` dentro de `vision-service`.

## Modelo

| Etapa | Modelo | Estado |
|---|---|---|
| Detección de personas | YOLOv8m preentrenado en COCO (clase `person`) | Se descarga solo |
| Clasificación del rol | EfficientNetB0 con transfer learning | `vision-service/data_roles/modelo/clasificador_roles.keras` |

Resultado del clasificador en el conjunto de prueba (74 recortes): accuracy 0,797 y F1 macro 0,787.

## Reentrenar el clasificador

`vision-service/roles_salon.py` tiene los subcomandos `descargar`, `recortar`, `dividir`, `entrenar` y `contar`. El dataset no está en el repositorio porque las imágenes vienen de la web y no se pueden redistribuir; va en `vision-service/data_roles/`.

## Entrenar YOLOv8m con los roles

Alternativa a las dos etapas: YOLOv8m detecta a la persona y da su rol en un solo paso. El etiquetado corre en cualquier PC; el entrenamiento necesita GPU (en un PC sin GPU tarda alrededor de una hora por época), por eso se hace en Google Colab.

Dentro de `vision-service`, con el entorno activado:

```powershell
python roles_salon.py descargar --por-busqueda 150   # opcional: más fotos
python roles_salon.py etiquetar                      # cajas automáticas
```

No vuelvas a ejecutar `recortar`: borra los recortes ya revisados.

Abre `data_roles/revision_yolo/` y borra las fotos con cajas malas o con alguien que no tenga el rol de la carpeta. Después:

```powershell
python roles_salon.py dividir-yolo
tar -a -cf yolo.zip -C data_roles yolo
```

En un cuaderno de Colab con GPU, sube `yolo.zip` y ejecuta:

```python
!pip install -q ultralytics==8.4.201
!unzip -q yolo.zip
!yolo detect train model=yolov8m.pt data=/content/yolo/data.yaml imgsz=640 epochs=80 batch=16 patience=20 freeze=10 name=roles exist_ok=True
```

Descarga `best.pt` (la ruta aparece al final del entrenamiento) a `vision-service` y compáralo con el sistema actual:

```powershell
python roles_salon.py comparar --pesos best.pt
```

Si gana, cópialo como `data_roles/modelo/roles_yolo.pt`: la API lo usa automáticamente al reiniciarse. Para volver a las dos etapas, borra ese archivo.

`comparar` mide el error de rol en las fotos de internet del test. Para saber si funciona en un salón de verdad, prueba también con fotos de salones.
