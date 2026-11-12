#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lógica del taller de transfer learning, en dos etapas:

    Foto ──► [1] YOLOv8 (preentrenado, detecta "persona")
         ──► recorte por persona
         ──► [2] EfficientNetB0 + transfer learning (estudiante / persona_aseo / persona_seguridad)
         ──► conteo por rol + imagen anotada

Flujo de uso (cada paso es un subcomando):

    python roles_salon.py descargar     # 1. baja fotos de la web (por clase)
    python roles_salon.py recortar      # 2. YOLO recorta cada persona
    #   >>> REVISA A MANO data_roles/recortes/ y borra los recortes incorrectos <<<
    python roles_salon.py dividir       # 3. train / validation / test (agrupado por foto)
    python roles_salon.py entrenar      # 4. base congelada + fine-tuning + evaluación
    python roles_salon.py contar foto_salon.jpg   # 5. cuenta personas por rol

Alternativa: YOLOv8m entrenado directamente con los tres roles (ver README):

    python roles_salon.py etiquetar     # cajas automáticas y fotos para revisar
    #   >>> REVISA A MANO data_roles/revision_yolo/ y borra las fotos mal etiquetadas <<<
    python roles_salon.py dividir-yolo  # dataset en formato YOLO + data.yaml
    yolo detect train model=yolov8m.pt data=data_roles/yolo/data.yaml ...
    python roles_salon.py comparar      # dos etapas vs. YOLO con roles
    python roles_salon.py contar salon.jpg --mostrar

Si ya tienes un modelo entrenado (clasificador_roles.keras + class_names.json),
solo necesitas el paso 5.

Dependencias:
    pip install -U ddgs ultralytics tensorflow keras scikit-learn matplotlib pillow requests numpy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = (224, 224)
PERSON_CLASS = 0  # id de 'person' en COCO

CLASSES = ["estudiante", "persona_aseo", "persona_seguridad"]

# Términos en español e inglés; mencionan implementos y ropa que debe aprender el modelo.
SEARCHES = {
    "estudiante": [
        "estudiante con mochila en salón de clase",
        "estudiantes con cuadernos sentados en el aula",
        "student with backpack and notebook classroom",
        "universitario estudiando en el salón",
        # Ropa oscura y de pie: sin esto el modelo confunde "camiseta negra" con uniforme de aseo/seguridad.
        "estudiante universitario camiseta negra exponiendo frente al tablero",
        "estudiantes de pie presentando proyecto en clase",
        "university student black t-shirt presenting in front of class",
        "college students standing giving presentation classroom",
        "student wearing black hoodie standing in classroom",
        "estudiante con ropa oscura de pie en el pasillo de la universidad",
    ],
    "persona_aseo": [
        "persona de aseo limpiando salón de clase con trapeadora",
        "señora del aseo con escoba en colegio",
        "personal de servicios generales limpiando colegio",
        "janitor mopping school classroom",
        "cleaning staff uniform broom school",
        "auxiliar de aseo con uniforme y carro de limpieza",
        "operaria de aseo trapeando pasillo universidad",
        "persona de servicios generales barriendo aula",
        "school custodian cleaning hallway",
        "cleaning lady uniform mop bucket office",
        "janitor portrait uniform cleaning cart",
    ],
    "persona_seguridad": [
        "guardia de seguridad uniforme en colegio",
        "vigilante de seguridad en universidad",
        "security guard uniform school hallway",
        "guardia de seguridad dentro de un salón de clase",
        "vigilante privado con chaleco en colegio",
        "guarda de seguridad con radio en universidad",
        "school security officer standing",
        "campus security guard portrait",
        "security guard in school corridor students",
        "uniformed security guard university entrance",
    ],
}

# Para aseo/seguridad somos estrictos: si la foto tiene muchas personas, la etiqueta es ambigua.
MAX_PERSONS_IN_IMAGE = {"estudiante": 10, "persona_aseo": 2, "persona_seguridad": 2}
MAX_CROPS_PER_IMAGE = {"estudiante": 4, "persona_aseo": 1, "persona_seguridad": 1}

# COCO no tiene escoba/trapeadora: estos objetos son solo "evidencia" que se muestra
# junto a cada persona; NO alteran la decisión del clasificador.
OBJECTS_OF_INTEREST = {24: "mochila", 26: "bolso", 63: "laptop", 73: "libro/cuaderno"}

ROLE_COLORS = {
    "estudiante": "#1f77b4",
    "persona_aseo": "#2ca02c",
    "persona_seguridad": "#d62728",
    "indeterminado": "#7f7f7f",
}
ROLE_LABELS = {
    "estudiante": "Estudiante",
    "persona_aseo": "Aseo",
    "persona_seguridad": "Seguridad",
    "indeterminado": "Indeterminado",
}


def rutas(raiz: str) -> dict:
    raiz = Path(raiz)
    return {
        "raw": raiz / "escenas",        # fotos completas
        "crops": raiz / "recortes",     # una persona por imagen
        "split": raiz / "dataset",      # train / validation / test
        "modelo_dir": raiz / "modelo",
        "modelo": raiz / "modelo" / "clasificador_roles.keras",
        "clases": raiz / "modelo" / "class_names.json",
        "pesos_base": raiz / "modelo" / "pesos_base_congelada.weights.h5",
        "etiquetas": raiz / "etiquetas_yolo",
        "revision": raiz / "revision_yolo",
        "yolo": raiz / "yolo",
        "yolo_roles": raiz / "modelo" / "roles_yolo.pt",
        "solo_train": raiz / "solo_train.txt",  # fotos de salones reales: siempre a train
    }


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades generales
# ──────────────────────────────────────────────────────────────────────────────
def contar_imagenes(root) -> dict:
    return {f.name: len(list(f.glob("*.jpg"))) for f in sorted(Path(root).iterdir()) if f.is_dir()}


def source_id(path) -> str:
    """'0012__p1.jpg' -> '0012' (foto de la que salió el recorte)."""
    return Path(path).name.split("__p")[0]


def cargar_imagen(fuente: str) -> Image.Image:
    if str(fuente).startswith("http"):
        import requests
        r = requests.get(fuente, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
    # Las fotos de celular guardan la rotación en EXIF.
    return ImageOps.exif_transpose(Image.open(fuente)).convert("RGB")


_YOLO = None


def get_yolo(weights: str):
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(weights)
        # Una predicción de calentamiento ANTES de cargar TensorFlow: en Linux, si PyTorch inicializa
        # sus librerías después de TF, las dos copias de LLVM chocan y el proceso se cae
        # ("LLVM ERROR: inconsistency in registered CommandLine options").
        _YOLO.predict(Image.new("RGB", (64, 64)), verbose=False)
    return _YOLO


def detectar_personas(yolo, img: Image.Image, conf: float = 0.40):
    res = yolo.predict(img, classes=[PERSON_CLASS], conf=conf, verbose=False)[0]
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()


def recortar_con_margen(img: Image.Image, box, pad: float = 0.08) -> Image.Image:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - pad * w)
    y1 = max(0, y1 - pad * h)
    x2 = min(img.width, x2 + pad * w)
    y2 = min(img.height, y2 + pad * h)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def configurar_tf():
    """TensorFlow y PyTorch (YOLO) comparten GPU: evita que TF reserve toda la memoria."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    return tf


# ──────────────────────────────────────────────────────────────────────────────
# 1. Descarga de imágenes
# ──────────────────────────────────────────────────────────────────────────────
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cmd_descargar(args):
    import requests
    from ddgs import DDGS

    P = rutas(args.raiz)
    raw = P["raw"]
    raw.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    seen = {_sha256_bytes(p.read_bytes()) for p in raw.glob("*/*.jpg")}

    for clase, queries in SEARCHES.items():
        if args.clases and clase not in args.clases:
            continue
        class_dir = raw / clase
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = len(list(class_dir.glob("*.jpg")))
        print(f"\n=== {clase} ===")

        for q in queries:
            print(f"Buscando: {q!r}")
            try:
                results = DDGS().images(query=q, safesearch="moderate",
                                        max_results=args.por_busqueda, license_image="any")
            except Exception as e:
                print("  Error en búsqueda:", e)
                continue

            for r in results:
                url = r.get("image")
                if not url:
                    continue
                try:
                    resp = session.get(url, timeout=8)
                    resp.raise_for_status()
                    with Image.open(io.BytesIO(resp.content)) as im:
                        im = im.convert("RGB")
                        if min(im.size) < 200:
                            continue
                        buf = io.BytesIO()
                        im.save(buf, format="JPEG", quality=90)
                    data = buf.getvalue()
                    h = _sha256_bytes(data)
                    if h in seen:
                        continue
                    while (class_dir / f"{saved:04d}.jpg").exists():  # por si se borraron fotos
                        saved += 1
                    (class_dir / f"{saved:04d}.jpg").write_bytes(data)
                    seen.add(h)
                    saved += 1
                except Exception:
                    pass
            print(f"  Total guardadas hasta ahora: {saved}")
            time.sleep(1)

    print("\nDescarga terminada:", contar_imagenes(raw))
    print("Licencias: úsalas solo con fines educativos y no redistribuyas el dataset.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Detección de personas y recorte automático
# ──────────────────────────────────────────────────────────────────────────────
def cmd_recortar(args):
    P = rutas(args.raiz)
    if not P["raw"].exists():
        sys.exit(f"No existe {P['raw']}. Ejecuta primero: python roles_salon.py descargar")
    yolo = get_yolo(args.yolo)

    # Sin --solo-nuevas se rehace todo y se pierde la revisión a mano.
    if P["crops"].exists() and not args.solo_nuevas:
        shutil.rmtree(P["crops"])

    for class_dir in sorted(P["raw"].iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASSES:
            continue
        name = class_dir.name
        if args.clases and name not in args.clases:
            continue
        out_dir = P["crops"] / name
        out_dir.mkdir(parents=True, exist_ok=True)
        s = dict(fotos=0, sin_persona=0, ambiguas=0, recortes=0)
        # procesadas.txt recuerda qué fotos ya pasaron por aquí: si una no tiene recorte, se descartó a mano.
        registro = out_dir / "procesadas.txt"
        hechas = set()
        if args.solo_nuevas and registro.exists():
            hechas = set(registro.read_text(encoding="utf-8").split())
        elif args.solo_nuevas:  # carpetas anteriores al registro
            ultima = max((source_id(f) for f in out_dir.glob("*.jpg")), default="")
            hechas = {f.stem for f in class_dir.glob("*.jpg") if f.stem <= ultima}

        for path in sorted(class_dir.glob("*.jpg")):
            if path.stem in hechas:
                continue
            hechas.add(path.stem)
            s["fotos"] += 1
            with Image.open(path) as im:
                img = im.convert("RGB")
            boxes, _ = detectar_personas(yolo, img, args.conf)
            relevant = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= args.min_lado]

            if not relevant:
                s["sin_persona"] += 1
                continue
            if len(relevant) > MAX_PERSONS_IN_IMAGE[name]:
                s["ambiguas"] += 1
                continue

            relevant.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            for k, box in enumerate(relevant[:MAX_CROPS_PER_IMAGE[name]]):
                recortar_con_margen(img, box).save(out_dir / f"{path.stem}__p{k}.jpg",
                                                   format="JPEG", quality=90)
                s["recortes"] += 1
        registro.write_text("\n".join(sorted(hechas)) + "\n", encoding="utf-8")
        print(f"{name:20s}", s)

    print("\nRecortes por clase:", contar_imagenes(P["crops"]))
    print(f"\n>>> Revisa {P['crops']} y borra a mano los recortes que NO correspondan a la clase")
    print("    (p. ej. un estudiante dentro de persona_aseo, ilustraciones, carteles con texto).")
    print("    Meta: >= 80-100 recortes limpios por clase. Luego ejecuta: dividir")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Split agrupado por foto de origen (evita data leakage)
# ──────────────────────────────────────────────────────────────────────────────
def particion_fija(clase: str, recortes_por_foto: dict) -> dict:
    """Reparte las fotos de una clase en test/validation/train con el 15/15/70 % de sus RECORTES.

    Las fotos se ordenan por un hash fijo de su nombre y se llenan los grupos en ese orden. Así el
    reparto no depende del azar: agregar o borrar fotos solo puede mover las que quedan en el límite
    entre dos grupos, y el test se mantiene casi igual entre entrenamientos.
    """
    h = lambda foto: hashlib.sha256(f"{clase}/{foto}".encode()).hexdigest()
    total = sum(recortes_por_foto.values())
    grupos, acumulado = {}, 0
    for foto in sorted(recortes_por_foto, key=h):
        n = recortes_por_foto[foto]
        centro = (acumulado + n / 2) / total  # una foto va al grupo donde cae la mitad de sus recortes
        grupos[foto] = "test" if centro < 0.15 else "validation" if centro < 0.30 else "train"
        acumulado += n
    return grupos


def cmd_dividir(args):
    P = rutas(args.raiz)
    if not P["crops"].exists():
        sys.exit(f"No existe {P['crops']}. Ejecuta primero: python roles_salon.py recortar")
    if P["split"].exists():
        shutil.rmtree(P["split"])
    # Fotos que van siempre a train (p. ej. salones reales, escasos frente a los de internet).
    # Una línea por foto con la forma 'clase/foto', por ejemplo 'estudiante/0133'.
    solo_train = set()
    if P["solo_train"].exists():
        solo_train = {l.strip() for l in P["solo_train"].read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")}

    for class_dir in sorted(P["crops"].iterdir()):
        if not class_dir.is_dir():
            continue
        groups = defaultdict(list)
        for f in class_dir.glob("*.jpg"):
            groups[source_id(f)].append(f)

        split_keys = {"train": [], "validation": [], "test": []}
        forzadas = {k for k in groups if f"{class_dir.name}/{k}" in solo_train}
        reparto = particion_fija(class_dir.name, {k: len(v) for k, v in groups.items() if k not in forzadas})
        reparto.update({k: "train" for k in forzadas})
        for k in sorted(groups):
            split_keys[reparto[k]].append(k)
        for split_name, ks in split_keys.items():
            target = P["split"] / split_name / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for k in ks:
                for src in groups[k]:
                    shutil.copy2(src, target / src.name)

    for split in ["train", "validation", "test"]:
        print(f"{split:11s}", contar_imagenes(P["split"] / split))

    sources = {s: {(p.parent.name, source_id(p)) for p in (P["split"] / s).rglob("*.jpg")}
               for s in ["train", "validation", "test"]}
    assert not (sources["train"] & sources["validation"])
    assert not (sources["train"] & sources["test"])
    assert not (sources["validation"] & sources["test"])
    print("\n✓ Ninguna foto de origen se comparte entre train, validation y test.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Entrenamiento: base congelada -> fine-tuning -> selección por validation
# ──────────────────────────────────────────────────────────────────────────────
def construir_modelo(num_classes: int):
    import keras
    from keras import layers

    # Sin volteo vertical ni cambios fuertes de color: el color del uniforme es una señal útil.
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = base(x, training=False)  # EfficientNet ya normaliza internamente (entra 0-255)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="clasificador")(x)
    return keras.Model(inputs, outputs), base


def plot_history(h, title="", save_path=None, mostrar=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.history["accuracy"], label="train accuracy")
    axes[0].plot(h.history["val_accuracy"], label="val accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy (Train vs Val)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(h.history["loss"], label="train loss")
    axes[1].plot(h.history["val_loss"], label="val loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss (Train vs Val)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print("Curvas de aprendizaje guardadas en", save_path)
    if mostrar:
        plt.show()
    plt.close(fig)


def intervalo_bootstrap(y_true, y_pred, n: int = 2000, seed: int = SEED):
    """IC del 95 % de la accuracy remuestreando el test: con un test pequeño, la nota tiene margen de error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs = [np.mean(y_true[i] == y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def evaluar(model, dataset, class_names, titulo, png_path, mostrar=False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                                 confusion_matrix, f1_score)

    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_pred = model.predict(dataset, verbose=0).argmax(axis=1)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    lo, hi = intervalo_bootstrap(y_true, y_pred)
    print(f"\n--- {titulo} ---  accuracy={acc:.4f} (IC 95 %: {lo:.3f}-{hi:.3f})  F1 macro={f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    if mostrar:
        plt.show()
    plt.close(fig)
    print("Matriz de confusión guardada en", png_path)
    return acc, f1


def cmd_entrenar(args):
    if not getattr(args, "mostrar_graficas", False):
        import matplotlib
        matplotlib.use("Agg")

    tf = configurar_tf()
    import keras
    from sklearn.utils.class_weight import compute_class_weight
    from keras import layers

    P = rutas(args.raiz)
    if not (P["split"] / "train").exists():
        sys.exit(f"No existe {P['split']}. Ejecuta primero: python roles_salon.py dividir")
    P["modelo_dir"].mkdir(parents=True, exist_ok=True)

    kw = dict(image_size=IMG_SIZE, batch_size=args.batch, label_mode="int")
    train_ds = keras.utils.image_dataset_from_directory(P["split"] / "train", shuffle=True, seed=SEED, **kw)
    val_ds = keras.utils.image_dataset_from_directory(P["split"] / "validation", shuffle=False, **kw)
    test_ds = keras.utils.image_dataset_from_directory(P["split"] / "test", shuffle=False, **kw)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_labels = np.concatenate([y.numpy() for _, y in train_ds])
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)
    class_weight = {i: float(v) for i, v in enumerate(w)}
    print("Clases:", class_names)
    print("Train por clase:", dict(zip(class_names, np.bincount(train_labels))))
    print("class_weight:", {class_names[i]: round(v, 2) for i, v in class_weight.items()})

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model, base_model = construir_modelo(num_classes)

    # --- Etapa A: base congelada, solo se entrena la cabeza ---
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    h_base = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epocas_base, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True)])
    model.save_weights(P["pesos_base"])
    plot_history(h_base, "Base congelada: Curvas de Aprendizaje",
                 save_path=P["modelo_dir"] / "curvas_base_congelada.png",
                 mostrar=args.mostrar_graficas)
    acc_b, f1_b = evaluar(model, test_ds, class_names, "Antes del fine-tuning",
                          P["modelo_dir"] / "confusion_antes.png",
                          mostrar=args.mostrar_graficas)

    # --- Etapa B: fine-tuning de las últimas N capas (BatchNorm sigue congelada) ---
    base_model.trainable = True
    for layer in base_model.layers[:-args.n_descongeladas]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    print("Capas entrenables de la base:",
          sum(l.trainable for l in base_model.layers), "/", len(base_model.layers))

    model.compile(optimizer=keras.optimizers.Adam(1e-5),   # LR mucho menor
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    h_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epocas_ft, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                 restore_best_weights=True)])
    plot_history(h_ft, "Fine-tuning: Curvas de Aprendizaje",
                 save_path=P["modelo_dir"] / "curvas_fine_tuning.png",
                 mostrar=args.mostrar_graficas)
    acc_a, f1_a = evaluar(model, test_ds, class_names, "Después del fine-tuning",
                          P["modelo_dir"] / "confusion_despues.png",
                          mostrar=args.mostrar_graficas)

    print(f"\n{'':26s}{'accuracy':>10s}{'F1 macro':>10s}")
    print(f"{'Antes del fine-tuning':26s}{acc_b:10.4f}{f1_b:10.4f}")
    print(f"{'Después del fine-tuning':26s}{acc_a:10.4f}{f1_a:10.4f}")

    # La selección del modelo final se hace con VALIDATION, no con test.
    best_base = min(h_base.history["val_loss"])
    best_ft = min(h_ft.history["val_loss"])
    print(f"\nMejor val_loss base congelada: {best_base:.4f} | fine-tuning: {best_ft:.4f}")
    if best_ft <= best_base:
        print("Modelo final: fine-tuning")
    else:
        model.load_weights(P["pesos_base"])
        print("Modelo final: base congelada")

    model.save(P["modelo"])
    P["clases"].write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {P['modelo']} y {P['clases']}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Inferencia: contar personas por rol en una imagen de salón
# ──────────────────────────────────────────────────────────────────────────────
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# min_side es pequeño a propósito: en fotos de poca resolución las personas del fondo miden 20-40 px.
def analizar_salon(fuente, yolo, model, class_names, conf_person=0.40, min_conf_role=0.60,
                   min_side=16, show_objects=True, imgsz=960):
    """Detecta personas, clasifica el rol de cada una y devuelve (imagen, detecciones, conteo)."""
    img = cargar_imagen(fuente)
    # A 640 px se pierden estudiantes del fondo y una persona con su balde cuenta doble;
    # 1280 ya inventa personas (ver fotos_prueba/).
    res = yolo.predict(img, conf=conf_person, imgsz=imgsz, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)

    person_idx = [i for i, c in enumerate(cls)
                  if c == PERSON_CLASS and min(boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]) >= min_side]

    detections = []
    if person_idx:
        crops = np.stack([
            np.array(recortar_con_margen(img, boxes[i]).resize(IMG_SIZE), dtype="float32")
            for i in person_idx
        ])
        probs = model.predict(crops, verbose=0)

        for j, i in enumerate(person_idx):
            k = int(probs[j].argmax())
            conf = float(probs[j][k])
            role = class_names[k] if conf >= min_conf_role else "indeterminado"

            objs = []
            if show_objects:
                x1, y1, x2, y2 = boxes[i]
                for b, c in zip(boxes, cls):
                    if c in OBJECTS_OF_INTEREST:
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            objs.append(OBJECTS_OF_INTEREST[c])
            detections.append(dict(box=[float(v) for v in boxes[i]], role=role,
                                   conf=conf, objects=sorted(set(objs))))

    conteo = Counter({r: 0 for r in list(class_names) + ["indeterminado"]})
    conteo.update(d["role"] for d in detections)
    return img, detections, conteo


def dibujar_resultados(img: Image.Image, detections, conteo) -> Image.Image:
    W, H = img.size
    lw = max(2, W // 400)
    fs = max(12, W // 70)
    font = _font(fs)

    resumen = "  |  ".join(f"{ROLE_LABELS.get(r, r)}: {n}" for r, n in conteo.items()
                           if n > 0 or r != "indeterminado")
    header = f"Personas: {len(detections)}   {resumen}"
    header_h = int(fs * 2.2)
    canvas = Image.new("RGB", (W, H + header_h), "#111111")
    canvas.paste(img, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, header_h // 2), header, fill="white", font=_font(int(fs * 1.2)), anchor="lm")

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        y1 += header_h
        y2 += header_h
        color = ROLE_COLORS.get(d["role"], "#000000")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = f"{ROLE_LABELS.get(d['role'], d['role'])} {d['conf']:.0%}"
        if d["objects"]:
            label += "\n+ " + ", ".join(d["objects"])
        l, t, r, b = draw.multiline_textbbox((0, 0), label, font=font)
        tw, th = r - l + 6, b - t + 6
        ty = y1 - th if y1 - th >= header_h else y1
        draw.rectangle([x1, ty, x1 + tw, ty + th], fill=color)
        draw.multiline_text((x1 + 3, ty + 3), label, fill="white", font=font)
    return canvas


def cmd_contar(args):
    P = rutas(args.raiz)
    modelo_path = Path(args.modelo) if args.modelo else P["modelo"]
    clases_path = Path(args.clases) if args.clases else P["clases"]
    if not modelo_path.exists() or not clases_path.exists():
        sys.exit(f"No encuentro el modelo ({modelo_path}) o las clases ({clases_path}).\n"
                 "Entrena primero con 'entrenar' o indica --modelo y --clases.")

    yolo = get_yolo(args.yolo)  # antes que TensorFlow (ver get_yolo)
    configurar_tf()
    import keras
    model = keras.saving.load_model(modelo_path)
    class_names = json.loads(clases_path.read_text(encoding="utf-8"))

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    resultados = []

    for fuente in args.imagenes:
        try:
            img, dets, conteo = analizar_salon(
                fuente, yolo, model, class_names,
                conf_person=args.conf_persona, min_conf_role=args.min_conf_rol,
                min_side=args.min_lado, show_objects=not args.sin_objetos)
        except Exception as e:
            print(f"[ERROR] {fuente}: {e}")
            continue

        print(f"\nImagen: {fuente}")
        print(f"  Personas detectadas: {len(dets)}")
        for rol, n in conteo.items():
            if rol != "indeterminado" or n > 0:
                print(f"  {ROLE_LABELS.get(rol, rol):14s}: {n}")

        nombre = Path(str(fuente).split("?")[0]).stem or "imagen"
        out_path = salida / f"{nombre}_roles.jpg"
        anotada = dibujar_resultados(img, dets, conteo)
        anotada.save(out_path, quality=92)
        print(f"  Imagen anotada: {out_path}")
        if args.mostrar:
            anotada.show()

        resultados.append(dict(imagen=str(fuente), total=len(dets), conteo=dict(conteo),
                               detecciones=dets, imagen_anotada=str(out_path)))

    if args.json and resultados:
        json_path = salida / "resultados.json"
        json_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultados guardados en {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. YOLOv8m con roles: auto-etiquetado, dataset e inferencia
# ──────────────────────────────────────────────────────────────────────────────
def linea_yolo(box, ancho, alto, clase_id) -> str:
    """Caja xyxy en píxeles -> 'clase cx cy w h' normalizado (formato de etiquetas YOLO)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return (f"{clase_id} {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} "
            f"{(x2 - x1) / ancho:.6f} {(y2 - y1) / alto:.6f}")


def validar_clases(nombres):
    """Evita activar unos pesos que no se entrenaron con los tres roles."""
    if set(nombres.values()) != set(CLASSES):
        sys.exit(f"Los pesos tienen las clases {sorted(nombres.values())}, se esperaban {CLASSES}.")


def cmd_etiquetar(args):
    P = rutas(args.raiz)
    if not P["raw"].exists():
        sys.exit(f"No existe {P['raw']}. Ejecuta primero: python roles_salon.py descargar")
    yolo = get_yolo(args.yolo)

    for clase_id, name in enumerate(CLASSES):
        lab_dir, rev_dir = P["etiquetas"] / name, P["revision"] / name
        lab_dir.mkdir(parents=True, exist_ok=True)
        rev_dir.mkdir(parents=True, exist_ok=True)
        # Si todos los recortes de una foto ya revisada se borraron, esa foto se descartó a mano.
        con_recorte = {source_id(f) for f in (P["crops"] / name).glob("*.jpg")}
        ultima_revisada = max(con_recorte, default="")
        s = dict(nuevas=0, sin_persona=0, ambiguas=0, descartadas=0, ilegibles=0)

        for path in sorted((P["raw"] / name).glob("*.jpg")):
            # Es incremental: volver a ejecutarlo no deshace lo que ya se revisó.
            if (lab_dir / f"{path.stem}.txt").exists():
                continue
            if path.stem <= ultima_revisada and path.stem not in con_recorte:
                s["descartadas"] += 1
                continue
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
            except Exception:  # un archivo dañado no debe detener todo el etiquetado
                s["ilegibles"] += 1
                continue
            boxes, _ = detectar_personas(yolo, img, args.conf)
            if len(boxes) == 0:
                s["sin_persona"] += 1
                continue
            if len(boxes) > MAX_PERSONS_IN_IMAGE[name]:
                s["ambiguas"] += 1
                continue
            # Se etiqueta a TODAS las personas: una sin etiqueta le enseña a YOLO que es fondo.
            lineas = [linea_yolo(b, img.width, img.height, clase_id) for b in boxes]
            (lab_dir / f"{path.stem}.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
            draw = ImageDraw.Draw(img)
            for b in boxes:
                draw.rectangle([float(v) for v in b], outline="#ff0000", width=max(2, img.width // 200))
            img.save(rev_dir / path.name, format="JPEG", quality=85)
            s["nuevas"] += 1
        print(f"{name:20s}", s)

    print("\nPara revisar:", contar_imagenes(P["revision"]))
    print(f">>> Revisa {P['revision']} y borra las fotos con cajas o personas que no correspondan")
    print("    (p. ej. un profesor en estudiante, o un estudiante en persona_aseo). Luego: dividir-yolo")


def cmd_dividir_yolo(args):
    P = rutas(args.raiz)
    if not P["revision"].exists():
        sys.exit(f"No existe {P['revision']}. Ejecuta primero: python roles_salon.py etiquetar")
    if not any(P["revision"].glob("*/*.jpg")):
        sys.exit(f"No quedan fotos en {P['revision']}: no se cambió el dataset.")
    if P["yolo"].exists():
        shutil.rmtree(P["yolo"])
    rng = random.Random(SEED)

    for name in CLASSES:
        stems = sorted(f.stem for f in (P["revision"] / name).glob("*.jpg"))
        rng.shuffle(stems)
        n_train, n_val = int(len(stems) * 0.70), int(len(stems) * 0.15)
        partes = {"train": stems[:n_train], "val": stems[n_train:n_train + n_val],
                  "test": stems[n_train + n_val:]}
        for split, ks in partes.items():
            (P["yolo"] / "images" / split).mkdir(parents=True, exist_ok=True)
            (P["yolo"] / "labels" / split).mkdir(parents=True, exist_ok=True)
            for k in ks:
                # Los números de foto se repiten entre clases: el prefijo evita colisiones.
                shutil.copy2(P["raw"] / name / f"{k}.jpg", P["yolo"] / "images" / split / f"{name}_{k}.jpg")
                shutil.copy2(P["etiquetas"] / name / f"{k}.txt", P["yolo"] / "labels" / split / f"{name}_{k}.txt")
        print(f"{name:20s}", {k: len(v) for k, v in partes.items()})

    # Sin 'path': Ultralytics usa la carpeta del yaml, así funciona igual aquí y en Colab.
    nombres = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (P["yolo"] / "data.yaml").write_text(
        f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{nombres}\n", encoding="utf-8")
    print("\nDataset listo en", P["yolo"])


def analizar_salon_yolo(fuente, yolo_roles, conf_person=0.25, min_conf_role=0.50):
    """Igual que analizar_salon, pero YOLO da la caja y el rol en un solo paso."""
    img = cargar_imagen(fuente)
    # agnostic_nms: la misma persona no puede contarse con dos roles.
    res = yolo_roles.predict(img, conf=conf_person, agnostic_nms=True, verbose=False)[0]
    detections = []
    for box, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        # Con poca confianza la persona sigue contando, sin rol: no baja el total.
        role = yolo_roles.names[int(c)] if p >= min_conf_role else "indeterminado"
        detections.append(dict(box=[float(v) for v in box], role=role, conf=float(p), objects=[]))
    conteo = Counter({r: 0 for r in [*CLASSES, "indeterminado"]})
    conteo.update(d["role"] for d in detections)
    return img, detections, conteo


def cmd_comparar(args):
    """Error de rol por foto en test. El total no se compara: las cajas de test salen del detector de las dos etapas."""
    P = rutas(args.raiz)
    pesos = Path(args.pesos) if args.pesos else P["yolo_roles"]
    if not pesos.exists() or not (P["yolo"] / "images" / "test").exists():
        sys.exit(f"Faltan los pesos ({pesos}) o el dataset. Ejecuta dividir-yolo y entrena primero.")

    from ultralytics import YOLO
    yolo, roles = get_yolo(args.yolo), YOLO(pesos)  # antes que TensorFlow (ver get_yolo)
    configurar_tf()
    import keras
    model = keras.saving.load_model(P["modelo"])
    class_names = json.loads(P["clases"].read_text(encoding="utf-8"))
    validar_clases(roles.names)
    metodos = {"Dos etapas": lambda f: analizar_salon(f, yolo, model, class_names),
               "YOLO con roles": lambda f: analizar_salon_yolo(f, roles)}

    errores = defaultdict(list)
    for img in sorted((P["yolo"] / "images" / "test").glob("*.jpg")):
        lineas = (P["yolo"] / "labels" / "test" / f"{img.stem}.txt").read_text(encoding="utf-8").splitlines()
        real = Counter(CLASSES[int(l.split()[0])] for l in lineas)
        for nombre, analizar in metodos.items():
            _, _, conteo = analizar(img)
            errores[nombre].append(sum(abs(conteo[r] - real[r]) for r in CLASSES))

    for nombre, e in errores.items():
        print(f"{nombre:16s} error medio de rol por foto: {np.mean(e):.3f}  (fotos: {len(e)})")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Detección de personas en salones y conteo por rol "
                    "(estudiante / aseo / seguridad).")
    ap.add_argument("--raiz", default="data_roles", help="Carpeta de trabajo (datos y modelo).")
    ap.add_argument("--yolo", default="yolov8m.pt",
                    help="Pesos de YOLO: yolov8n.pt (rápido), yolov8s.pt, yolov8m.pt (más preciso).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("descargar", help="Descarga fotos de la web por clase.")
    p.add_argument("--por-busqueda", type=int, default=50)
    p.add_argument("--clases", nargs="+", choices=CLASSES, help="Solo estas clases.")
    p.set_defaults(func=cmd_descargar)

    p = sub.add_parser("recortar", help="YOLO detecta personas y las recorta.")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--min-lado", type=int, default=96)
    p.add_argument("--clases", nargs="+", choices=CLASSES, help="Solo estas clases.")
    p.add_argument("--solo-nuevas", action="store_true",
                   help="Recorta solo las fotos nuevas y conserva los recortes ya revisados.")
    p.set_defaults(func=cmd_recortar)

    p = sub.add_parser("dividir", help="Split train/validation/test agrupado por foto.")
    p.set_defaults(func=cmd_dividir)

    p = sub.add_parser("entrenar", help="Transfer learning + fine-tuning + evaluación.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epocas-base", type=int, default=15)
    p.add_argument("--epocas-ft", type=int, default=8)
    p.add_argument("--n-descongeladas", type=int, default=20)
    p.add_argument("--mostrar-graficas", action="store_true",
                   help="Mostrar las ventanas de gráficas de accuracy/loss y matrices de confusión.")
    p.set_defaults(func=cmd_entrenar)

    p = sub.add_parser("contar", help="Cuenta personas por rol en una o varias imágenes.")
    p.add_argument("imagenes", nargs="+", help="Rutas o URLs de imágenes de salones.")
    p.add_argument("--modelo", help="Ruta al .keras (por defecto, el entrenado en --raiz).")
    p.add_argument("--clases", help="Ruta al class_names.json.")
    p.add_argument("--salida", default="resultados", help="Carpeta para las imágenes anotadas.")
    p.add_argument("--conf-persona", type=float, default=0.40,
                   help="Más alto = menos falsos positivos de persona.")
    p.add_argument("--min-conf-rol", type=float, default=0.60,
                   help="Más alto = más personas marcadas como 'indeterminado'.")
    p.add_argument("--min-lado", type=int, default=16,
                   help="Lado mínimo (px) de una persona para considerarla.")
    p.add_argument("--sin-objetos", action="store_true",
                   help="No mostrar mochila/bolso/laptop/libro como evidencia.")
    p.add_argument("--json", action="store_true", help="Guardar resultados.json.")
    p.add_argument("--mostrar", action="store_true", help="Abrir la imagen anotada.")
    p.set_defaults(func=cmd_contar)

    p = sub.add_parser("etiquetar", help="Cajas automáticas por clase y fotos para revisar.")
    p.add_argument("--conf", type=float, default=0.40)
    p.set_defaults(func=cmd_etiquetar)

    p = sub.add_parser("dividir-yolo", help="Dataset en formato YOLO a partir de las fotos revisadas.")
    p.set_defaults(func=cmd_dividir_yolo)

    p = sub.add_parser("comparar", help="Compara las dos etapas con YOLO entrenado con roles.")
    p.add_argument("--pesos", help="Pesos de YOLO con roles (por defecto, modelo/roles_yolo.pt).")
    p.set_defaults(func=cmd_comparar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
