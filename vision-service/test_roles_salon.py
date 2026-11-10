"""Pruebas del etiquetado y del dataset YOLO (sin modelos). Ejecutar: python test_roles_salon.py"""
import tempfile
from argparse import Namespace

import numpy as np
from PIL import Image

import roles_salon as rs

assert rs.linea_yolo([10, 20, 30, 100], 100, 200, 2) == "2 0.200000 0.300000 0.200000 0.400000"

rs.get_yolo = lambda pesos: None
rs.detectar_personas = lambda yolo, img, conf: (np.array([[1.0, 1.0, 5.0, 5.0]]), np.array([0.9]))

with tempfile.TemporaryDirectory() as d:
    P = rs.rutas(d)
    args = Namespace(raiz=d, yolo="x", conf=0.4)
    for clase, stems in {"estudiante": ["0000", "0001", "0002", "0003"],
                         "persona_aseo": ["0000"], "persona_seguridad": ["0000"]}.items():
        (P["raw"] / clase).mkdir(parents=True)
        for s in stems:
            Image.new("RGB", (20, 20)).save(P["raw"] / clase / f"{s}.jpg")
    # 0001 se descartó a mano en 'recortar'; 0003 es una foto nueva.
    (P["crops"] / "estudiante").mkdir(parents=True)
    for s in ["0000", "0002"]:
        Image.new("RGB", (20, 20)).save(P["crops"] / "estudiante" / f"{s}__p0.jpg")

    rs.cmd_etiquetar(args)
    etiquetas = {f.stem for f in (P["etiquetas"] / "estudiante").glob("*.txt")}
    assert etiquetas == {"0000", "0002", "0003"}, etiquetas

    # Lo que se borra en la revisión no vuelve al repetir 'etiquetar' y no entra al dataset.
    (P["revision"] / "estudiante" / "0002.jpg").unlink()
    rs.cmd_etiquetar(args)
    assert not (P["revision"] / "estudiante" / "0002.jpg").exists()

    rs.cmd_dividir_yolo(Namespace(raiz=d))
    imagenes = list((P["yolo"] / "images").glob("*/*.jpg"))
    nombres = {f.name for f in imagenes}
    # El mismo número de foto en dos clases no debe pisarse.
    assert nombres == {"estudiante_0000.jpg", "estudiante_0003.jpg",
                       "persona_aseo_0000.jpg", "persona_seguridad_0000.jpg"}, nombres
    for f in imagenes:
        assert (P["yolo"] / "labels" / f.parent.name / f"{f.stem}.txt").exists()
    assert len(imagenes) == len(nombres)  # ninguna foto en dos particiones

# Un archivo dañado se salta y el resto se etiqueta.
with tempfile.TemporaryDirectory() as d:
    P = rs.rutas(d)
    (P["raw"] / "estudiante").mkdir(parents=True)
    (P["raw"] / "estudiante" / "0000.jpg").write_bytes(b"no es una imagen")
    Image.new("RGB", (20, 20)).save(P["raw"] / "estudiante" / "0001.jpg")
    rs.cmd_etiquetar(Namespace(raiz=d, yolo="x", conf=0.4))
    assert {f.stem for f in (P["etiquetas"] / "estudiante").glob("*.txt")} == {"0001"}

# Si se borraron todas las fotos, el dataset anterior no se pierde.
with tempfile.TemporaryDirectory() as d:
    P = rs.rutas(d)
    (P["revision"] / "estudiante").mkdir(parents=True)
    P["yolo"].mkdir(parents=True)
    (P["yolo"] / "data.yaml").write_text("x")
    try:
        rs.cmd_dividir_yolo(Namespace(raiz=d))
        raise AssertionError("debía detenerse")
    except SystemExit:
        pass
    assert (P["yolo"] / "data.yaml").exists()

# Unos pesos con otras clases se rechazan.
rs.validar_clases(dict(enumerate(rs.CLASSES)))
try:
    rs.validar_clases({0: "person"})
    raise AssertionError("debía rechazarlos")
except SystemExit:
    pass

print("OK")
