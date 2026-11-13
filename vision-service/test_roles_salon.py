"""Pruebas del armado del dataset y las métricas (sin modelos). Ejecutar: python test_roles_salon.py"""
import tempfile
from argparse import Namespace

import numpy as np
from PIL import Image

import roles_salon as rs

rs.get_yolo = lambda pesos: None
rs.detectar_personas = lambda yolo, img, conf: (np.array([[1.0, 1.0, 5.0, 5.0]]), np.array([0.9]))

# 'recortar --solo-nuevas' no toca lo ya revisado: ni recortes borrados a mano ni otras clases.
with tempfile.TemporaryDirectory() as d:
    P = rs.rutas(d)
    for clase in ["estudiante", "persona_seguridad"]:
        (P["raw"] / clase).mkdir(parents=True)
        (P["crops"] / clase).mkdir(parents=True)
    for s in ["0000", "0001", "0002", "0003"]:
        Image.new("RGB", (200, 200)).save(P["raw"] / "persona_seguridad" / f"{s}.jpg")
    Image.new("RGB", (200, 200)).save(P["raw"] / "estudiante" / "0000.jpg")
    # 0000 se revisó y quedó; 0001 se borró a mano; 0002 y 0003 son nuevas.
    Image.new("RGB", (5, 5)).save(P["crops"] / "persona_seguridad" / "0000__p0.jpg")
    (P["crops"] / "persona_seguridad" / "procesadas.txt").write_text("0000\n0001\n")
    rs.detectar_personas = lambda yolo, img, conf: (np.array([[0.0, 0.0, 150.0, 150.0]]), np.array([0.9]))
    rs.cmd_recortar(Namespace(raiz=d, yolo="x", conf=0.4, min_lado=96,
                              clases=["persona_seguridad"], solo_nuevas=True))
    assert {f.name for f in (P["crops"] / "persona_seguridad").glob("*.jpg")} == \
        {"0000__p0.jpg", "0002__p0.jpg", "0003__p0.jpg"}
    assert Image.open(P["crops"] / "persona_seguridad" / "0000__p0.jpg").size == (5, 5)
    assert not list((P["crops"] / "estudiante").glob("*.jpg"))
    # Una segunda pasada no repite nada.
    (P["crops"] / "persona_seguridad" / "0002__p0.jpg").unlink()
    rs.cmd_recortar(Namespace(raiz=d, yolo="x", conf=0.4, min_lado=96,
                              clases=["persona_seguridad"], solo_nuevas=True))
    assert not (P["crops"] / "persona_seguridad" / "0002__p0.jpg").exists()

# La partición fija reparte 70/15/15 de los RECORTES, aunque cada foto tenga un número distinto de ellos.
fotos = {f"{i:04d}": 1 + i % 4 for i in range(120)}  # como estudiante: de 1 a 4 recortes por foto
reparto = rs.particion_fija("estudiante", fotos)
assert reparto == rs.particion_fija("estudiante", dict(fotos))  # siempre igual
recortes = {g: sum(n for f, n in fotos.items() if reparto[f] == g) for g in ["train", "validation", "test"]}
total = sum(fotos.values())
assert abs(recortes["test"] / total - 0.15) < 0.02 and abs(recortes["validation"] / total - 0.15) < 0.02, recortes
# Agregar una foto nueva mueve como mucho las fotos del límite entre grupos.
nuevo = rs.particion_fija("estudiante", {**fotos, "9999": 2})
assert sum(nuevo[f] != reparto[f] for f in fotos) <= 2
lo, hi = rs.intervalo_bootstrap([0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 1], n=300)
assert lo <= 5 / 6 <= hi

# 'dividir' manda siempre a train las fotos listadas en solo_train.txt.
with tempfile.TemporaryDirectory() as d:
    P = rs.rutas(d)
    (P["crops"] / "estudiante").mkdir(parents=True)
    for i in range(40):
        Image.new("RGB", (5, 5)).save(P["crops"] / "estudiante" / f"{i:04d}__p0.jpg")
    rs.cmd_dividir(Namespace(raiz=d))
    en_test = sorted(f.name for f in (P["split"] / "test" / "estudiante").glob("*.jpg"))
    P["solo_train"].write_text("# salones reales\nestudiante/" + en_test[0].split("__")[0] + "\n")
    rs.cmd_dividir(Namespace(raiz=d))
    assert (P["split"] / "train" / "estudiante" / en_test[0]).exists()
    assert not (P["split"] / "test" / "estudiante" / en_test[0]).exists()

print("OK")
