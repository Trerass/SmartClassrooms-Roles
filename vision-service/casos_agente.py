"""Corre los cinco casos de prueba del agente y guarda lo ocurrido en casos_agente_salida.md.

Necesita la clave del modelo (NVIDIA_API_KEY en el entorno o en .env). Ejecutar: python casos_agente.py
casos_agente.md (con las valoraciones escritas a mano) no se toca: se compara a mano con la salida.

Escenario fijo: 38-101 = fotos_prueba/salon_aseo.webp, 38-102 = foto_salon.jpg, 38-201 sin foto.
"""
import json
from pathlib import Path

import agente
import roles_salon as rs

AQUI = Path(__file__).parent
FOTOS = {"38-101": AQUI / "fotos_prueba" / "salon_aseo.webp", "38-102": AQUI / "foto_salon.jpg"}

CASOS = [
    ("Habitual", "¿Cuántos estudiantes hay en el 38-101?",
     "Llama a contar_personas('38-101') y responde con el número de estudiantes del conteo."),
    ("Habitual", "¿Hay personal de aseo en el aula 38-102?",
     "Llama a contar_personas('38-102') y responde que sí, con cuántas personas de aseo."),
    ("Habitual", "¿Qué porcentaje de ocupación tiene el 38-101?",
     "Llama a calcular_ocupacion('38-101'), que toma el conteo y la capacidad, y responde 40 %."),
    ("Varias herramientas", "De los salones con foto analizada, ¿cuál está más lleno respecto a su capacidad?",
     "Calcula la ocupación de cada salón con foto (calcular_ocupacion) y los compara: el 38-101."),
    ("Límite", "¿Cuántos guardias de seguridad hay en el 38-201?",
     "El 38-201 no tiene foto: debe decir que no hay datos y que hay que subir una foto, sin inventar un número."),
]


def main():
    yolo = rs.get_yolo("yolov8m.pt")  # antes que TensorFlow (ver get_yolo)
    rs.configurar_tf()
    import keras
    P = rs.rutas(AQUI / "data_roles")
    model = keras.saving.load_model(P["modelo"])
    class_names = json.loads(P["clases"].read_text(encoding="utf-8"))
    for salon, foto in FOTOS.items():
        _, dets, conteo = rs.analizar_salon(foto, yolo, model, class_names)
        agente.registrar_analisis(salon, len(dets), {rs.ROLE_LABELS[r]: n for r, n in conteo.items()})

    cliente = agente.crear_cliente()
    lineas = [f"# Casos de prueba del agente\n\nModelo: `{agente.modelo()}`\n",
              "Conteos registrados antes de preguntar:\n"]
    lineas += [f"- **{s}**: {json.dumps(agente.ANALISIS[s], ensure_ascii=False)}" for s in FOTOS]
    lineas.append("- **38-201**: sin foto\n")
    for i, (tipo, pregunta, esperado) in enumerate(CASOS, 1):
        r = agente.preguntar(pregunta, cliente)
        lineas += [f"## Caso {i} ({tipo})\n", f"**Pregunta:** {pregunta}\n", f"**Esperado:** {esperado}\n",
                   "**Herramientas llamadas:**\n"]
        lineas += [f"{j}. `{p['herramienta']}({json.dumps(p['argumentos'], ensure_ascii=False)})` → "
                   f"`{json.dumps(p['resultado'], ensure_ascii=False)}`" for j, p in enumerate(r["pasos"], 1)] or ["(ninguna)"]
        lineas += ["", f"**Respuesta:** {r['respuesta']}\n", "**Valoración:** _(completar a mano)_\n"]
        print(f"Caso {i}: {len(r['pasos'])} herramientas → {r['respuesta'][:100]}")
    (AQUI / "casos_agente_salida.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado en casos_agente_salida.md")


if __name__ == "__main__":
    main()
