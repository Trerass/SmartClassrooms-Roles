"""Agente con herramientas que responde preguntas sobre la ocupación de los salones.

El modelo de lenguaje no ve las fotos: decide qué herramienta llamar y responde con lo que
devuelven. La visión (YOLOv8m + EfficientNetB0) es una de esas herramientas.

    Pregunta ─► LLM ─► ¿herramienta? ─sí─► se ejecuta ─► resultado al LLM ─► (repite)
                                    └no─► respuesta final

Configuración (nunca en el código): variable de entorno NVIDIA_API_KEY o archivo .env
en esta carpeta con NVIDIA_API_KEY=nvapi-...  Opcionales: AGENTE_BASE_URL, AGENTE_MODELO.

Uso por consola (analiza las fotos y luego pregunta):

    python agente.py --foto 38-101=fotos_prueba/salon_aseo.webp "¿Hay personal de aseo en el 38-101?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Mismos salones que la página (frontend/src/app/App.tsx).
SALONES = {
    "38-101": {"nombre": "Aula 38-101", "capacidad": 40},
    "38-102": {"nombre": "Aula 38-102", "capacidad": 35},
    "38-201": {"nombre": "Aula 38-201", "capacidad": 45},
}

# Último análisis de cada salón. Solo conteos: la foto no se guarda.
ANALISIS: dict[str, dict] = {}

MAX_LLAMADAS = 6  # herramientas por pregunta; al llegar aquí el agente se detiene

SISTEMA = """Eres el asistente de ocupación de salones de la Universidad EAFIT.
Respondes preguntas sobre cuántas personas hay en cada salón y qué rol tienen
(estudiante, personal de aseo o personal de seguridad), y sobre su ocupación frente a la capacidad.

Reglas:
- Usa SIEMPRE las herramientas para obtener datos. Nunca inventes conteos ni capacidades.
- Para porcentajes de ocupación usa calcular_ocupacion; no hagas la cuenta tú.
- Si un salón no existe o no tiene foto analizada, dilo y explica qué falta (subir y procesar la foto).
- Las personas "indeterminadas" son personas cuyo rol el modelo no pudo decidir: cuéntalas en el total,
  pero no las asignes a ningún rol.
- Si la pregunta no trata sobre los salones, di que solo puedes responder sobre su ocupación.
- Responde en español, en pocas frases y en texto plano: sin Markdown (nada de asteriscos, viñetas ni encabezados), porque el chat lo muestra tal cual."""


# ──────────────────────────────────────────────────────────────────────────────
# Herramientas
# ──────────────────────────────────────────────────────────────────────────────
def _normalizar(salon: str) -> str:
    return str(salon).lower().replace("aula", "").replace("salón", "").replace("salon", "").strip()


def listar_salones() -> dict:
    """Salones disponibles, su capacidad y si ya tienen una foto analizada."""
    return {"salones": [
        {"id": k, "nombre": v["nombre"], "capacidad": v["capacidad"], "tiene_foto_analizada": k in ANALISIS}
        for k, v in SALONES.items()
    ]}


def contar_personas(salon: str) -> dict:
    """Conteo por rol del último análisis de la foto del salón."""
    s = _normalizar(salon)
    if s not in SALONES:
        return {"error": f"El salón '{salon}' no existe. Salones disponibles: {', '.join(SALONES)}."}
    if s not in ANALISIS:
        return {"error": f"El salón {s} no tiene ninguna foto analizada. Hay que subir y procesar una foto en la página."}
    return {"salon": s, **ANALISIS[s]}


def calcular_ocupacion(salon: str) -> dict:
    """Ocupación del salón con su último conteo y su capacidad. El modelo no pasa números: no puede equivocarlos."""
    conteo = contar_personas(salon)
    if "error" in conteo:
        return conteo
    personas, capacidad = conteo["total"], SALONES[conteo["salon"]]["capacidad"]
    return {"salon": conteo["salon"], "personas": personas, "capacidad": capacidad,
            "ocupacion_pct": round(100 * personas / capacidad, 1),
            "supera_capacidad": personas > capacidad,
            "cupos_libres": max(capacidad - personas, 0)}


HERRAMIENTAS = {f.__name__: f for f in [listar_salones, contar_personas, calcular_ocupacion]}

ESQUEMAS = [
    {"type": "function", "function": {
        "name": "listar_salones",
        "description": "Lista los salones disponibles con su id, nombre, capacidad máxima y si ya tienen "
                       "una foto analizada. Úsala para conocer la capacidad o los ids válidos.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "contar_personas",
        "description": "Devuelve cuántas personas detectó el modelo de visión en la última foto del salón, "
                       "en total y por rol (Estudiante, Aseo, Seguridad, Indeterminado), y la hora del análisis.",
        "parameters": {"type": "object", "properties": {
            "salon": {"type": "string", "description": "Id del salón, por ejemplo '38-101'."}},
            "required": ["salon"]}}},
    {"type": "function", "function": {
        "name": "calcular_ocupacion",
        "description": "Calcula la ocupación de un salón con su última foto analizada y su capacidad: "
                       "personas, capacidad, porcentaje, cupos libres y si se supera la capacidad. "
                       "No hace falta llamar antes a contar_personas ni a listar_salones.",
        "parameters": {"type": "object", "properties": {
            "salon": {"type": "string", "description": "Id del salón, por ejemplo '38-101'."}},
            "required": ["salon"]}}},
]


def registrar_analisis(salon: str, total: int, roles: dict[str, int]) -> None:
    """La API llama a esto cada vez que procesa una foto de un salón. Ignora salones desconocidos."""
    if _normalizar(salon) not in SALONES:
        return
    ANALISIS[_normalizar(salon)] = {"total": total, "por_rol": roles,
                                    "hora_analisis": datetime.now().strftime("%H:%M")}


def borrar_analisis(salon: str) -> None:
    """Al quitar la foto en la página, el agente deja de usar su conteo."""
    ANALISIS.pop(_normalizar(salon), None)


def ejecutar_herramienta(nombre: str, argumentos: str) -> dict:
    """Ejecuta la herramienta pedida. Los errores vuelven al modelo como texto, no detienen el agente."""
    if nombre not in HERRAMIENTAS:
        return {"error": f"La herramienta '{nombre}' no existe."}
    try:
        args = json.loads(argumentos or "{}")
        return HERRAMIENTAS[nombre](**args)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Argumentos inválidos para {nombre}: {e}"}


# ──────────────────────────────────────────────────────────────────────────────
# Ciclo del agente
# ──────────────────────────────────────────────────────────────────────────────
def crear_cliente():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).parent / ".env")
    clave = os.getenv("NVIDIA_API_KEY") or os.getenv("AGENTE_API_KEY")
    if not clave:
        raise RuntimeError("Falta la clave del modelo: define NVIDIA_API_KEY en el entorno o en vision-service/.env.")
    return OpenAI(api_key=clave, base_url=os.getenv("AGENTE_BASE_URL", "https://integrate.api.nvidia.com/v1"))


def modelo() -> str:
    return os.getenv("AGENTE_MODELO", "nvidia/nemotron-3.5-super-vl-120b-a12b")


def preguntar(pregunta: str, cliente=None, max_llamadas: int = MAX_LLAMADAS) -> dict:
    """Devuelve {'respuesta', 'pasos': [{herramienta, argumentos, resultado}], 'limite_alcanzado'}."""
    cliente = cliente or crear_cliente()
    mensajes = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": pregunta}]
    pasos = []

    while True:
        # Al llegar al límite se le quitan las herramientas: debe responder con lo que ya tiene.
        sin_cupo = len(pasos) >= max_llamadas
        r = cliente.chat.completions.create(
            model=modelo(), messages=mensajes, temperature=0,
            **({} if sin_cupo else {"tools": ESQUEMAS, "tool_choice": "auto"}))
        msg = r.choices[0].message
        if sin_cupo or not msg.tool_calls:
            respuesta = (msg.content or "").strip()
            if sin_cupo:
                respuesta = (f"(Se alcanzó el límite de {max_llamadas} consultas a herramientas.) " + respuesta).strip()
            return {"respuesta": respuesta, "pasos": pasos, "limite_alcanzado": sin_cupo}

        mensajes.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": t.id, "type": "function",
                                         "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                        for t in msg.tool_calls]})
        for t in msg.tool_calls:
            if len(pasos) >= max_llamadas:
                resultado = {"error": "Límite de llamadas alcanzado; responde con la información que ya tienes."}
            else:
                resultado = ejecutar_herramienta(t.function.name, t.function.arguments)
                pasos.append({"herramienta": t.function.name, "argumentos": _legibles(t.function.arguments),
                              "resultado": resultado})
            mensajes.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(resultado, ensure_ascii=False)})


def _legibles(argumentos: str):
    """Argumentos como dict para mostrarlos; si el modelo mandó JSON roto, el texto tal cual."""
    try:
        return json.loads(argumentos or "{}")
    except json.JSONDecodeError:
        return argumentos


# ──────────────────────────────────────────────────────────────────────────────
# Consola: analiza fotos de ejemplo y pregunta (para las pruebas del README)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Pregunta al agente de ocupación de salones.")
    ap.add_argument("pregunta")
    ap.add_argument("--foto", action="append", default=[], metavar="SALON=RUTA",
                    help="Analiza esta foto como la del salón antes de preguntar. Se puede repetir.")
    args = ap.parse_args()

    if args.foto:
        import roles_salon as rs
        yolo = rs.get_yolo("yolov8m.pt")  # antes que TensorFlow (ver get_yolo)
        rs.configurar_tf()
        import keras
        P = rs.rutas(Path(__file__).parent / "data_roles")
        model = keras.saving.load_model(P["modelo"])
        class_names = json.loads(P["clases"].read_text(encoding="utf-8"))
        for par in args.foto:
            salon, ruta = par.split("=", 1)
            _, dets, conteo = rs.analizar_salon(ruta, yolo, model, class_names)
            registrar_analisis(salon, len(dets), {rs.ROLE_LABELS[r]: n for r, n in conteo.items()})
            print(f"Foto de {salon}: {len(dets)} personas {dict(conteo)}")

    res = preguntar(args.pregunta)
    print(f"\nPregunta: {args.pregunta}")
    for i, p in enumerate(res["pasos"], 1):
        print(f"  [{i}] {p['herramienta']}({json.dumps(p['argumentos'], ensure_ascii=False)})")
        print(f"      → {json.dumps(p['resultado'], ensure_ascii=False)}")
    print(f"\nRespuesta: {res['respuesta']}")


if __name__ == "__main__":
    sys.exit(main())
