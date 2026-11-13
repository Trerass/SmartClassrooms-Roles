"""Pruebas del agente con un LLM simulado (sin clave ni red). Ejecutar: python test_agente.py"""
import json
from types import SimpleNamespace as NS

import agente


def llamada(nombre, **args):
    return NS(id=f"call_{nombre}", function=NS(name=nombre, arguments=json.dumps(args)))


class LLMFalso:
    """Devuelve las respuestas en orden y guarda lo que recibió en cada turno."""

    def __init__(self, *turnos):
        self.turnos, self.recibido = list(turnos), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.recibido.append(kw)
        t = self.turnos.pop(0)
        msg = NS(content=t, tool_calls=None) if isinstance(t, str) else NS(content="", tool_calls=t)
        return NS(choices=[NS(message=msg)])


agente.ANALISIS.clear()
agente.registrar_analisis("38-101", 17, {"Estudiante": 13, "Aseo": 1, "Seguridad": 1, "Indeterminado": 2})

agente.registrar_analisis("38-102", 3, {"Estudiante": 3})
agente.borrar_analisis("Aula 38-102")
assert "38-102" not in agente.ANALISIS
agente.registrar_analisis("salón inventado", 9, {})
assert "salón inventado" not in agente.ANALISIS and len(agente.ANALISIS) == 1

# Herramientas por separado.
assert agente.contar_personas("Aula 38-101")["total"] == 17  # acepta el nombre además del id
assert "no existe" in agente.contar_personas("38-999")["error"]
assert "no tiene ninguna foto" in agente.contar_personas("38-102")["error"]
assert agente.calcular_ocupacion(50, 40) == {"personas": 50, "capacidad": 40, "ocupacion_pct": 125.0,
                                             "supera_capacidad": True, "cupos_libres": 0}
assert "error" in agente.calcular_ocupacion(5, 0)
assert [s["tiene_foto_analizada"] for s in agente.listar_salones()["salones"]] == [True, False, False]
assert "no existe" in agente.ejecutar_herramienta("borrar_todo", "{}")["error"]
assert "inválidos" in agente.ejecutar_herramienta("contar_personas", "{no es json")["error"]

# Ciclo: encadena herramientas y el resultado de cada una vuelve al modelo.
llm = LLMFalso([llamada("contar_personas", salon="38-101")],
               [llamada("calcular_ocupacion", personas=17, capacidad=40)],
               "El 38-101 tiene 17 personas, un 42,5 % de su capacidad.")
r = agente.preguntar("¿Qué tan lleno está el 38-101?", llm)
assert [p["herramienta"] for p in r["pasos"]] == ["contar_personas", "calcular_ocupacion"]
assert r["pasos"][1]["resultado"]["ocupacion_pct"] == 42.5
assert r["respuesta"].startswith("El 38-101") and not r["limite_alcanzado"]
ultimo = llm.recibido[-1]["messages"][-1]
assert ultimo["role"] == "tool" and json.loads(ultimo["content"])["ocupacion_pct"] == 42.5

# Sin herramientas cuando no hacen falta.
r = agente.preguntar("¿Qué hora es en Tokio?", LLMFalso("Solo puedo responder sobre la ocupación de los salones."))
assert r["pasos"] == [] and "salones" in r["respuesta"]

# Límite de llamadas: después de N herramientas se le quitan y debe responder.
bucle = [[llamada("listar_salones")] for _ in range(3)] + ["No pude terminar."]
llm = LLMFalso(*bucle)
r = agente.preguntar("Pregunta que lo hace repetir", llm, max_llamadas=3)
assert len(r["pasos"]) == 3 and r["limite_alcanzado"] and "límite de 3" in r["respuesta"]
assert "tools" not in llm.recibido[-1] and "tools" in llm.recibido[0]

# Si pide varias herramientas a la vez y se pasa del límite, las de más no se ejecutan.
llm = LLMFalso([llamada("listar_salones"), llamada("contar_personas", salon="38-101")], "Listo.")
r = agente.preguntar("x", llm, max_llamadas=1)
assert len(r["pasos"]) == 1 and r["limite_alcanzado"]
assert "Límite" in json.loads(llm.recibido[-1]["messages"][-1]["content"])["error"]

print("OK")
