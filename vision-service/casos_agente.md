# Casos de prueba del agente

Modelo: `nvidia/nemotron-3.5-super-vl-120b-a12b`

Conteos registrados antes de preguntar:

- **38-101**: {"total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "22:19"}
- **38-102**: {"total": 5, "por_rol": {"Estudiante": 4, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "22:19"}
- **38-201**: sin foto

## Caso 1 (Habitual)

**Pregunta:** ¿Cuántos estudiantes hay en el 38-101?

**Esperado:** Llama a contar_personas('38-101') y responde con el número de estudiantes del conteo.

**Herramientas llamadas:**

1. `contar_personas({"salon": "38-101"})` → `{"salon": "38-101", "total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "22:19"}`

**Respuesta:** Hay 15 estudiantes en el salón 38-101.

**Valoración:** ✅ Una sola herramienta, la adecuada; la respuesta coincide con el conteo (15 estudiantes).

## Caso 2 (Habitual)

**Pregunta:** ¿Hay personal de aseo en el aula 38-102?

**Esperado:** Llama a contar_personas('38-102') y responde que sí, con cuántas personas de aseo.

**Herramientas llamadas:**

1. `contar_personas({"salon": "38-102"})` → `{"salon": "38-102", "total": 5, "por_rol": {"Estudiante": 4, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "22:19"}`

**Respuesta:** Sí, hay una persona con rol de personal de aseo en el aula 38-102.

**Valoración:** ✅ Correcta, con una sola herramienta. Interpretó "aula 38-102" como el id del salón.

## Caso 3 (Habitual)

**Pregunta:** ¿Qué porcentaje de ocupación tiene el 38-101?

**Esperado:** Llama a calcular_ocupacion('38-101'), que toma el conteo y la capacidad, y responde 40 %.

**Herramientas llamadas:**

1. `calcular_ocupacion({"salon": "38-101"})` → `{"salon": "38-101", "personas": 16, "capacidad": 40, "ocupacion_pct": 40.0, "supera_capacidad": false, "cupos_libres": 24}`

**Respuesta:** El salón 38-101 tiene una ocupación del 40 por ciento. Hay 16 personas de una capacidad de 40, con 24 cupos libres y no supera su capacidad.

**Valoración:** ✅ Una sola llamada: la herramienta tomó el conteo y la capacidad, y el modelo no hizo la cuenta ni pasó números.

## Caso 4 (Varias herramientas)

**Pregunta:** De los salones con foto analizada, ¿cuál está más lleno respecto a su capacidad?

**Esperado:** Calcula la ocupación de cada salón con foto (calcular_ocupacion) y los compara: el 38-101.

**Herramientas llamadas:**

1. `listar_salones({})` → `{"salones": [{"id": "38-101", "nombre": "Aula 38-101", "capacidad": 40, "tiene_foto_analizada": true}, {"id": "38-102", "nombre": "Aula 38-102", "capacidad": 35, "tiene_foto_analizada": true}, {"id": "38-201", "nombre": "Aula 38-201", "capacidad": 45, "tiene_foto_analizada": false}]}`
2. `calcular_ocupacion({"salon": "38-101"})` → `{"salon": "38-101", "personas": 16, "capacidad": 40, "ocupacion_pct": 40.0, "supera_capacidad": false, "cupos_libres": 24}`
3. `calcular_ocupacion({"salon": "38-102"})` → `{"salon": "38-102", "personas": 5, "capacidad": 35, "ocupacion_pct": 14.3, "supera_capacidad": false, "cupos_libres": 30}`

**Respuesta:** El salón más lleno respecto a su capacidad es el 38-101, con un 40 por ciento de ocupación. El salón 38-102 tiene un 14.3 por ciento.

**Valoración:** ✅ 3 llamadas (antes eran 5). Usó listar_salones para saber qué salones tienen foto, comparó bien e ignoró el 38-201.

## Caso 5 (Límite)

**Pregunta:** ¿Cuántos guardias de seguridad hay en el 38-201?

**Esperado:** El 38-201 no tiene foto: debe decir que no hay datos y que hay que subir una foto, sin inventar un número.

**Herramientas llamadas:**

1. `contar_personas({"salon": "38-201"})` → `{"error": "El salón 38-201 no tiene ninguna foto analizada. Hay que subir y procesar una foto en la página."}`

**Respuesta:** El salón 38-201 no tiene ninguna foto analizada todavía, por lo que no puedo decirte cuántos guardias de seguridad hay. Falta subir y procesar una foto en la página para obtener esa información.

**Valoración:** ✅ No inventó un número y explicó que falta subir y procesar la foto.

