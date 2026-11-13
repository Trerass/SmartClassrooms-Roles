# Casos de prueba del agente

Modelo: `nvidia/nemotron-3.5-super-vl-120b-a12b`

Conteos registrados antes de preguntar:

- **38-101**: {"total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}
- **38-102**: {"total": 5, "por_rol": {"Estudiante": 4, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}
- **38-201**: sin foto

## Caso 1 (Habitual)

**Pregunta:** ¿Cuántos estudiantes hay en el 38-101?

**Esperado:** Llama a contar_personas('38-101') y responde con el número de estudiantes del conteo.

**Herramientas llamadas:**

1. `contar_personas({"salon": "38-101"})` → `{"salon": "38-101", "total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}`

**Respuesta:** En el salón 38-101 hay **15 estudiantes**.

**Valoración:** ✅ Una sola herramienta, la adecuada; la respuesta coincide con el conteo.

## Caso 2 (Habitual)

**Pregunta:** ¿Hay personal de aseo en el aula 38-102?

**Esperado:** Llama a contar_personas('38-102') y responde que sí, con cuántas personas de aseo.

**Herramientas llamadas:**

1. `listar_salones({})` → `{"salones": [{"id": "38-101", "nombre": "Aula 38-101", "capacidad": 40, "tiene_foto_analizada": true}, {"id": "38-102", "nombre": "Aula 38-102", "capacidad": 35, "tiene_foto_analizada": true}, {"id": "38-201", "nombre": "Aula 38-201", "capacidad": 45, "tiene_foto_analizada": false}]}`
2. `contar_personas({"salon": "38-102"})` → `{"salon": "38-102", "total": 5, "por_rol": {"Estudiante": 4, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}`

**Respuesta:** Sí, hay 1 persona de personal de aseo en el aula 38-102.

**Valoración:** ✅ Correcta. Llamó a listar_salones de más, sin efecto en la respuesta.

## Caso 3 (Habitual)

**Pregunta:** ¿Qué porcentaje de ocupación tiene el 38-101?

**Esperado:** Obtiene el total (contar_personas) y la capacidad (listar_salones) y llama a calcular_ocupacion.

**Herramientas llamadas:**

1. `listar_salones({})` → `{"salones": [{"id": "38-101", "nombre": "Aula 38-101", "capacidad": 40, "tiene_foto_analizada": true}, {"id": "38-102", "nombre": "Aula 38-102", "capacidad": 35, "tiene_foto_analizada": true}, {"id": "38-201", "nombre": "Aula 38-201", "capacidad": 45, "tiene_foto_analizada": false}]}`
2. `contar_personas({"salon": "38-101"})` → `{"salon": "38-101", "total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}`
3. `calcular_ocupacion({"personas": 16, "capacidad": 40})` → `{"personas": 16, "capacidad": 40, "ocupacion_pct": 40.0, "supera_capacidad": false, "cupos_libres": 24}`

**Respuesta:** El salón 38-101 tiene una ocupación del 40.0% (16 personas de 40 de capacidad). No supera la capacidad y quedan 24 cupos libres.

**Valoración:** ✅ Encadenó las 3 herramientas y no hizo la cuenta por su cuenta.

## Caso 4 (Varias herramientas)

**Pregunta:** De los salones con foto analizada, ¿cuál está más lleno respecto a su capacidad?

**Esperado:** Consulta los salones, cuenta en cada uno con foto, calcula la ocupación de cada uno y los compara.

**Herramientas llamadas:**

1. `listar_salones({})` → `{"salones": [{"id": "38-101", "nombre": "Aula 38-101", "capacidad": 40, "tiene_foto_analizada": true}, {"id": "38-102", "nombre": "Aula 38-102", "capacidad": 35, "tiene_foto_analizada": true}, {"id": "38-201", "nombre": "Aula 38-201", "capacidad": 45, "tiene_foto_analizada": false}]}`
2. `contar_personas({"salon": "38-101"})` → `{"salon": "38-101", "total": 16, "por_rol": {"Estudiante": 15, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}`
3. `contar_personas({"salon": "38-102"})` → `{"salon": "38-102", "total": 5, "por_rol": {"Estudiante": 4, "Aseo": 1, "Seguridad": 0, "Indeterminado": 0}, "hora_analisis": "19:27"}`
4. `calcular_ocupacion({"personas": 16, "capacidad": 40})` → `{"personas": 16, "capacidad": 40, "ocupacion_pct": 40.0, "supera_capacidad": false, "cupos_libres": 24}`
5. `calcular_ocupacion({"personas": 5, "capacidad": 35})` → `{"personas": 5, "capacidad": 35, "ocupacion_pct": 14.3, "supera_capacidad": false, "cupos_libres": 30}`

**Respuesta:** El salón más lleno respecto a su capacidad es el **38-101 (Aula 38-101)**, con un **40.0% de ocupación** (16 personas de 40). El salón 38-102 tiene solo un 14.3% (5 personas de 35).

**Valoración:** ✅ 5 llamadas (límite 6). Comparó bien e ignoró el salón sin foto.

## Caso 5 (Límite)

**Pregunta:** ¿Cuántos guardias de seguridad hay en el 38-201?

**Esperado:** El 38-201 no tiene foto: debe decir que no hay datos y que hay que subir una foto, sin inventar un número.

**Herramientas llamadas:**

1. `listar_salones({})` → `{"salones": [{"id": "38-101", "nombre": "Aula 38-101", "capacidad": 40, "tiene_foto_analizada": true}, {"id": "38-102", "nombre": "Aula 38-102", "capacidad": 35, "tiene_foto_analizada": true}, {"id": "38-201", "nombre": "Aula 38-201", "capacidad": 45, "tiene_foto_analizada": false}]}`

**Respuesta:** El salón 38-201 existe, pero aún no tiene una foto analizada, por lo que no hay datos de cuántas personas hay ni de sus roles. Para poder responder, es necesario subir y procesar una foto de ese salón.

**Valoración:** ✅ No inventó un número y explicó que falta la foto.

