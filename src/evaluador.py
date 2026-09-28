"""
Agente evaluador: un LLM como juez de la respuesta del sistema RAG.

Revisa tres dimensiones, cada una de 0 a 10:
  * relevancia:  ¿los chunks recuperados tienen que ver con la pregunta?
  * fidelidad:   ¿la respuesta sale de los chunks, o inventa? (alucinaciones)
  * completitud: ¿la respuesta cubre toda la pregunta?

EL PUNTAJE FINAL LO CALCULA PYTHON, NO EL MODELO
  Promedio de las tres, con una regla: si la fidelidad es menor a 5, el
  puntaje no puede superarla. Una respuesta que inventa no se compensa con
  ser completa. Y el modelo tampoco se valida solo: Python revisa que cada
  puntaje sea un entero de 0 a 10 y que la justificación sea concreta.

AVISO: por defecto el juez es el mismo modelo que respondió, y un modelo
tiende a ser indulgente consigo mismo. Con EVALUATOR_MODEL se usa otro.
"""

import json

from src import busqueda, config, llm_client


DIMENSIONES = ["relevancia", "fidelidad", "completitud"]
FIDELIDAD_MINIMA = 5
LARGO_MINIMO_RAZON = 50
INTENTOS = 2   # si el juez responde algo inválido, se le pregunta una vez más
# Diagnóstico (Tarea 8): con TEMPERATURA_EVALUADOR (0) el juez "piensa" antes
# de escribir el JSON, y esos tokens de razonamiento a veces se comen casi
# todo max_tokens: el texto queda cortado, por ejemplo
#   '{\n  "relevancia": 5,\n  "fidelidad": 10,\n  "completitud": 10,'
# A temperatura 0 esto es determinístico: reintentar con la misma pregunta
# repite el mismo corte. Por eso el reintento usa otra temperatura.
TEMPERATURA_DE_REINTENTO = 0.4


def armar_mensajes(salida):
    """Arma el pedido al juez con la pregunta, la respuesta y los chunks."""
    prompt = config.leer_prompt("evaluador.txt")
    prompt = prompt.replace("{PREGUNTA}", salida["user_question"])
    prompt = prompt.replace("{RESPUESTA}", salida["system_answer"])
    prompt = prompt.replace("{CHUNKS}", busqueda.formatear_chunks(salida["chunks_related"]))
    return [{"role": "user", "content": prompt}]


def interpretar_evaluacion(texto):
    """Lee el JSON del juez y lo valida. Lanza ValueError si algo no cumple."""
    datos = json.loads(texto)   # si no es JSON lanza JSONDecodeError, que es un ValueError
    if not isinstance(datos, dict):
        raise ValueError("El juez no devolvió un objeto JSON.")
    dimensiones = {}
    for nombre in DIMENSIONES:
        valor = datos.get(nombre)
        # bool es un tipo de int en Python (True == 1): lo descartamos aparte.
        if not isinstance(valor, int) or isinstance(valor, bool) or not 0 <= valor <= 10:
            raise ValueError("La dimensión '" + nombre + "' debe ser un entero de 0 a 10.")
        dimensiones[nombre] = valor
    razon = str(datos.get("reason", "")).strip()
    if len(razon) < LARGO_MINIMO_RAZON:
        raise ValueError("El reason debe tener al menos " + str(LARGO_MINIMO_RAZON) + " caracteres.")
    return {"dimensiones": dimensiones, "reason": razon}


def calcular_puntaje(dimensiones):
    """Promedio redondeado de las dimensiones; una fidelidad baja pone un techo."""
    puntaje = round(sum(dimensiones.values()) / len(dimensiones))
    if dimensiones["fidelidad"] < FIDELIDAD_MINIMA:
        puntaje = min(puntaje, dimensiones["fidelidad"])
    return puntaje


def describir_dimensiones(dimensiones):
    """Las dimensiones en una línea, por ejemplo: relevancia 9, fidelidad 9, completitud 7."""
    partes = []
    for nombre in DIMENSIONES:
        partes.append(nombre + " " + str(dimensiones[nombre]))
    return ", ".join(partes)


def evaluar_respuesta(cliente, salida):
    """Devuelve {"score", "reason", "dimensiones"}. Si no puede evaluar, score es None."""
    ultimo_problema = ""
    for intento in range(INTENTOS):
        temperatura = config.TEMPERATURA_EVALUADOR if intento == 0 else TEMPERATURA_DE_REINTENTO
        try:
            texto = llm_client.preguntar(cliente, armar_mensajes(salida), cliente["modelo_evaluador"],
                                         temperatura, formato_json=True)
            evaluacion = interpretar_evaluacion(texto)
        except (ValueError, llm_client.ErrorDelModelo) as problema:
            ultimo_problema = str(problema)
            continue
        puntaje = calcular_puntaje(evaluacion["dimensiones"])
        razon = ("Puntaje " + str(puntaje) + " (" + describir_dimensiones(evaluacion["dimensiones"]) +
                 "): " + evaluacion["reason"])
        return {"score": puntaje, "reason": razon, "dimensiones": evaluacion["dimensiones"]}
    return {"score": None, "dimensiones": None,
            "reason": "No se pudo evaluar la respuesta después de " + str(INTENTOS) +
                      " intentos. Último problema: " + ultimo_problema}
