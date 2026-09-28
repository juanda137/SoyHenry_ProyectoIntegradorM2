"""
Genera los archivos de outputs/ corriendo el sistema de punta a punta:

    python -m src.generar_ejemplos

  sample_queries.json         las respuestas, con el contrato de 3 claves
  sample_evaluations.json     la evaluación de cada respuesta
  metricas_recuperacion.json  knn contra híbrida, sobre las mismas preguntas

LA MÉTRICA DE RECUPERACIÓN
  Es la de la rúbrica: qué porcentaje de los chunks recuperados contiene al
  menos una palabra clave de la pregunta. La pregunta fuera de alcance se
  deja afuera del promedio: ahí lo correcto es NO encontrar nada.
  Medir los dos métodos con el mismo vector de pregunta hace la comparación
  justa: lo único que cambia es la búsqueda.
"""

import json
import os
import sys
import time

from src import busqueda, config, evaluador, llm_client, query


PREGUNTAS_DE_EJEMPLO = [
    {"tipo": "si_no", "pregunta": "¿Puedo cancelar una solicitud de vacaciones que ya fue aprobada?"},
    {"tipo": "como_hacer", "pregunta": "¿Cómo configuro el inicio de sesión único (SSO) con Microsoft Entra ID?"},
    {"tipo": "dato_exacto", "pregunta": "¿Cuántos días de vacaciones pendientes se pueden trasladar al año siguiente?"},
    {"tipo": "dos_partes", "pregunta": "¿En qué formatos puedo exportar los recibos de nómina y cuánto tiempo se conservan?"},
    {"tipo": "fuera_de_alcance", "pregunta": "¿Cuál es la receta de la paella valenciana?"},
]
METODOS = ["knn", "hibrida"]
PAUSA_ENTRE_PREGUNTAS = 4   # segundos: la capa gratuita limita pedidos por minuto


def porcentaje_con_palabras_clave(pregunta, chunks):
    """Porcentaje de chunks que contienen al menos una palabra clave de la pregunta."""
    if len(chunks) == 0:
        return 0.0
    claves = set(busqueda.palabras_clave(pregunta))
    con_clave = 0
    for chunk in chunks:
        # "&" entre dos conjuntos da las palabras que están en los dos.
        if len(claves & set(busqueda.separar_en_palabras(chunk["texto"]))) > 0:
            con_clave = con_clave + 1
    return round(100 * con_clave / len(chunks), 1)


def medir_recuperacion(busqueda_preparada, vector, pregunta):
    """Corre knn e híbrida con el mismo vector y mide cada una."""
    medicion = {}
    for metodo in METODOS:
        chunks = busqueda.buscar_chunks(busqueda_preparada, vector, pregunta, metodo)
        ids = []
        similitudes = []
        for chunk in chunks:
            ids.append(chunk["id"])
            similitudes.append(chunk["similitud"])
        medicion[metodo] = {"chunks_devueltos": len(chunks), "ids": ids, "similitudes": similitudes,
                            "porcentaje_con_palabras_clave": porcentaje_con_palabras_clave(pregunta, chunks)}
    return medicion


def resumir_metricas(mediciones):
    """Promedio del porcentaje con palabras clave por método (sin la fuera de alcance)."""
    resumen = {}
    for metodo in METODOS:
        porcentajes = []
        for medicion in mediciones:
            if medicion["tipo"] != "fuera_de_alcance":
                porcentajes.append(medicion[metodo]["porcentaje_con_palabras_clave"])
        resumen[metodo] = round(sum(porcentajes) / len(porcentajes), 1)
    return resumen


def procesar_pregunta(cliente, busqueda_preparada, dimension, ejemplo):
    """Una pregunta de ejemplo: respuesta, evaluación y medición de la búsqueda."""
    pregunta = ejemplo["pregunta"]
    salida, error = query.responder(cliente, busqueda_preparada, dimension, pregunta)
    if error is not None:
        print("  [aviso] el modelo falló: " + error, file=sys.stderr)
    evaluacion = evaluador.evaluar_respuesta(cliente, salida)
    evaluacion_con_pregunta = {"tipo": ejemplo["tipo"], "user_question": pregunta,
                               "score": evaluacion["score"], "reason": evaluacion["reason"],
                               "dimensiones": evaluacion["dimensiones"]}
    vector = query.embeber_pregunta(cliente, pregunta, dimension)
    medicion = medir_recuperacion(busqueda_preparada, vector, pregunta)
    medicion["tipo"] = ejemplo["tipo"]
    medicion["pregunta"] = pregunta
    return salida, evaluacion_con_pregunta, medicion


def guardar_json(nombre, datos):
    """Guarda datos en outputs/<nombre> como JSON legible."""
    os.makedirs(config.CARPETA_OUTPUTS, exist_ok=True)
    ruta = os.path.join(config.CARPETA_OUTPUTS, nombre)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    print("Guardado: " + ruta)


def generar(cliente, indice):
    """Corre todas las preguntas de ejemplo y guarda los tres archivos."""
    preparada = busqueda.preparar_busqueda(indice["chunks"])
    respuestas, evaluaciones, mediciones = [], [], []
    for ejemplo in PREGUNTAS_DE_EJEMPLO:
        print("Pregunta (" + ejemplo["tipo"] + "): " + ejemplo["pregunta"])
        salida, evaluacion, medicion = procesar_pregunta(cliente, preparada, indice["dimension"], ejemplo)
        respuestas.append(salida)
        evaluaciones.append(evaluacion)
        mediciones.append(medicion)
        print("  score: " + str(evaluacion["score"]) + " | chunks: " + str(medicion["hibrida"]["ids"]))
        time.sleep(PAUSA_ENTRE_PREGUNTAS)
    guardar_json("sample_queries.json", respuestas)
    guardar_json("sample_evaluations.json", evaluaciones)
    guardar_json("metricas_recuperacion.json", {
        "similitud_minima": config.SIMILITUD_MINIMA,
        "promedio_porcentaje_con_palabras_clave": resumir_metricas(mediciones),
        "por_pregunta": mediciones})


def main():
    """Punto de entrada de python -m src.generar_ejemplos"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        cliente = llm_client.crear_cliente()
        indice = query.cargar_indice(config.RUTA_INDICE, cliente)
        generar(cliente, indice)
    except (config.ErrorDeConfiguracion, query.ErrorDeIndice, llm_client.ErrorDelModelo) as problema:
        print("Error: " + str(problema), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
