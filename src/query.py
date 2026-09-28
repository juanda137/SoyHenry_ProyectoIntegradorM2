"""
Pipeline de consulta: pregunta -> búsqueda -> contexto -> LLM -> JSON

    python -m src.query "¿Cómo solicito vacaciones?"
    python -m src.query "..." --metodo knn        (sólo semántica, para comparar)
    python -m src.query "..." --evaluar           (suma el agente evaluador)

ESTO USA RAG (Retrieval-Augmented Generation)
  En vez de confiar en lo que el modelo "recuerda", primero RECUPERAMOS los
  fragmentos relevantes de nuestra documentación y después le pedimos que
  GENERE la respuesta usando sólo esos fragmentos. Dos pasos, en ese orden:
    1. recuperar -> embeber_pregunta() + busqueda.buscar_chunks()
    2. generar   -> generar_respuesta()
  Así la documentación se actualiza re-indexando (sin reentrenar el modelo),
  el prompt lleva sólo los 2-5 fragmentos útiles (no el documento entero) y
  cada respuesta dice de qué fragmentos salió.

SALIDA
  stdout: sólo el JSON, con exactamente user_question, system_answer y
  chunks_related, para que otro sistema lo lea sin limpiarlo.
  stderr: el diagnóstico (similitudes, avisos y la evaluación).
"""

import argparse
import json
import os
import sys

from src import busqueda, config, evaluador, llm_client


CLAVES_DE_SALIDA = ["user_question", "system_answer", "chunks_related"]
CLAVES_DE_CHUNK = ["id", "seccion", "similitud", "texto"]


class ErrorDeIndice(Exception):
    """El índice no existe o no es compatible con el modelo configurado."""


def validar_pregunta(pregunta):
    """Limpia la pregunta; falla con un mensaje claro si está vacía o es muy larga."""
    if pregunta is None or pregunta.strip() == "":
        raise ValueError("La pregunta está vacía.")
    pregunta = pregunta.strip()
    if len(pregunta) > config.LARGO_MAXIMO_PREGUNTA:
        raise ValueError("La pregunta supera los " + str(config.LARGO_MAXIMO_PREGUNTA) + " caracteres.")
    return pregunta


def cargar_indice(ruta, cliente):
    """Lee el índice y verifica que se haya hecho con el mismo modelo de embeddings."""
    if not os.path.exists(ruta):
        raise ErrorDeIndice("No existe el índice. Primero corré: python -m src.build_index")
    with open(ruta, encoding="utf-8") as archivo:
        indice = json.load(archivo)
    if indice["modelo_embedding"] != cliente["modelo_embedding"]:
        raise ErrorDeIndice("El índice se hizo con " + indice["modelo_embedding"] + " y el modelo "
                            "configurado es " + cliente["modelo_embedding"] + ". Los embeddings de "
                            "modelos distintos no se comparan: corré python -m src.build_index")
    return indice


def embeber_pregunta(cliente, pregunta, dimension):
    """Etapa 1: la pregunta como vector, con la misma dimensión que los chunks."""
    vector = llm_client.embeber_textos(cliente, [pregunta])[0]
    if len(vector) != dimension:
        raise ErrorDeIndice("La pregunta tiene " + str(len(vector)) + " dimensiones y el índice " +
                            str(dimension) + ". Volvé a correr python -m src.build_index")
    return vector


def generar_respuesta(cliente, pregunta, chunks_relacionados):
    """Etapas 3 y 4: arma el contexto en el prompt y el LLM responde con él."""
    contexto = busqueda.formatear_chunks(chunks_relacionados)
    sistema = config.leer_prompt("respuesta.txt").replace("{CONTEXTO}", contexto)
    mensajes = [{"role": "system", "content": sistema},
                {"role": "user", "content": pregunta}]
    return llm_client.preguntar(cliente, mensajes, cliente["modelo_chat"],
                                config.TEMPERATURA_RESPUESTA)


def validar_salida(salida):
    """Verifica el contrato: exactamente las 3 claves, con los tipos correctos."""
    if sorted(salida.keys()) != sorted(CLAVES_DE_SALIDA):
        raise ValueError("La salida debe tener exactamente: " + ", ".join(CLAVES_DE_SALIDA))
    if not isinstance(salida["user_question"], str) or not isinstance(salida["system_answer"], str):
        raise ValueError("user_question y system_answer deben ser texto.")
    if not isinstance(salida["chunks_related"], list):
        raise ValueError("chunks_related debe ser una lista.")
    for chunk in salida["chunks_related"]:
        if sorted(chunk.keys()) != CLAVES_DE_CHUNK:
            raise ValueError("Cada chunk debe tener: " + ", ".join(CLAVES_DE_CHUNK))
    return salida


def responder(cliente, busqueda_preparada, dimension, pregunta, metodo="hibrida"):
    """
    RAG completo para una pregunta. Devuelve (salida, error).
    Si el proveedor falla, la salida IGUAL respeta el contrato y "error" dice qué pasó.
    """
    chunks_relacionados = []
    error = None
    try:
        vector = embeber_pregunta(cliente, pregunta, dimension)                            # 1
        chunks_relacionados = busqueda.buscar_chunks(busqueda_preparada, vector,
                                                     pregunta, metodo)                     # 2
        respuesta = generar_respuesta(cliente, pregunta, chunks_relacionados)              # 3 y 4
    except llm_client.ErrorDelModelo as problema:
        error = str(problema)
        respuesta = ("No pude generar la respuesta en este momento (" + error + "). "
                     "Los fragmentos relacionados que se encontraron van en chunks_related.")
    salida = {"user_question": pregunta, "system_answer": respuesta,
              "chunks_related": chunks_relacionados}
    return validar_salida(salida), error


def crear_parser():
    """Argumentos de línea de comandos (también los usa main.py)."""
    parser = argparse.ArgumentParser(description="Bot de FAQs para HR: responde con RAG y devuelve JSON.")
    parser.add_argument("pregunta", nargs="?", default=None,
                        help="La pregunta. Si no la escribís, se te pide.")
    parser.add_argument("--provider", choices=sorted(config.PROVEEDORES), default=None,
                        help="Proveedor. Por defecto, LLM_PROVIDER del .env")
    parser.add_argument("--model", default=None, help="Modelo de chat. Por defecto, CHAT_MODEL del .env")
    parser.add_argument("--metodo", choices=["hibrida", "knn"], default="hibrida",
                        help="Método de búsqueda (por defecto, hibrida)")
    parser.add_argument("--evaluar", action="store_true",
                        help="Evalúa la respuesta; el resultado sale por stderr")
    return parser


def mostrar_diagnostico(salida, metodo):
    """Por stderr: qué chunks se recuperaron y con qué similitud."""
    print("Búsqueda " + metodo + ": " + str(len(salida["chunks_related"])) + " chunks", file=sys.stderr)
    for chunk in salida["chunks_related"]:
        print("  chunk " + str(chunk["id"]) + " | similitud " + str(chunk["similitud"]) +
              " | " + chunk["seccion"], file=sys.stderr)


def ejecutar_consulta(args):
    """Corre una consulta desde la línea de comandos. Devuelve el código de salida."""
    try:
        pregunta = args.pregunta
        if pregunta is None:
            pregunta = input("Tu pregunta: ")
        pregunta = validar_pregunta(pregunta)
        cliente = llm_client.crear_cliente(args.provider, args.model)
        indice = cargar_indice(config.RUTA_INDICE, cliente)
        preparada = busqueda.preparar_busqueda(indice["chunks"])
        salida, error = responder(cliente, preparada, indice["dimension"], pregunta, args.metodo)
    except (ValueError, config.ErrorDeConfiguracion, ErrorDeIndice) as problema:
        print("Error: " + str(problema), file=sys.stderr)
        return 2
    print(json.dumps(salida, ensure_ascii=False, indent=2))
    mostrar_diagnostico(salida, args.metodo)
    if error is not None:
        return 1
    if args.evaluar:
        evaluacion = evaluador.evaluar_respuesta(cliente, salida)
        print("Evaluación:\n" + json.dumps(evaluacion, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


def main():
    """Punto de entrada de python -m src.query"""
    # Forzamos UTF-8: en Windows, al redirigir la salida a un archivo, Python
    # usaría la codificación del sistema y las tildes saldrían rotas.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(ejecutar_consulta(crear_parser().parse_args()))


if __name__ == "__main__":
    main()
