"""
Pipeline de indexación: documento -> chunks -> embeddings -> index/index.json

Se corre una vez, y cada vez que cambia el documento:
    python -m src.build_index

Etapas, una función cada una:
  1. chunking.cargar_documento()   lee el archivo, cuidando la codificación
  2. chunking.partir_en_chunks()   chunking por estructura (pares P/R)
  3. generar_embeddings()          un vector por chunk
  4. guardar_indice()              chunks + vectores + metadata en JSON

Antes de gastar una sola llamada a la API se verifica lo que exige la
consigna (20+ chunks, 50-500 tokens, documento capturado completo). Si algo
falla, el índice NO se escribe: mejor un error claro que un índice defectuoso.
"""

import argparse
import datetime
import json
import os
import sys

from src import chunking, config, llm_client


class ErrorDeIndexacion(Exception):
    """Los chunks o los embeddings no cumplen lo que el índice necesita."""


def verificar_chunks(texto, chunks):
    """Lista de problemas del chunking (vacía si todo está bien)."""
    problemas = []
    if len(chunks) < config.MINIMO_DE_CHUNKS:
        problemas.append("Hay " + str(len(chunks)) + " chunks; se necesitan al menos " +
                         str(config.MINIMO_DE_CHUNKS) + ".")
    for chunk in chunks:
        if not config.CHUNK_MIN_TOKENS <= chunk["tokens"] <= config.CHUNK_MAX_TOKENS:
            problemas.append("El chunk " + str(chunk["id"]) + " tiene " + str(chunk["tokens"]) +
                             " tokens (debe estar entre " + str(config.CHUNK_MIN_TOKENS) +
                             " y " + str(config.CHUNK_MAX_TOKENS) + ").")
    for oracion in chunking.oraciones_perdidas(texto, chunks):
        problemas.append("Texto que no quedó en ningún chunk: " + oracion[:80])
    return problemas


def generar_embeddings(cliente, chunks):
    """Etapa 3: un vector por chunk. Verifica que la cantidad coincida."""
    textos = []
    for chunk in chunks:
        textos.append(chunking.texto_para_embeber(chunk))
    vectores = llm_client.embeber_textos(cliente, textos)
    if len(vectores) != len(chunks):
        raise ErrorDeIndexacion("Llegaron " + str(len(vectores)) + " embeddings para " +
                                str(len(chunks)) + " chunks.")
    return vectores


def armar_indice(cliente, chunks, vectores):
    """Junta chunks, vectores y la metadata necesaria para usarlos después."""
    registros = []
    for posicion in range(len(chunks)):
        registro = dict(chunks[posicion])   # copia, para no tocar el original
        registro["vector"] = vectores[posicion]
        registros.append(registro)
    return {
        "proveedor": cliente["proveedor"],
        # El modelo y la dimensión se guardan porque la pregunta se tiene que
        # embeber con el MISMO modelo: vectores de modelos distintos no se comparan.
        "modelo_embedding": cliente["modelo_embedding"],
        "dimension": len(vectores[0]),
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "cantidad_de_chunks": len(registros),
        "chunks": registros,
    }


def guardar_indice(indice, ruta):
    """Etapa 4: escribe el índice en JSON, creando la carpeta si hace falta."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(indice, archivo, ensure_ascii=False)


def construir_indice(proveedor=None):
    """Las 4 etapas en orden. Devuelve el índice guardado."""
    texto = chunking.cargar_documento(config.RUTA_DOCUMENTO)            # etapa 1
    chunks = chunking.partir_en_chunks(texto)                            # etapa 2
    problemas = verificar_chunks(texto, chunks)
    if len(problemas) > 0:
        raise ErrorDeIndexacion("El chunking no cumple los requisitos:\n  - " +
                                "\n  - ".join(problemas))
    cliente = llm_client.crear_cliente(proveedor)
    vectores = generar_embeddings(cliente, chunks)                       # etapa 3
    indice = armar_indice(cliente, chunks, vectores)
    guardar_indice(indice, config.RUTA_INDICE)                           # etapa 4
    return indice


def describir_indice(indice):
    """Resumen legible del índice: chunks, tokens y embeddings."""
    tokens = []
    for chunk in indice["chunks"]:
        tokens.append(chunk["tokens"])
    return ("Índice guardado en " + config.RUTA_INDICE + "\n" +
            "  chunks: " + str(indice["cantidad_de_chunks"]) + "\n" +
            "  tokens por chunk: mínimo " + str(min(tokens)) + ", máximo " + str(max(tokens)) +
            ", promedio " + str(round(sum(tokens) / len(tokens))) + "\n" +
            "  embeddings: " + str(len(indice["chunks"])) + " vectores de " +
            str(indice["dimension"]) + " dimensiones (" + indice["modelo_embedding"] + ")\n" +
            "  cobertura del documento: 100% (verificada antes de guardar)")


def main():
    """Construye el índice desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Construye el índice de la FAQ.")
    parser.add_argument("--provider", choices=sorted(config.PROVEEDORES), default=None)
    args = parser.parse_args()
    try:
        indice = construir_indice(args.provider)
    except (FileNotFoundError, config.ErrorDeConfiguracion, ErrorDeIndexacion,
            llm_client.ErrorDelModelo) as problema:
        print("Error: " + str(problema), file=sys.stderr)
        sys.exit(2)
    print(describir_indice(indice))


if __name__ == "__main__":
    main()
