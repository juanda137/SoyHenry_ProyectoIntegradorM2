"""
Carga del documento y fragmentación (chunking).

ESTRATEGIA: CHUNKING POR ESTRUCTURA
  El documento es una FAQ: cada pregunta con su respuesta ya es una unidad de
  significado completa. Por eso el corte natural es el par P/R, y no un tamaño
  fijo que podría dejar la pregunta en un chunk y la respuesta en otro.

  Dos ajustes para respetar los límites de tamaño (ver config.py):
    * Un bloque con menos de CHUNK_MIN_TOKENS se une a su vecino de la misma
      sección: un fragmento muy corto no tiene contexto suficiente.
    * Un bloque con más de CHUNK_MAX_TOKENS se parte por oraciones, repitiendo
      SOLAPAMIENTO_ORACIONES oraciones entre pedazos para no perder el hilo.

TOKENS
  Se cuentan con tiktoken (cl100k_base), que corre local y gratis. Es el
  tokenizador de los modelos de OpenAI; con Gemini el número exacto varía un
  poco, pero sirve como medida estándar para el rango de 50 a 500.
"""

import os
import re
import sys

import tiktoken

from src import config


PREFIJO_SECCION = "# Sección:"
PREFIJO_PREGUNTA = "P:"
SECCION_INICIAL = "General"   # para el texto que está antes de la primera sección

# Fin de oración: ".", "?" o "!" precedido por una letra o un paréntesis, y
# seguido de espacio. Pedir la letra antes del punto evita cortar en "1. Entrá"
# (los pasos numerados). re.IGNORECASE hace que valga también para mayúsculas.
FIN_DE_ORACION = re.compile(r"(?<=[a-záéíóúñü)][.!?])\s+", re.IGNORECASE)


def cargar_documento(ruta):
    """Lee el documento en UTF-8; si falla, reintenta con latin-1 y avisa."""
    if not os.path.exists(ruta):
        raise FileNotFoundError("No encuentro el documento: " + ruta)
    try:
        # "utf-8-sig" es UTF-8 que además ignora la marca (BOM) que algunos
        # editores de Windows agregan al principio del archivo.
        with open(ruta, encoding="utf-8-sig") as archivo:
            return archivo.read()
    except UnicodeDecodeError:
        print("  [aviso] El documento no está en UTF-8; lo leo como latin-1.", file=sys.stderr)
        with open(ruta, encoding="latin-1") as archivo:
            return archivo.read()


def contar_tokens(texto):
    """Cantidad de tokens de un texto según el tokenizador cl100k_base."""
    # tiktoken guarda el tokenizador en memoria: pedirlo muchas veces es gratis.
    codificador = tiktoken.get_encoding("cl100k_base")
    return len(codificador.encode(texto))


def guardar_bloque(bloques, seccion, lineas):
    """Agrega a "bloques" el texto de esas líneas, si no está vacío."""
    texto = "\n".join(lineas).strip()
    if texto != "":
        bloques.append({"seccion": seccion, "texto": texto})


def separar_en_bloques(texto):
    """Parte el documento en bloques: cada par P/R, o el texto suelto de una sección."""
    bloques = []
    seccion = SECCION_INICIAL
    lineas = []
    for linea in texto.splitlines():
        if linea.startswith(PREFIJO_SECCION):
            guardar_bloque(bloques, seccion, lineas)
            seccion = linea[len(PREFIJO_SECCION):].strip()
            lineas = []
        elif linea.startswith(PREFIJO_PREGUNTA):
            guardar_bloque(bloques, seccion, lineas)
            lineas = [linea]
        else:
            lineas.append(linea)
    guardar_bloque(bloques, seccion, lineas)
    return bloques


def conviene_unir(anterior, bloque):
    """True si dos bloques vecinos van juntos: misma sección, alguno chico, y entran."""
    if anterior["seccion"] != bloque["seccion"]:
        return False
    alguno_chico = (contar_tokens(anterior["texto"]) < config.CHUNK_MIN_TOKENS or
                    contar_tokens(bloque["texto"]) < config.CHUNK_MIN_TOKENS)
    juntos = anterior["texto"] + "\n\n" + bloque["texto"]
    return alguno_chico and contar_tokens(juntos) <= config.CHUNK_MAX_TOKENS


def unir_bloques_chicos(bloques):
    """Une cada bloque demasiado corto con su vecino de la misma sección."""
    resultado = []
    for bloque in bloques:
        if len(resultado) > 0 and conviene_unir(resultado[-1], bloque):
            anterior = resultado[-1]
            resultado[-1] = {"seccion": anterior["seccion"],
                             "texto": anterior["texto"] + "\n\n" + bloque["texto"]}
        else:
            resultado.append(bloque)
    return resultado


def separar_en_oraciones(texto):
    """Parte un texto en oraciones: primero por líneas, después por fin de oración."""
    oraciones = []
    for linea in texto.splitlines():
        for oracion in FIN_DE_ORACION.split(linea.strip()):
            if oracion != "":
                oraciones.append(oracion)
    return oraciones


def agrupar_oraciones(oraciones):
    """Junta oraciones en pedazos de hasta CHUNK_MAX_TOKENS, con solapamiento."""
    pedazos = []
    actual = []
    for oracion in oraciones:
        if len(actual) > 0 and contar_tokens(" ".join(actual + [oracion])) > config.CHUNK_MAX_TOKENS:
            pedazos.append(" ".join(actual))
            # El pedazo nuevo arranca repitiendo las últimas oraciones del anterior.
            actual = actual[len(actual) - config.SOLAPAMIENTO_ORACIONES:]
        actual.append(oracion)
    if len(actual) > 0:
        pedazos.append(" ".join(actual))
    return pedazos


def partir_bloque_grande(texto):
    """Devuelve el texto tal cual si entra en el límite; si no, lo parte por oraciones."""
    if contar_tokens(texto) <= config.CHUNK_MAX_TOKENS:
        return [texto]
    return agrupar_oraciones(separar_en_oraciones(texto))


def partir_en_chunks(texto):
    """Documento completo -> lista de chunks {id, seccion, texto, tokens}."""
    chunks = []
    for bloque in unir_bloques_chicos(separar_en_bloques(texto)):
        for pedazo in partir_bloque_grande(bloque["texto"]):
            chunks.append({
                "id": len(chunks) + 1,
                "seccion": bloque["seccion"],
                "texto": pedazo,
                "tokens": contar_tokens(pedazo),
            })
    return chunks


def texto_para_embeber(chunk):
    """Lo que se embebe: la sección adelante le da contexto al vector."""
    return "Sección: " + chunk["seccion"] + ". " + chunk["texto"]


def sin_espacios(texto):
    """Quita espacios y saltos de línea, para comparar sólo el contenido."""
    return "".join(texto.split())


def oraciones_perdidas(texto_documento, chunks):
    """Oraciones del documento que no quedaron en ningún chunk (lo ideal: ninguna)."""
    capturado = ""
    for chunk in chunks:
        # Los encabezados de sección no quedan en el texto del chunk sino en su
        # metadata; los sumamos para que también cuenten como capturados.
        capturado = capturado + PREFIJO_SECCION + chunk["seccion"] + chunk["texto"]
    capturado = sin_espacios(capturado)
    perdidas = []
    for oracion in separar_en_oraciones(texto_documento):
        if sin_espacios(oracion) not in capturado:
            perdidas.append(oracion)
    return perdidas
