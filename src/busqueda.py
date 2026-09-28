"""
Búsqueda de los chunks relevantes: semántica, léxica e híbrida.

POR QUÉ HÍBRIDA
  La búsqueda semántica (embeddings + coseno) entiende sinónimos y paráfrasis,
  pero puede traer chunks "del mismo tema" que no mencionan lo que se preguntó.
  La léxica (BM25) premia las coincidencias exactas ("SSO", "IBAN", "Slack"),
  pero no entiende sinónimos. Juntas se cubren una a la otra.

CÓMO SE COMBINAN: RRF (Reciprocal Rank Fusion)
  El coseno da números como 0.74 y BM25 como 3.2: escalas distintas que no se
  pueden sumar. RRF ignora los puntajes y mira sólo la POSICIÓN de cada chunk
  en cada ranking: cada aparición suma 1 / (RRF_K + posición).

FILTRO POR RANGO Y TOP-K
  De lo fusionado quedan los chunks con coseno >= SIMILITUD_MINIMA (búsqueda
  por rango), entre TOP_K_MIN y TOP_K_MAX. El umbral usa el coseno porque es
  el único puntaje con significado absoluto; RRF sólo sirve para ordenar.

El BM25 está adaptado de ai-engineer-lab/common/busqueda.py.
"""

import math

import numpy

from src import chunking, config


# Palabras tan comunes que no ayudan a distinguir un chunk de otro.
# Ya están sin tildes porque se comparan después de separar_en_palabras().
PALABRAS_VACIAS = {
    "a", "al", "como", "con", "cual", "cuando", "cuanto", "cuantos", "cuantas", "de",
    "del", "donde", "el", "en", "es", "esta", "este", "hay", "hago", "la", "las", "le",
    "lo", "los", "me", "mi", "mis", "no", "o", "para", "pero", "por", "puedo", "que",
    "se", "si", "sin", "su", "sus", "tengo", "un", "una", "y", "ya", "yo",
}


def separar_en_palabras(texto):
    """Minúsculas, sin tildes ni signos: "¿Cómo pido días?" -> ["como", "pido", "dias"]."""
    texto = texto.lower()
    for con_tilde, sin_tilde in [["á", "a"], ["é", "e"], ["í", "i"], ["ó", "o"], ["ú", "u"], ["ü", "u"]]:
        texto = texto.replace(con_tilde, sin_tilde)
    limpio = ""
    for caracter in texto:
        if caracter.isalnum():
            limpio = limpio + caracter
        else:
            limpio = limpio + " "
    return limpio.split()


def palabras_clave(texto):
    """Las palabras de un texto que sirven para buscar (sin las palabras vacías)."""
    claves = []
    for palabra in separar_en_palabras(texto):
        if palabra not in PALABRAS_VACIAS:
            claves.append(palabra)
    return claves


def similitud_coseno(vector_consulta, matriz):
    """
    Coseno entre la consulta y cada fila de la matriz: A·B / (|A| · |B|).
    Devuelve un arreglo de NumPy con un puntaje por chunk (1 = misma dirección).
    """
    consulta = numpy.array(vector_consulta, dtype=float)
    matriz = numpy.array(matriz, dtype=float)
    productos_punto = numpy.dot(matriz, consulta)                                # A·B
    normas = numpy.linalg.norm(matriz, axis=1) * numpy.linalg.norm(consulta)    # |A|·|B|
    return productos_punto / normas


def preparar_bm25(textos):
    """Lo que BM25 necesita saber de la colección, calculado una sola vez."""
    documentos = []
    apariciones = {}   # en cuántos documentos aparece cada palabra
    largo_total = 0
    for texto in textos:
        palabras = palabras_clave(texto)
        documentos.append(palabras)
        largo_total = largo_total + len(palabras)
        for palabra in set(palabras):   # set(): cuenta documentos, no repeticiones
            apariciones[palabra] = apariciones.get(palabra, 0) + 1
    largo_promedio = 1
    if len(documentos) > 0 and largo_total > 0:
        largo_promedio = largo_total / len(documentos)
    return {"documentos": documentos, "apariciones": apariciones,
            "total": len(documentos), "largo_promedio": largo_promedio}


def puntaje_bm25(indice, palabras_consulta, posicion, k1=1.5, b=0.75):
    """Puntaje BM25 de un documento: IDF x saturación x normalización por largo."""
    palabras_documento = indice["documentos"][posicion]
    largo_relativo = len(palabras_documento) / indice["largo_promedio"]
    puntaje = 0.0
    for palabra in palabras_consulta:
        veces = palabras_documento.count(palabra)
        if veces == 0:
            continue
        con_la_palabra = indice["apariciones"][palabra]
        # IDF: una palabra que aparece en pocos documentos distingue más.
        idf = math.log(1 + (indice["total"] - con_la_palabra + 0.5) / (con_la_palabra + 0.5))
        # k1 hace que repetir una palabra rinda cada vez menos; b penaliza
        # a los documentos más largos que el promedio.
        puntaje = puntaje + idf * veces * (k1 + 1) / (veces + k1 * (1 - b + b * largo_relativo))
    return puntaje


def ranking_bm25(indice, consulta):
    """Posiciones de los documentos con puntaje BM25 mayor a 0, de mejor a peor."""
    palabras = palabras_clave(consulta)
    puntuados = []
    for posicion in range(indice["total"]):
        puntaje = puntaje_bm25(indice, palabras, posicion)
        if puntaje > 0:   # 0 = ninguna palabra coincide: sumarlo sería ruido
            puntuados.append([posicion, puntaje])
    puntuados.sort(key=lambda par: par[1], reverse=True)
    posiciones = []
    for par in puntuados:
        posiciones.append(par[0])
    return posiciones


def fusionar_rrf(rankings):
    """Reciprocal Rank Fusion: combina rankings por posición. Devuelve posiciones."""
    puntajes = {}
    for ranking in rankings:
        for lugar in range(len(ranking)):
            posicion = ranking[lugar]
            puntajes[posicion] = puntajes.get(posicion, 0) + 1.0 / (config.RRF_K + lugar + 1)
    ordenados = sorted(puntajes.items(), key=lambda par: par[1], reverse=True)
    posiciones = []
    for par in ordenados:
        posiciones.append(par[0])
    return posiciones


def ranking_semantico(similitudes):
    """Posiciones de todos los chunks, de mayor a menor similitud coseno."""
    posiciones = []
    # argsort ordena de menor a mayor; [::-1] da vuelta el resultado.
    for posicion in numpy.argsort(similitudes)[::-1]:
        posiciones.append(int(posicion))
    return posiciones


def elegir_chunks(ranking, similitudes):
    """Filtro por rango (coseno >= SIMILITUD_MINIMA) y entre TOP_K_MIN y TOP_K_MAX chunks."""
    elegidos = []
    for posicion in ranking:
        if similitudes[posicion] >= config.SIMILITUD_MINIMA and len(elegidos) < config.TOP_K_MAX:
            elegidos.append(posicion)
    # Si casi nada pasó el umbral, completamos con los mejores del ranking:
    # el prompt se encarga de responder "no está en la documentación".
    for posicion in ranking:
        if len(elegidos) >= config.TOP_K_MIN:
            break
        if posicion not in elegidos:
            elegidos.append(posicion)
    return elegidos


def preparar_busqueda(chunks):
    """Del índice guardado a lo que usa la búsqueda: matriz de vectores + BM25."""
    vectores = []
    textos = []
    for chunk in chunks:
        vectores.append(chunk["vector"])
        textos.append(chunking.texto_para_embeber(chunk))
    return {"chunks": chunks, "matriz": numpy.array(vectores, dtype=float),
            "bm25": preparar_bm25(textos)}


def buscar_chunks(busqueda_preparada, vector_pregunta, pregunta, metodo="hibrida"):
    """Devuelve entre TOP_K_MIN y TOP_K_MAX chunks relevantes, con su similitud coseno."""
    similitudes = similitud_coseno(vector_pregunta, busqueda_preparada["matriz"])
    ranking = ranking_semantico(similitudes)
    if metodo == "hibrida":
        semanticos = ranking[:config.CANDIDATOS_POR_METODO]
        lexicos = ranking_bm25(busqueda_preparada["bm25"], pregunta)[:config.CANDIDATOS_POR_METODO]
        ranking = fusionar_rrf([semanticos, lexicos])
    elif metodo != "knn":
        raise ValueError("Método de búsqueda desconocido: " + metodo)
    resultados = []
    for posicion in elegir_chunks(ranking, similitudes):
        chunk = busqueda_preparada["chunks"][posicion]
        resultados.append({"id": chunk["id"], "seccion": chunk["seccion"], "texto": chunk["texto"],
                           "similitud": round(float(similitudes[posicion]), 4)})
    return resultados


def formatear_chunks(chunks):
    """Los chunks como texto para un prompt: "[chunk 7 | Nómina]" + su texto."""
    partes = []
    for chunk in chunks:
        partes.append("[chunk " + str(chunk["id"]) + " | " + chunk["seccion"] + "]\n" + chunk["texto"])
    return "\n\n".join(partes)
