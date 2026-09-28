"""
Configuración del proyecto: credenciales, modelos y parámetros.

LAS CREDENCIALES NUNCA ESTÁN EN EL CÓDIGO
  Se leen de variables de entorno con os.getenv(). En desarrollo las cargamos
  desde un archivo .env que no se sube al repositorio (ver .env.example).

UN SOLO CÓDIGO, DOS PROVEEDORES
  El SDK "openai" es un cliente del protocolo /chat/completions y /embeddings,
  no de la empresa. Google publica ese mismo protocolo para Gemini en una URL
  de compatibilidad. Por eso cambiar de proveedor es cambiar base_url + clave.

PRIORIDAD DE LA CONFIGURACIÓN (en todo el proyecto)
  1. flag de línea de comandos   2. variable de entorno   3. valor por defecto
"""

import os

from dotenv import load_dotenv


# --- Rutas ------------------------------------------------------------------
# __file__ es este archivo (src/config.py); dos dirname() suben a proyecto_2/.
CARPETA_DEL_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DOCUMENTO = os.path.join(CARPETA_DEL_PROYECTO, "data", "faq_document.txt")
RUTA_INDICE = os.path.join(CARPETA_DEL_PROYECTO, "index", "index.json")
CARPETA_PROMPTS = os.path.join(CARPETA_DEL_PROYECTO, "prompts")
CARPETA_OUTPUTS = os.path.join(CARPETA_DEL_PROYECTO, "outputs")

# Lo que trae .env.example. Si la clave sigue valiendo esto, no se completó.
CLAVE_DE_EJEMPLO = "your-key-here"

PROVEEDORES = {
    "openai": {
        "variable_clave": "OPENAI_API_KEY",
        "modelo_chat": "gpt-4o-mini",
        "modelo_embedding": "text-embedding-3-small",
        "base_url": None,   # None = la URL de fábrica del SDK (OpenAI)
        "donde_conseguir_clave": "https://platform.openai.com/api-keys",
    },
    "gemini": {
        "variable_clave": "GEMINI_API_KEY",
        # gemini-3.5-flash fue el más estable en proyecto_1: gemini-3.7-flash
        # devolvía HTTP 503 (sobrecarga) seguido.
        "modelo_chat": "gemini-3.5-flash",
        "modelo_embedding": "gemini-embedding-001",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "donde_conseguir_clave": "https://aistudio.google.com/apikey",
    },
}

# --- Chunking (ver src/chunking.py) -----------------------------------------
CHUNK_MIN_TOKENS = 50        # menos que esto no tiene contexto suficiente
CHUNK_MAX_TOKENS = 500       # más que esto mezcla temas y diluye el embedding
SOLAPAMIENTO_ORACIONES = 1   # oraciones repetidas al partir un bloque grande
MINIMO_DE_CHUNKS = 20        # lo exige la consigna

# --- Búsqueda (ver src/busqueda.py) -----------------------------------------
TOP_K_MIN = 2                # la rúbrica pide devolver entre 2 y 5 chunks
TOP_K_MAX = 5
CANDIDATOS_POR_METODO = 10   # cuántos aporta cada método antes de fusionar
RRF_K = 60                   # amortiguador de RRF; el valor del paper original
SIMILITUD_MINIMA = 0.60      # PROVISIONAL: se calibra midiendo en la Tarea 8

# --- Modelo -----------------------------------------------------------------
TEMPERATURA_RESPUESTA = 0.2  # baja: queremos fidelidad al texto, no creatividad
TEMPERATURA_EVALUADOR = 0    # el juez tiene que ser lo más estable posible
# Generoso a propósito: los modelos de razonamiento gastan tokens "pensando"
# antes de escribir, y este límite cubre pensamiento + respuesta juntos.
MAX_TOKENS = 1200
TIMEOUT_SEGUNDOS = 30
REINTENTOS = 3               # ante errores pasajeros (429, 5xx, red)
ESPERA_BASE_SEGUNDOS = 5     # la espera se duplica en cada reintento: 5, 10
TAMANO_DEL_LOTE = 64         # textos por pedido de embeddings
LARGO_MAXIMO_PREGUNTA = 500  # caracteres


class ErrorDeConfiguracion(Exception):
    """Falta una credencial o el proveedor no existe."""


def elegir(valor_de_cli, variable_de_entorno, valor_por_defecto):
    """Prioridad de siempre: flag de CLI, después variable de entorno, después defecto."""
    if valor_de_cli:
        return valor_de_cli
    desde_entorno = os.getenv(variable_de_entorno, "").strip()
    if desde_entorno != "":
        return desde_entorno
    return valor_por_defecto


def resolver_proveedor(proveedor=None):
    """Nombre del proveedor en minúsculas; falla si no está en PROVEEDORES."""
    proveedor = elegir(proveedor, "LLM_PROVIDER", "gemini").strip().lower()
    if proveedor not in PROVEEDORES:
        raise ErrorDeConfiguracion(
            "Proveedor desconocido: '" + proveedor + "'. Opciones: openai, gemini")
    return proveedor


def leer_clave(proveedor):
    """Lee la clave del proveedor desde el entorno, con un mensaje claro si falta."""
    datos = PROVEEDORES[proveedor]
    clave = os.getenv(datos["variable_clave"], "").strip()
    if clave == "" or clave == CLAVE_DE_EJEMPLO:
        raise ErrorDeConfiguracion(
            "Falta la variable " + datos["variable_clave"] + " para usar " + proveedor + ".\n"
            "  1) Copiá .env.example a .env\n"
            "  2) Pegá tu clave en " + datos["variable_clave"] + "=\n"
            "  3) Conseguís una clave en: " + datos["donde_conseguir_clave"])
    return clave


def cargar_configuracion(proveedor=None, modelo_chat=None):
    """Resuelve proveedor, clave y modelos. Devuelve un diccionario."""
    load_dotenv(os.path.join(CARPETA_DEL_PROYECTO, ".env"))
    proveedor = resolver_proveedor(proveedor)
    datos = PROVEEDORES[proveedor]
    chat = elegir(modelo_chat, "CHAT_MODEL", datos["modelo_chat"])
    return {
        "proveedor": proveedor,
        "clave": leer_clave(proveedor),
        "base_url": datos["base_url"],
        "modelo_chat": chat,
        "modelo_embedding": elegir(None, "EMBEDDING_MODEL", datos["modelo_embedding"]),
        "modelo_evaluador": elegir(None, "EVALUATOR_MODEL", chat),
    }


def leer_prompt(nombre):
    """Lee un prompt de prompts/. Viven fuera del código para iterarlos aparte."""
    with open(os.path.join(CARPETA_PROMPTS, nombre), encoding="utf-8") as archivo:
        return archivo.read()
