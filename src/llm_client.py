"""
Conexión con el proveedor: embeddings y chat.

Una sola responsabilidad: hablar con la API. Quien usa este archivo no sabe si
del otro lado está OpenAI o Gemini.

REINTENTOS
  Los errores pasajeros (429 por límite de uso, 5xx, cortes de red, timeout)
  se reintentan con una espera que se duplica cada vez. Los demás 4xx NO: un
  pedido mal armado o una clave inválida van a fallar igual, y reintentarlos
  sólo quema cuota y esconde el problema real.
"""

import sys
import time

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    OpenAI, RateLimitError)

from src import config


class ErrorDelModelo(Exception):
    """La llamada al proveedor falló y no tiene sentido seguir intentando."""


def crear_cliente(proveedor=None, modelo_chat=None):
    """Prepara la conexión y devuelve un diccionario con lo necesario para usarla."""
    datos = config.cargar_configuracion(proveedor, modelo_chat)
    # max_retries=0: los reintentos los maneja llamar_con_reintentos(), así se
    # ven en pantalla y no se suman a los reintentos internos del SDK.
    conexion = OpenAI(api_key=datos["clave"], base_url=datos["base_url"],
                      timeout=config.TIMEOUT_SEGUNDOS, max_retries=0)
    return {
        "api": conexion,
        "proveedor": datos["proveedor"],
        "modelo_chat": datos["modelo_chat"],
        "modelo_embedding": datos["modelo_embedding"],
        "modelo_evaluador": datos["modelo_evaluador"],
    }


def llamar_con_reintentos(llamada):
    """
    Ejecuta llamada() y la reintenta ante errores pasajeros.
    "llamada" es una función sin argumentos (la armamos con lambda): así la
    misma lógica de reintentos sirve para embeddings y para chat.
    """
    ultimo_error = ""
    for intento in range(1, config.REINTENTOS + 1):
        try:
            return llamada()
        except (RateLimitError, APITimeoutError, APIConnectionError) as error:
            ultimo_error = type(error).__name__
        except APIStatusError as error:
            if error.status_code < 500:
                raise ErrorDelModelo("Error " + str(error.status_code) + " del proveedor: "
                                     "revisá la clave y el nombre del modelo.")
            ultimo_error = "HTTP " + str(error.status_code)
        if intento < config.REINTENTOS:
            espera = config.ESPERA_BASE_SEGUNDOS * 2 ** (intento - 1)
            print("  [reintento " + str(intento) + "/" + str(config.REINTENTOS) + "] " +
                  ultimo_error + ", espero " + str(espera) + "s", file=sys.stderr)
            time.sleep(espera)
    raise ErrorDelModelo("Se agotaron los " + str(config.REINTENTOS) +
                         " intentos. Último error: " + ultimo_error)


def embeber_textos(cliente, textos):
    """Devuelve un vector por texto, pidiéndolos en lotes de TAMANO_DEL_LOTE."""
    vectores = []
    for inicio in range(0, len(textos), config.TAMANO_DEL_LOTE):
        lote = textos[inicio:inicio + config.TAMANO_DEL_LOTE]
        respuesta = llamar_con_reintentos(lambda: cliente["api"].embeddings.create(
            model=cliente["modelo_embedding"], input=lote))
        for dato in respuesta.data:
            vectores.append(dato.embedding)
    return vectores


def preguntar(cliente, mensajes, modelo, temperatura, formato_json=False):
    """Manda los mensajes al modelo de chat y devuelve el texto de la respuesta."""
    argumentos = {"model": modelo, "messages": mensajes,
                  "temperature": temperatura, "max_tokens": config.MAX_TOKENS}
    if formato_json:
        argumentos["response_format"] = {"type": "json_object"}
    # **argumentos "desarma" el diccionario: es lo mismo que escribir
    # model=..., messages=..., temperature=..., uno por uno.
    respuesta = llamar_con_reintentos(
        lambda: cliente["api"].chat.completions.create(**argumentos))
    eleccion = respuesta.choices[0]
    texto = eleccion.message.content or ""
    if texto.strip() == "" and eleccion.finish_reason == "length":
        raise ErrorDelModelo("El modelo agotó max_tokens (" + str(config.MAX_TOKENS) +
                             ") razonando y no llegó a responder. Subí MAX_TOKENS en src/config.py.")
    if texto.strip() == "":
        raise ErrorDelModelo("El modelo devolvió una respuesta vacía.")
    return texto.strip()
