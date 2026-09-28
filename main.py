"""
Punto de entrada del proyecto:

    python main.py "¿Cómo solicito vacaciones?"

Hace lo mismo que python -m src.query (y acepta los mismos flags: --metodo,
--evaluar, --provider, --model), con una diferencia: si todavía no existe el
índice, primero lo construye.
"""

import json
import os
import sys

from src import build_index, config, llm_client, query


def leer_modelo_del_indice():
    """Modelo de embeddings con el que se guardó el índice existente, o None si no hay índice."""
    if not os.path.exists(config.RUTA_INDICE):
        return None
    with open(config.RUTA_INDICE, encoding="utf-8") as archivo:
        return json.load(archivo)["modelo_embedding"]


def asegurar_indice(proveedor):
    """
    Construye el índice si todavía no existe, o si existe pero quedó hecho con
    otro modelo de embeddings que el configurado. Devuelve False si no se pudo.
    """
    try:
        modelo_configurado = config.cargar_configuracion(proveedor)["modelo_embedding"]
        modelo_del_indice = leer_modelo_del_indice()
        if modelo_del_indice is None:
            print("Todavía no hay índice: lo construyo primero.", file=sys.stderr)
        elif modelo_del_indice != modelo_configurado:
            print("El índice se hizo con " + modelo_del_indice + " y el modelo configurado es " +
                  modelo_configurado + ": lo reconstruyo.", file=sys.stderr)
        else:
            return True
        indice = build_index.construir_indice(proveedor)
    except (FileNotFoundError, config.ErrorDeConfiguracion, build_index.ErrorDeIndexacion,
            llm_client.ErrorDelModelo) as problema:
        print("Error: " + str(problema), file=sys.stderr)
        return False
    # Por stderr, para que stdout quede sólo con el JSON de la respuesta.
    print(build_index.describir_indice(indice), file=sys.stderr)
    return True


def main():
    """Valida la pregunta, asegura el índice y responde."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = query.crear_parser().parse_args()
    if args.pregunta is not None:
        try:
            query.validar_pregunta(args.pregunta)
        except ValueError as problema:
            # Falla rápido: sin esto se gastarían llamadas a la API indexando
            # antes de descubrir que la pregunta es inválida.
            print("Error: " + str(problema), file=sys.stderr)
            sys.exit(2)
    if not asegurar_indice(args.provider):
        sys.exit(2)
    sys.exit(query.ejecutar_consulta(args))


if __name__ == "__main__":
    main()
