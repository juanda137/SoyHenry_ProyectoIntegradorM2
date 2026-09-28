"""
Punto de entrada del proyecto:

    python main.py "¿Cómo solicito vacaciones?"

Hace lo mismo que python -m src.query (y acepta los mismos flags: --metodo,
--evaluar, --provider, --model), con una diferencia: si todavía no existe el
índice, primero lo construye.
"""

import os
import sys

from src import build_index, config, llm_client, query


def asegurar_indice(proveedor):
    """Construye el índice si todavía no existe. Devuelve False si no se pudo."""
    if os.path.exists(config.RUTA_INDICE):
        return True
    print("Todavía no hay índice: lo construyo primero.", file=sys.stderr)
    try:
        indice = build_index.construir_indice(proveedor)
    except (FileNotFoundError, config.ErrorDeConfiguracion, build_index.ErrorDeIndexacion,
            llm_client.ErrorDelModelo) as problema:
        print("Error: " + str(problema), file=sys.stderr)
        return False
    # Por stderr, para que stdout quede sólo con el JSON de la respuesta.
    print(build_index.describir_indice(indice), file=sys.stderr)
    return True


def main():
    """Asegura el índice y responde la pregunta."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = query.crear_parser().parse_args()
    if not asegurar_indice(args.provider):
        sys.exit(2)
    sys.exit(query.ejecutar_consulta(args))


if __name__ == "__main__":
    main()
