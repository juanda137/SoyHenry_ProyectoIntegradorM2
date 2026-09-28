"""
Verifica sobre el propio código dos requisitos de la rúbrica:
  * ninguna función supera 30 líneas (sin contar su docstring);
  * cada archivo tiene los imports arriba, después las funciones y al final
    el bloque principal (if __name__ == "__main__").
"""

import ast
import os

CARPETA_DEL_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGO_MAXIMO = 30
ORDEN = {"import": 0, "def": 1, "main": 2}


def archivos_a_revisar():
    """Todos los .py de src/ más main.py (si ya existe)."""
    carpeta_src = os.path.join(CARPETA_DEL_PROYECTO, "src")
    rutas = []
    for nombre in sorted(os.listdir(carpeta_src)):
        if nombre.endswith(".py"):
            rutas.append(os.path.join(carpeta_src, nombre))
    ruta_main = os.path.join(CARPETA_DEL_PROYECTO, "main.py")
    if os.path.exists(ruta_main):
        rutas.append(ruta_main)
    return rutas


def leer_arbol(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return ast.parse(archivo.read())


def lineas_sin_docstring(funcion):
    total = funcion.end_lineno - funcion.lineno + 1
    primero = funcion.body[0]
    es_docstring = (isinstance(primero, ast.Expr) and isinstance(primero.value, ast.Constant)
                    and isinstance(primero.value.value, str))
    if es_docstring:
        total = total - (primero.end_lineno - primero.lineno + 1)
    return total


def etapa_de(nodo):
    if isinstance(nodo, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
        return "def"
    if isinstance(nodo, ast.If) and "__name__" in ast.unparse(nodo.test):
        return "main"
    return None


def test_ninguna_funcion_supera_30_lineas():
    demasiado_largas = []
    for ruta in archivos_a_revisar():
        for nodo in ast.walk(leer_arbol(ruta)):
            if isinstance(nodo, ast.FunctionDef) and lineas_sin_docstring(nodo) > LARGO_MAXIMO:
                demasiado_largas.append(os.path.basename(ruta) + ":" + nodo.name)
    assert demasiado_largas == []


def test_imports_arriba_funciones_en_el_medio_main_al_final():
    for ruta in archivos_a_revisar():
        etapas = []
        for nodo in leer_arbol(ruta).body:
            etapa = etapa_de(nodo)
            if etapa is not None:
                etapas.append(etapa)
        assert etapas == sorted(etapas, key=lambda etapa: ORDEN[etapa]), ruta
