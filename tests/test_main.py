import json
import sys

import pytest

import main
from src import build_index, config


def test_asegura_indice_construye_si_no_existe(monkeypatch, tmp_path, capsys):
    ruta = tmp_path / "index.json"
    monkeypatch.setattr(config, "RUTA_INDICE", str(ruta))
    monkeypatch.setattr(config, "cargar_configuracion",
                        lambda proveedor: {"modelo_embedding": "modelo-1"})
    llamadas = []
    monkeypatch.setattr(build_index, "construir_indice",
                        lambda proveedor: llamadas.append(proveedor) or
                        {"modelo_embedding": "modelo-1"})
    monkeypatch.setattr(build_index, "describir_indice", lambda indice: "resumen")
    assert main.asegurar_indice("gemini") is True
    assert llamadas == ["gemini"]
    assert "lo construyo primero" in capsys.readouterr().err


def test_asegura_indice_no_reconstruye_si_el_modelo_coincide(monkeypatch, tmp_path):
    ruta = tmp_path / "index.json"
    ruta.write_text(json.dumps({"modelo_embedding": "modelo-1"}), encoding="utf-8")
    monkeypatch.setattr(config, "RUTA_INDICE", str(ruta))
    monkeypatch.setattr(config, "cargar_configuracion",
                        lambda proveedor: {"modelo_embedding": "modelo-1"})

    def fallar(proveedor):
        raise AssertionError("no debería reconstruir el índice")

    monkeypatch.setattr(build_index, "construir_indice", fallar)
    assert main.asegurar_indice("gemini") is True


def test_asegura_indice_reconstruye_si_el_modelo_cambio(monkeypatch, tmp_path, capsys):
    ruta = tmp_path / "index.json"
    ruta.write_text(json.dumps({"modelo_embedding": "viejo"}), encoding="utf-8")
    monkeypatch.setattr(config, "RUTA_INDICE", str(ruta))
    monkeypatch.setattr(config, "cargar_configuracion",
                        lambda proveedor: {"modelo_embedding": "nuevo"})
    llamadas = []
    monkeypatch.setattr(build_index, "construir_indice",
                        lambda proveedor: llamadas.append(proveedor) or
                        {"modelo_embedding": "nuevo"})
    monkeypatch.setattr(build_index, "describir_indice", lambda indice: "resumen")
    assert main.asegurar_indice("gemini") is True
    assert llamadas == ["gemini"]
    salida = capsys.readouterr().err
    assert "viejo" in salida and "nuevo" in salida


def test_asegura_indice_atrapa_error_de_configuracion(monkeypatch, tmp_path, capsys):
    ruta = tmp_path / "index.json"
    monkeypatch.setattr(config, "RUTA_INDICE", str(ruta))

    def fallar(proveedor):
        raise config.ErrorDeConfiguracion("falta la clave")

    monkeypatch.setattr(config, "cargar_configuracion", fallar)
    assert main.asegurar_indice("gemini") is False
    assert "falta la clave" in capsys.readouterr().err


def test_main_valida_la_pregunta_antes_de_indexar(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "   "])

    def fallar(proveedor):
        raise AssertionError("no debería intentar indexar sin validar antes")

    monkeypatch.setattr(main, "asegurar_indice", fallar)
    with pytest.raises(SystemExit) as salida:
        main.main()
    assert salida.value.code == 2
    assert "vacía" in capsys.readouterr().err
