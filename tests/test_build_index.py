import json

import pytest

from src import build_index, config, llm_client


def cliente_falso():
    return {"proveedor": "gemini", "modelo_embedding": "embedding-falso"}


def test_verificar_chunks_detecta_pocos_chunks_y_tamanos_invalidos():
    texto = "P: ¿Uno?\nR: Uno."
    chunks = [{"id": 1, "seccion": "General", "texto": "P: ¿Uno?\nR: Uno.", "tokens": 8}]
    problemas = build_index.verificar_chunks(texto, chunks)
    assert len(problemas) == 2
    assert "al menos 20" in problemas[0]
    assert "8 tokens" in problemas[1]


def test_generar_embeddings_detecta_cantidad_distinta(monkeypatch):
    monkeypatch.setattr(llm_client, "embeber_textos", lambda cliente, textos: [[0.1]])
    chunks = [{"seccion": "A", "texto": "uno"}, {"seccion": "A", "texto": "dos"}]
    with pytest.raises(build_index.ErrorDeIndexacion):
        build_index.generar_embeddings(cliente_falso(), chunks)


def test_armar_indice_junta_chunks_vectores_y_metadata():
    chunks = [{"id": 1, "seccion": "A", "texto": "uno", "tokens": 60}]
    indice = build_index.armar_indice(cliente_falso(), chunks, [[0.1, 0.2, 0.3]])
    assert indice["dimension"] == 3
    assert indice["modelo_embedding"] == "embedding-falso"
    assert indice["cantidad_de_chunks"] == 1
    assert indice["chunks"][0]["vector"] == [0.1, 0.2, 0.3]
    assert "vector" not in chunks[0]   # no modifica la lista original


def test_construir_indice_de_punta_a_punta_sin_api(monkeypatch, tmp_path):
    ruta = tmp_path / "index" / "index.json"
    monkeypatch.setattr(config, "RUTA_INDICE", str(ruta))
    monkeypatch.setattr(llm_client, "crear_cliente", lambda proveedor=None: cliente_falso())

    def embeber_falso(cliente, textos):
        vectores = []
        for texto in textos:
            vectores.append([float(len(texto)), 1.0])
        return vectores

    monkeypatch.setattr(llm_client, "embeber_textos", embeber_falso)
    indice = build_index.construir_indice()
    with open(ruta, encoding="utf-8") as archivo:
        guardado = json.load(archivo)
    assert guardado["cantidad_de_chunks"] == len(guardado["chunks"]) >= 20
    assert guardado["dimension"] == 2
    assert "chunks:" in build_index.describir_indice(indice)
