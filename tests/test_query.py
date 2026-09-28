import json

import pytest

from src import busqueda, llm_client, query
from tests.ayudantes import chunks_falsos


def cliente_falso():
    return {"modelo_chat": "chat-falso", "modelo_embedding": "embedding-falso",
            "modelo_evaluador": "chat-falso"}


def preparar_falsos(monkeypatch, respuesta="Sí, se puede [chunk 1]."):
    monkeypatch.setattr(llm_client, "embeber_textos", lambda cliente, textos: [[1.0, 0.0]])

    def preguntar_falso(cliente, mensajes, modelo, temperatura, formato_json=False):
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setattr(llm_client, "preguntar", preguntar_falso)
    return busqueda.preparar_busqueda(chunks_falsos(12))


def test_validar_pregunta():
    assert query.validar_pregunta("  ¿Hola?  ") == "¿Hola?"
    with pytest.raises(ValueError):
        query.validar_pregunta("   ")
    with pytest.raises(ValueError):
        query.validar_pregunta("x" * 501)


def test_responder_devuelve_exactamente_el_contrato(monkeypatch):
    preparada = preparar_falsos(monkeypatch)
    salida, error = query.responder(cliente_falso(), preparada, 2, "¿Puedo?")
    assert error is None
    assert list(salida.keys()) == query.CLAVES_DE_SALIDA
    assert salida["system_answer"] == "Sí, se puede [chunk 1]."
    assert 2 <= len(salida["chunks_related"]) <= 5
    json.dumps(salida)   # tiene que poder serializarse


def test_si_el_modelo_falla_igual_se_respeta_el_contrato(monkeypatch):
    preparada = preparar_falsos(monkeypatch, llm_client.ErrorDelModelo("proveedor caído"))
    salida, error = query.responder(cliente_falso(), preparada, 2, "¿Puedo?")
    assert error == "proveedor caído"
    assert list(salida.keys()) == query.CLAVES_DE_SALIDA
    assert "proveedor caído" in salida["system_answer"]
    assert len(salida["chunks_related"]) >= 2


def test_el_prompt_lleva_los_chunks_recuperados(monkeypatch):
    mensajes_enviados = []
    monkeypatch.setattr(llm_client, "preguntar",
                        lambda cliente, mensajes, modelo, temperatura, formato_json=False:
                        mensajes_enviados.append(mensajes) or "ok")
    query.generar_respuesta(cliente_falso(), "¿Puedo?",
                            [{"id": 3, "seccion": "Nómina", "texto": "Dato."}])
    sistema = mensajes_enviados[0][0]["content"]
    assert "[chunk 3 | Nómina]\nDato." in sistema
    assert "{CONTEXTO}" not in sistema
    assert mensajes_enviados[0][1] == {"role": "user", "content": "¿Puedo?"}


def test_validar_salida_rechaza_claves_de_mas():
    salida = {"user_question": "a", "system_answer": "b", "chunks_related": [], "score": 8}
    with pytest.raises(ValueError):
        query.validar_salida(salida)


def test_cargar_indice_rechaza_otro_modelo(tmp_path):
    ruta = tmp_path / "index.json"
    ruta.write_text(json.dumps({"modelo_embedding": "otro-modelo", "dimension": 2, "chunks": []}),
                    encoding="utf-8")
    with pytest.raises(query.ErrorDeIndice, match="otro-modelo"):
        query.cargar_indice(str(ruta), cliente_falso())


def test_cargar_indice_inexistente_explica_que_hacer(tmp_path):
    with pytest.raises(query.ErrorDeIndice, match="build_index"):
        query.cargar_indice(str(tmp_path / "no_existe.json"), cliente_falso())


def test_dimension_distinta_da_error(monkeypatch):
    monkeypatch.setattr(llm_client, "embeber_textos", lambda cliente, textos: [[1.0, 0.0, 0.0]])
    with pytest.raises(query.ErrorDeIndice):
        query.embeber_pregunta(cliente_falso(), "¿Hola?", 2)
