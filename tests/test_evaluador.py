import json

import pytest

from src import evaluador, llm_client

RAZON = "Usa los chunks 8 y 9, que responden la pregunta, pero omite el plazo del chunk 9."


def salida_de_ejemplo():
    return {"user_question": "¿Puedo cancelar?",
            "system_answer": "Sí, hasta 48 horas antes [chunk 8].",
            "chunks_related": [{"id": 8, "seccion": "Vacaciones", "texto": "…", "similitud": 0.8}]}


def evaluacion_en_json(relevancia=9, fidelidad=9, completitud=7, razon=RAZON):
    return json.dumps({"relevancia": relevancia, "fidelidad": fidelidad,
                       "completitud": completitud, "reason": razon})


def test_interpretar_una_evaluacion_valida():
    resultado = evaluador.interpretar_evaluacion(evaluacion_en_json())
    assert resultado["dimensiones"] == {"relevancia": 9, "fidelidad": 9, "completitud": 7}
    assert resultado["reason"] == RAZON


@pytest.mark.parametrize("texto", [
    evaluacion_en_json(relevancia=11),
    evaluacion_en_json(fidelidad="8"),
    evaluacion_en_json(completitud=True),
    evaluacion_en_json(razon="Corta."),
    "esto no es JSON",
    "[1, 2, 3]",
])
def test_interpretar_rechaza_evaluaciones_invalidas(texto):
    with pytest.raises(ValueError):
        evaluador.interpretar_evaluacion(texto)


def test_puntaje_es_el_promedio_redondeado():
    assert evaluador.calcular_puntaje({"relevancia": 9, "fidelidad": 9, "completitud": 7}) == 8


def test_la_fidelidad_baja_pone_un_techo():
    # promedio 7.67, pero la respuesta inventa: el puntaje no puede superar 3
    assert evaluador.calcular_puntaje({"relevancia": 10, "fidelidad": 3, "completitud": 10}) == 3


def test_el_prompt_lleva_pregunta_respuesta_y_chunks():
    contenido = evaluador.armar_mensajes(salida_de_ejemplo())[0]["content"]
    assert "¿Puedo cancelar?" in contenido
    assert "[chunk 8 | Vacaciones]" in contenido
    assert "{PREGUNTA}" not in contenido


def test_reintenta_una_vez_si_el_juez_responde_mal(monkeypatch):
    respuestas = ["no es JSON", evaluacion_en_json()]
    monkeypatch.setattr(llm_client, "preguntar",
                        lambda cliente, mensajes, modelo, temperatura, formato_json=False: respuestas.pop(0))
    resultado = evaluador.evaluar_respuesta({"modelo_evaluador": "juez"}, salida_de_ejemplo())
    assert resultado["score"] == 8
    assert resultado["reason"].startswith("Puntaje 8 (relevancia 9, fidelidad 9, completitud 7): ")
    assert len(resultado["reason"]) >= 50


def test_si_no_puede_evaluar_no_inventa_un_puntaje(monkeypatch):
    monkeypatch.setattr(llm_client, "preguntar",
                        lambda cliente, mensajes, modelo, temperatura, formato_json=False: "no es JSON")
    resultado = evaluador.evaluar_respuesta({"modelo_evaluador": "juez"}, salida_de_ejemplo())
    assert resultado["score"] is None
    assert len(resultado["reason"]) >= 50
