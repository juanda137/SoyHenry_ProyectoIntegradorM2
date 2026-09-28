from types import SimpleNamespace

# openai==3.19.2 depende de httpx2 (un fork propio), no de httpx: si el
# proyecto no tiene httpx instalado a mano, "import httpx" rompe la
# recolección de tests en una instalación limpia. Las excepciones de
# openai aceptan objetos httpx2.Request/Response sin problema.
import httpx2
import pytest
from openai import APIConnectionError, APIStatusError

from src import config, llm_client


def pedido_falso():
    return httpx2.Request("POST", "https://ejemplo.test")


def error_de_red():
    return APIConnectionError(request=pedido_falso())


def error_http(codigo):
    respuesta = httpx2.Response(codigo, request=pedido_falso())
    return APIStatusError("error de prueba", response=respuesta, body=None)


def sin_esperas(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda segundos: None)


def test_reintenta_errores_pasajeros_y_despues_responde(monkeypatch):
    sin_esperas(monkeypatch)
    intentos = []

    def llamada():
        intentos.append(1)
        if len(intentos) < 3:
            raise error_de_red()
        return "ok"

    assert llm_client.llamar_con_reintentos(llamada) == "ok"
    assert len(intentos) == 3


def test_no_reintenta_errores_4xx(monkeypatch):
    sin_esperas(monkeypatch)
    intentos = []

    def llamada():
        intentos.append(1)
        raise error_http(401)

    with pytest.raises(llm_client.ErrorDelModelo):
        llm_client.llamar_con_reintentos(llamada)
    assert len(intentos) == 1


def test_agota_los_reintentos_con_errores_5xx(monkeypatch):
    sin_esperas(monkeypatch)
    intentos = []

    def llamada():
        intentos.append(1)
        raise error_http(503)

    with pytest.raises(llm_client.ErrorDelModelo):
        llm_client.llamar_con_reintentos(llamada)
    assert len(intentos) == config.REINTENTOS


def cliente_de_chat_falso(texto, motivo="stop"):
    eleccion = SimpleNamespace(message=SimpleNamespace(content=texto), finish_reason=motivo)
    respuesta = SimpleNamespace(choices=[eleccion])
    completions = SimpleNamespace(create=lambda **argumentos: respuesta)
    return {"api": SimpleNamespace(chat=SimpleNamespace(completions=completions))}


def test_preguntar_devuelve_el_texto_limpio():
    cliente = cliente_de_chat_falso("  Hola  ")
    assert llm_client.preguntar(cliente, [], "modelo", 0) == "Hola"


def test_preguntar_explica_si_se_agoto_max_tokens():
    cliente = cliente_de_chat_falso(None, "length")
    with pytest.raises(llm_client.ErrorDelModelo, match="max_tokens"):
        llm_client.preguntar(cliente, [], "modelo", 0)


def test_preguntar_rechaza_texto_parcial_cortado_por_max_tokens():
    # finish_reason "length" con texto no vacío: un JSON a medias, por
    # ejemplo, que si no se detecta parece una respuesta completa.
    cliente = cliente_de_chat_falso('{\n  "relevancia": 5,\n  "fidelidad": 10,\n  "completitud": 10,',
                                    "length")
    with pytest.raises(llm_client.ErrorDelModelo, match="max_tokens"):
        llm_client.preguntar(cliente, [], "modelo", 0)


def test_embeber_textos_pide_por_lotes(monkeypatch):
    monkeypatch.setattr(config, "TAMANO_DEL_LOTE", 2)
    pedidos = []

    def crear(model, input):
        pedidos.append(input)
        datos = []
        for texto in input:
            datos.append(SimpleNamespace(embedding=[float(len(texto))]))
        return SimpleNamespace(data=datos)

    cliente = {"api": SimpleNamespace(embeddings=SimpleNamespace(create=crear)),
               "modelo_embedding": "modelo"}
    vectores = llm_client.embeber_textos(cliente, ["a", "bb", "ccc", "dddd", "eeeee"])
    assert vectores == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert len(pedidos) == 3
