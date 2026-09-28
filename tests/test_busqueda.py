import math

import pytest

from src import busqueda, config
from tests.ayudantes import chunks_falsos


def test_similitud_coseno_con_la_formula():
    resultado = busqueda.similitud_coseno([1, 0], [[1, 0], [0, 1], [1, 1]])
    assert resultado[0] == pytest.approx(1.0)
    assert resultado[1] == pytest.approx(0.0)
    assert resultado[2] == pytest.approx(1 / math.sqrt(2))


def test_palabras_clave_normaliza_y_filtra_palabras_vacias():
    assert busqueda.palabras_clave("¿Cómo solicito las VACACIONES?") == ["solicito", "vacaciones"]


def test_bm25_premia_la_coincidencia_exacta():
    indice = busqueda.preparar_bm25(["Configurar SSO con Entra ID",
                                     "Solicitar vacaciones",
                                     "Recibos de nómina en PDF"])
    assert busqueda.ranking_bm25(indice, "¿Cómo activo el SSO?") == [0]


def test_rrf_premia_lo_que_aparece_en_ambos_rankings():
    # 0: 1/61 + 1/62   2: 1/63 + 1/61   1: 1/62
    assert busqueda.fusionar_rrf([[0, 1, 2], [2, 0]]) == [0, 2, 1]


def test_buscar_devuelve_entre_2_y_5_chunks_con_el_formato_correcto():
    preparada = busqueda.preparar_busqueda(chunks_falsos(12))
    for metodo in ["knn", "hibrida"]:
        resultado = busqueda.buscar_chunks(preparada, [1, 0], "texto de prueba", metodo)
        assert config.TOP_K_MIN <= len(resultado) <= config.TOP_K_MAX
        assert sorted(resultado[0].keys()) == ["id", "seccion", "similitud", "texto"]


def test_filtro_por_rango_descarta_lo_poco_parecido(monkeypatch):
    monkeypatch.setattr(config, "SIMILITUD_MINIMA", 0.95)
    preparada = busqueda.preparar_busqueda(chunks_falsos(12))
    resultado = busqueda.buscar_chunks(preparada, [1, 0], "x", "knn")
    ids = []
    for chunk in resultado:
        ids.append(chunk["id"])
    assert ids == [1, 2, 3]   # cosenos 1.0, 0.989 y 0.955; el 4 da 0.900


def test_si_nada_pasa_el_umbral_devuelve_igual_el_minimo(monkeypatch):
    monkeypatch.setattr(config, "SIMILITUD_MINIMA", 2.0)
    preparada = busqueda.preparar_busqueda(chunks_falsos(12))
    resultado = busqueda.buscar_chunks(preparada, [1, 0], "x", "knn")
    assert len(resultado) == config.TOP_K_MIN


def test_metodo_desconocido_da_error():
    preparada = busqueda.preparar_busqueda(chunks_falsos(3))
    with pytest.raises(ValueError):
        busqueda.buscar_chunks(preparada, [1, 0], "x", "otro")


def test_formatear_chunks_numera_con_id_y_seccion():
    texto = busqueda.formatear_chunks([{"id": 7, "seccion": "Nómina", "texto": "Hola"}])
    assert texto == "[chunk 7 | Nómina]\nHola"
