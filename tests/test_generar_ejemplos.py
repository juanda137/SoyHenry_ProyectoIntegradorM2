from src import generar_ejemplos


def test_porcentaje_con_palabras_clave():
    chunks = [{"texto": "Podés cancelar las vacaciones hasta 48 horas antes."},
              {"texto": "Los recibos se guardan diez años."}]
    assert generar_ejemplos.porcentaje_con_palabras_clave("¿Puedo cancelar vacaciones?", chunks) == 50.0


def test_porcentaje_sin_chunks_es_cero():
    assert generar_ejemplos.porcentaje_con_palabras_clave("¿Algo?", []) == 0.0


def test_resumir_metricas_deja_afuera_la_pregunta_fuera_de_alcance():
    mediciones = [
        {"tipo": "si_no", "knn": {"porcentaje_con_palabras_clave": 100.0},
         "hibrida": {"porcentaje_con_palabras_clave": 100.0}},
        {"tipo": "como_hacer", "knn": {"porcentaje_con_palabras_clave": 50.0},
         "hibrida": {"porcentaje_con_palabras_clave": 80.0}},
        {"tipo": "fuera_de_alcance", "knn": {"porcentaje_con_palabras_clave": 0.0},
         "hibrida": {"porcentaje_con_palabras_clave": 0.0}},
    ]
    assert generar_ejemplos.resumir_metricas(mediciones) == {"knn": 75.0, "hibrida": 90.0}


def test_hay_al_menos_3_preguntas_y_una_fuera_de_alcance():
    tipos = []
    for ejemplo in generar_ejemplos.PREGUNTAS_DE_EJEMPLO:
        tipos.append(ejemplo["tipo"])
    assert len(tipos) >= 3
    assert "fuera_de_alcance" in tipos
