import pytest

from src import chunking, config


def test_separar_en_bloques_reconoce_secciones_y_preguntas():
    texto = ("Intro del documento.\n"
             "# Sección: Nómina\n"
             "P: ¿Uno?\nR: Respuesta uno.\n\n"
             "P: ¿Dos?\nR: Respuesta dos.\n")
    assert chunking.separar_en_bloques(texto) == [
        {"seccion": "General", "texto": "Intro del documento."},
        {"seccion": "Nómina", "texto": "P: ¿Uno?\nR: Respuesta uno."},
        {"seccion": "Nómina", "texto": "P: ¿Dos?\nR: Respuesta dos."},
    ]


def test_bloque_chico_se_une_con_su_vecino_de_la_misma_seccion():
    largo = "P: ¿Larga?\nR: " + "dato importante " * 40
    bloques = [
        {"seccion": "A", "texto": "P: ¿Corta?\nR: Sí."},
        {"seccion": "A", "texto": largo},
        {"seccion": "B", "texto": "P: ¿Otra corta?\nR: No."},
    ]
    unidos = chunking.unir_bloques_chicos(bloques)
    assert len(unidos) == 2
    assert unidos[0]["texto"].startswith("P: ¿Corta?")
    assert "¿Larga?" in unidos[0]["texto"]
    assert unidos[1]["seccion"] == "B"   # nunca se cruza de sección


def test_separar_en_oraciones_no_corta_pasos_numerados():
    texto = "Seguí estos pasos: 1. Entrá a Ajustes. 2. Elegí Seguridad."
    assert chunking.separar_en_oraciones(texto) == [
        "Seguí estos pasos: 1. Entrá a Ajustes.",
        "2. Elegí Seguridad.",
    ]


def test_bloque_grande_se_parte_por_oraciones_con_solapamiento():
    oraciones = []
    for numero in range(60):
        oraciones.append("Esta es la oración número " + str(numero) + " del procedimiento largo.")
    pedazos = chunking.partir_bloque_grande(" ".join(oraciones))
    assert len(pedazos) >= 2
    for pedazo in pedazos:
        assert chunking.contar_tokens(pedazo) <= config.CHUNK_MAX_TOKENS
    ultima_del_primero = chunking.separar_en_oraciones(pedazos[0])[-1]
    assert pedazos[1].startswith(ultima_del_primero)


def test_cargar_documento_en_latin1(tmp_path):
    ruta = tmp_path / "doc.txt"
    ruta.write_bytes("Política de nómina".encode("latin-1"))
    assert chunking.cargar_documento(str(ruta)) == "Política de nómina"


def test_documento_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        chunking.cargar_documento(str(tmp_path / "no_existe.txt"))


def test_oraciones_perdidas_detecta_texto_faltante():
    texto = "# Sección: A\nP: ¿Uno?\nR: Respuesta completa."
    chunks = chunking.partir_en_chunks(texto)
    assert chunking.oraciones_perdidas(texto, chunks) == []
    chunks[0]["texto"] = "P: ¿Uno?"
    assert chunking.oraciones_perdidas(texto, chunks) == ["R: Respuesta completa."]


# --- Sobre el documento real -------------------------------------------------

def documento_real():
    texto = chunking.cargar_documento(config.RUTA_DOCUMENTO)
    return texto, chunking.partir_en_chunks(texto)


def test_documento_real_tiene_al_menos_1000_palabras():
    texto, chunks = documento_real()
    assert len(texto.split()) >= 1000


def test_documento_real_da_20_chunks_o_mas():
    texto, chunks = documento_real()
    assert len(chunks) >= config.MINIMO_DE_CHUNKS


def test_documento_real_chunks_entre_50_y_500_tokens():
    texto, chunks = documento_real()
    for chunk in chunks:
        assert config.CHUNK_MIN_TOKENS <= chunk["tokens"] <= config.CHUNK_MAX_TOKENS, chunk["id"]


def test_documento_real_se_captura_completo():
    texto, chunks = documento_real()
    assert chunking.oraciones_perdidas(texto, chunks) == []


def test_ids_consecutivos_desde_1():
    texto, chunks = documento_real()
    ids = []
    for chunk in chunks:
        ids.append(chunk["id"])
    assert ids == list(range(1, len(chunks) + 1))
