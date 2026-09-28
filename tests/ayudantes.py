import math


def chunks_falsos(cantidad):
    """Chunks con vectores 2D en ángulos crecientes: el 1 es el más parecido a [1, 0]."""
    chunks = []
    for numero in range(cantidad):
        angulo = numero * 0.15
        chunks.append({
            "id": numero + 1,
            "seccion": "Sección de prueba",
            "texto": "Texto de prueba número " + str(numero + 1),
            "tokens": 60,
            "vector": [math.cos(angulo), math.sin(angulo)],
        })
    return chunks
