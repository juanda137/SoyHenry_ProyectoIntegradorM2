import pytest

from src import config


def test_elegir_respeta_la_prioridad(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "desde-entorno")
    assert config.elegir("desde-cli", "CHAT_MODEL", "por-defecto") == "desde-cli"
    assert config.elegir(None, "CHAT_MODEL", "por-defecto") == "desde-entorno"
    monkeypatch.delenv("CHAT_MODEL")
    assert config.elegir(None, "CHAT_MODEL", "por-defecto") == "por-defecto"


def test_proveedor_por_defecto_es_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.resolver_proveedor(None) == "gemini"
    assert config.resolver_proveedor(" OpenAI ") == "openai"


def test_proveedor_desconocido_da_error_claro():
    with pytest.raises(config.ErrorDeConfiguracion):
        config.resolver_proveedor("otro")


def test_la_clave_de_ejemplo_cuenta_como_faltante(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your-key-here")
    with pytest.raises(config.ErrorDeConfiguracion) as error:
        config.leer_clave("gemini")
    assert "GEMINI_API_KEY" in str(error.value)


def test_la_clave_real_se_devuelve(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-prueba  ")
    assert config.leer_clave("openai") == "sk-prueba"
