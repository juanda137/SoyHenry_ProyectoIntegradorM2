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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config.resolver_proveedor(None) == "gemini"
    assert config.resolver_proveedor(" OpenAI ") == "openai"


def test_proveedor_por_defecto_usa_openai_si_solo_hay_clave_de_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    assert config.resolver_proveedor(None) == "openai"


def test_proveedor_por_defecto_usa_gemini_si_solo_hay_clave_de_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "clave-de-prueba")
    assert config.resolver_proveedor(None) == "gemini"


def test_LLM_PROVIDER_explicito_gana_aunque_sobre_la_clave_de_otro_proveedor(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert config.resolver_proveedor(None) == "gemini"


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
