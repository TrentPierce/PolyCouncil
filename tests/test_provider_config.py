from council import (
    API_SERVICE_GEMINI,
    API_SERVICE_OPENROUTER,
    PROVIDER_LM_STUDIO,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI_COMPAT,
    endpoints,
    make_provider_config,
    parse_models_response,
    request_headers,
)


def test_make_provider_config_defaults_and_normalization():
    cfg = make_provider_config(PROVIDER_OLLAMA, "", "", "")
    assert cfg.provider_type == PROVIDER_OLLAMA
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model_path == "model"

    cfg2 = make_provider_config("unknown_provider", "", "", "")
    assert cfg2.provider_type == PROVIDER_LM_STUDIO
    assert cfg2.base_url == "http://localhost:1234"
    assert cfg2.model_path == "v1/models"


def test_parse_models_response_openai_compatible():
    provider = make_provider_config(PROVIDER_OPENAI_COMPAT, "http://example.com", "k", "v1/models")
    data = {"data": [{"id": "m1"}, {"name": "m2"}, {"id": "m1"}]}
    assert parse_models_response(provider, data) == ["m1", "m2"]


def test_parse_models_response_ollama():
    provider = make_provider_config(PROVIDER_OLLAMA, "http://localhost:11434", "", "")
    data = {"models": [{"name": "llama3"}, {"name": "phi4"}, {"name": "llama3"}]}
    assert parse_models_response(provider, data) == ["llama3", "phi4"]


def test_openrouter_endpoints_and_headers():
    provider = make_provider_config(
        PROVIDER_OPENAI_COMPAT,
        "",
        "sk-or-test",
        "",
        api_service=API_SERVICE_OPENROUTER,
    )
    ep = endpoints(provider)
    assert ep["chat"] == "https://openrouter.ai/api/v1/chat/completions"
    assert ep["models"] == "https://openrouter.ai/api/v1/models"
    headers = request_headers(provider)
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_gemini_compatible_endpoints():
    provider = make_provider_config(
        PROVIDER_OPENAI_COMPAT,
        "",
        "gemini-key",
        "",
        api_service=API_SERVICE_GEMINI,
    )
    ep = endpoints(provider)
    assert ep["chat"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert ep["models"] == "https://generativelanguage.googleapis.com/v1beta/openai/models"
