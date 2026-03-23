from types import SimpleNamespace

from core.api_client import endpoints, parse_models_response


def test_parse_models_response_handles_openai_shape():
    provider = SimpleNamespace(provider_type="openai_compatible")
    data = {"data": [{"id": "gpt-4.1"}, {"name": "o4-mini"}, {"id": "gpt-4.1"}]}
    assert parse_models_response(provider, data) == ["gpt-4.1", "o4-mini"]


def test_parse_models_response_handles_ollama_shape():
    provider = SimpleNamespace(provider_type="ollama")
    data = {"models": [{"name": "llama3.2"}, {"name": "qwen3:32b"}, {"name": "llama3.2"}]}
    assert parse_models_response(provider, data) == ["llama3.2", "qwen3:32b"]


def test_endpoints_respects_versioned_hosted_base():
    provider = SimpleNamespace(
        provider_type="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        model_path="models",
    )
    resolved = endpoints(provider)
    assert resolved["chat"] == "https://openrouter.ai/api/v1/chat/completions"
    assert resolved["models"] == "https://openrouter.ai/api/v1/models"
