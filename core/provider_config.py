from __future__ import annotations

from dataclasses import dataclass


PROVIDER_LM_STUDIO = "lm_studio"
PROVIDER_OPENAI_COMPAT = "openai_compatible"
PROVIDER_OLLAMA = "ollama"

PROVIDER_LABELS = {
    PROVIDER_LM_STUDIO: "LM Studio (OpenAI-compatible)",
    PROVIDER_OPENAI_COMPAT: "OpenAI-compatible API",
    PROVIDER_OLLAMA: "Ollama",
}

API_SERVICE_CUSTOM = "custom"
API_SERVICE_OPENAI = "openai"
API_SERVICE_OPENROUTER = "openrouter"
API_SERVICE_GEMINI = "gemini"

API_SERVICE_LABELS = {
    API_SERVICE_CUSTOM: "Custom",
    API_SERVICE_OPENAI: "OpenAI",
    API_SERVICE_OPENROUTER: "OpenRouter",
    API_SERVICE_GEMINI: "Google Gemini",
}

API_SERVICE_PRESETS = {
    API_SERVICE_CUSTOM: {"base_url": "", "model_path": ""},
    API_SERVICE_OPENAI: {"base_url": "https://api.openai.com", "model_path": "v1/models"},
    API_SERVICE_OPENROUTER: {"base_url": "https://openrouter.ai/api/v1", "model_path": "models"},
    API_SERVICE_GEMINI: {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model_path": "models"},
}


@dataclass
class ProviderConfig:
    provider_type: str
    base_url: str
    api_key: str = ""
    model_path: str = ""
    api_service: str = API_SERVICE_CUSTOM


def provider_label(provider_type: str) -> str:
    return PROVIDER_LABELS.get(provider_type, provider_type)


def api_service_label(service: str) -> str:
    return API_SERVICE_LABELS.get(service, API_SERVICE_LABELS[API_SERVICE_CUSTOM])


def normalize_api_service(service: str) -> str:
    if service in API_SERVICE_LABELS:
        return service
    return API_SERVICE_CUSTOM


def service_preset(service: str) -> dict:
    return dict(API_SERVICE_PRESETS.get(normalize_api_service(service), API_SERVICE_PRESETS[API_SERVICE_CUSTOM]))


def normalize_provider_type(provider_type: str) -> str:
    if provider_type in (PROVIDER_LM_STUDIO, PROVIDER_OPENAI_COMPAT, PROVIDER_OLLAMA):
        return provider_type
    return PROVIDER_LM_STUDIO


def provider_defaults(provider_type: str) -> tuple[str, str]:
    provider_type = normalize_provider_type(provider_type)
    if provider_type == PROVIDER_OLLAMA:
        return "http://localhost:11434", "model"
    return "http://localhost:1234", "v1/models"


def canonicalize_base_url(provider_type: str, base_url: str) -> str:
    normalized_type = normalize_provider_type(provider_type)
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned

    suffixes = []
    if normalized_type == PROVIDER_LM_STUDIO:
        suffixes = ["/v1/models", "/v1/chat/completions", "/v1"]
    elif normalized_type == PROVIDER_OLLAMA:
        suffixes = ["/api/tags", "/api/chat", "/api"]
    else:
        suffixes = ["/v1/models", "/models", "/v1/chat/completions", "/chat/completions"]

    lower_cleaned = cleaned.lower()
    for suffix in suffixes:
        if lower_cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.rstrip("/")


def make_provider_config(
    provider_type: str,
    base_url: str,
    api_key: str = "",
    model_path: str = "",
    api_service: str = API_SERVICE_CUSTOM,
) -> ProviderConfig:
    default_base, default_model_path = provider_defaults(provider_type)
    normalized_type = normalize_provider_type(provider_type)
    normalized_service = normalize_api_service(api_service)
    preset = service_preset(normalized_service) if normalized_type == PROVIDER_OPENAI_COMPAT else {"base_url": "", "model_path": ""}
    normalized_base = canonicalize_base_url(normalized_type, base_url or default_base)
    if normalized_type == PROVIDER_OPENAI_COMPAT and normalized_service != API_SERVICE_CUSTOM and not base_url.strip():
        normalized_base = preset["base_url"]
    normalized_path = (model_path or default_model_path).strip().lstrip("/")
    if normalized_type == PROVIDER_OPENAI_COMPAT and normalized_service != API_SERVICE_CUSTOM and not model_path.strip():
        normalized_path = preset["model_path"]
    if normalized_type == PROVIDER_LM_STUDIO:
        normalized_path = "v1/models"
    elif normalized_type == PROVIDER_OLLAMA:
        normalized_path = "model"
    return ProviderConfig(
        provider_type=normalized_type,
        base_url=normalized_base,
        api_key=(api_key or "").strip(),
        model_path=normalized_path,
        api_service=normalized_service,
    )
