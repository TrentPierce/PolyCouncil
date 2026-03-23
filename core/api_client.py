from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, List, Optional

import aiohttp

from core.provider_config import (
    API_SERVICE_OPENROUTER,
    PROVIDER_LM_STUDIO,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI_COMPAT,
)


class ModelFetchError(RuntimeError):
    pass


def request_headers(provider: Any) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider.provider_type == PROVIDER_OPENAI_COMPAT and provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    if provider.provider_type == PROVIDER_OPENAI_COMPAT and provider.api_service == API_SERVICE_OPENROUTER:
        headers.setdefault("HTTP-Referer", "https://github.com/TrentPierce/PolyCouncil")
        headers.setdefault("X-Title", "PolyCouncil")
    return headers


def endpoints(provider: Any) -> dict[str, str]:
    base_url = provider.base_url.rstrip("/")
    if provider.provider_type == PROVIDER_LM_STUDIO:
        return {
            "chat": f"{base_url}/v1/chat/completions",
            "models": f"{base_url}/v1/models",
        }
    if provider.provider_type == PROVIDER_OLLAMA:
        return {
            "chat": f"{base_url}/api/chat",
            "models": f"{base_url}/api/tags",
        }
    model_path = provider.model_path or "v1/models"
    lower = base_url.lower()
    has_versioned_base = (
        lower.endswith("/v1")
        or lower.endswith("/v1beta")
        or lower.endswith("/v1beta/openai")
        or lower.endswith("/openai")
        or lower.endswith("/api/v1")
    )
    if has_versioned_base:
        chat_url = f"{base_url}/chat/completions"
        models_url = f"{base_url}/{(model_path or 'models').lstrip('/')}"
    else:
        chat_url = f"{base_url}/v1/chat/completions"
        models_url = f"{base_url}/{model_path}"
    return {
        "chat": chat_url,
        "models": models_url,
    }


def parse_models_response(provider: Any, data: dict[str, Any]) -> List[str]:
    ids: List[str] = []
    if provider.provider_type == PROVIDER_OLLAMA:
        models = data.get("models", [])
        if not isinstance(models, list):
            return []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name:
                ids.append(str(name))
        return sorted(set(ids))
    model_entries = data.get("data", [])
    if not isinstance(model_entries, list):
        return []
    for item in model_entries:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        if model_id:
            ids.append(str(model_id))
    return sorted(set(ids))


async def fetch_models(
    provider: Any,
    *,
    provider_label: Optional[Callable[[str], str]] = None,
    timeout_sec: int = 20,
) -> List[str]:
    model_url = endpoints(provider)["models"]
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url, headers=request_headers(provider), timeout=timeout) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise ModelFetchError(f"HTTP {resp.status} from {model_url}: {text[:800]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelFetchError(f"Endpoint returned non-JSON from {model_url}") from exc
        models = parse_models_response(provider, data)
        if models:
            return models
        raise ModelFetchError(
            f"No models found at {model_url}. Check whether the base URL already includes a path like /v1 or /v1/models."
        )
    except asyncio.TimeoutError as exc:
        provider_name = provider_label(provider.provider_type) if provider_label else provider.provider_type
        raise ModelFetchError(f"{provider_name} request timed out after {timeout_sec} seconds") from exc
    except Exception as exc:
        if isinstance(exc, ModelFetchError):
            raise
        provider_name = provider_label(provider.provider_type) if provider_label else provider.provider_type
        raise ModelFetchError(f"{provider_name} request failed: {exc}") from exc


async def call_model(
    session: aiohttp.ClientSession,
    provider: Any,
    model: str,
    user_prompt: str,
    *,
    debug_hook: Optional[Callable[[str, Any], None]] = None,
    temperature: Optional[float] = None,
    sys_prompt: Optional[str] = None,
    json_schema: Optional[dict[str, Any]] = None,
    timeout_sec: int = 120,
    images: Optional[List[str]] = None,
    web_search: bool = False,
) -> Any:
    url = endpoints(provider)["chat"]
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    headers = request_headers(provider)
    images = images or []

    if provider.provider_type == PROVIDER_OLLAMA:
        assembled_prompt = user_prompt if not sys_prompt else f"{sys_prompt}\n\n{user_prompt}"
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": assembled_prompt}],
        }
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}
        if debug_hook:
            debug_hook("CALL payload", {"provider": provider.provider_type, "url": url, "payload": payload})
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {model}:\n{text[:800]}")
                try:
                    data = json.loads(text)
                except Exception as exc:
                    raise RuntimeError(f"Non-JSON response for {model}:\n{text[:800]}") from exc
                if debug_hook:
                    debug_hook("CALL raw_response", data)
                return data.get("message", {}).get("content", "")
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Model timed out after {timeout_sec} seconds: {model}") from exc

    messages: list[dict[str, Any]] = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    if images:
        content_list: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_url in images:
            content_list.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append({"role": "user", "content": content_list})
    else:
        messages.append({"role": "user", "content": user_prompt})

    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if json_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    if web_search:
        payload["tools"] = [{
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        }]

    if debug_hook:
        debug_hook("CALL payload", {"provider": provider.provider_type, "url": url, "payload": payload})
    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {model}:\n{text[:800]}")
            try:
                data = json.loads(text)
            except Exception as exc:
                raise RuntimeError(f"Non-JSON response for {model}:\n{text[:800]}") from exc
            if debug_hook:
                debug_hook("CALL raw_response", data)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"Model timed out after {timeout_sec} seconds: {model}") from exc
