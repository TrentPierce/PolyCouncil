from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.app_state import app_config_dir, legacy_root_dir
from core.provider_config import (
    API_SERVICE_CUSTOM,
    PROVIDER_LM_STUDIO,
    PROVIDER_OLLAMA,
)
from core.secure_store import get_secret, secure_store_available, set_secret


APP_DIR = Path(__file__).resolve().parent.parent
LEGACY_ROOT = legacy_root_dir() if legacy_root_dir else APP_DIR
CONFIG_DIR = app_config_dir() if app_config_dir else APP_DIR
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LEGACY_SETTINGS_PATH = LEGACY_ROOT / "council_settings.json"
ACTIVE_API_KEY_REF = "active-provider"
SETTINGS_SCHEMA_VERSION = 2

_settings_lock = threading.Lock()


@dataclass
class SettingsRecord:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    provider_type: str = PROVIDER_LM_STUDIO
    base_url: str = "http://localhost:1234"
    api_key: str = ""
    model_path: str = ""
    api_service: str = API_SERVICE_CUSTOM
    debug: bool = False
    single_voter_enabled: bool = False
    single_voter_model: str = ""
    max_concurrency: int = 1
    roles_enabled: bool = False
    personas: list[dict[str, Any]] = field(default_factory=list)
    persona_assignments: dict[str, str] = field(default_factory=dict)
    provider_profiles: list[dict[str, Any]] = field(default_factory=list)
    rubric_weights: dict[str, int] = field(default_factory=dict)
    timeout_seconds: int = 120
    reduce_motion: bool = False
    theme_mode: str = "system"
    window_geometry: str = ""
    window_state: str = ""
    secure_keyring_available: bool = False
    secure_storage_ok: bool = True


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: Any, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except Exception:
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _normalize_profile(profile: Any) -> Optional[dict[str, str]]:
    if not isinstance(profile, dict):
        return None
    return {
        "id": str(profile.get("id") or uuid.uuid4()),
        "provider_type": str(profile.get("provider_type", PROVIDER_LM_STUDIO)),
        "api_service": str(profile.get("api_service", API_SERVICE_CUSTOM)),
        "base_url": str(profile.get("base_url", "")),
        "api_key": str(profile.get("api_key", "")),
        "model_path": str(profile.get("model_path", "")),
    }


def _normalize_settings_record(
    loaded: Optional[dict[str, Any]],
    *,
    default_provider_type: str,
    default_api_service: str,
) -> dict[str, Any]:
    raw = dict(loaded or {})
    schema_version = _as_int(raw.get("schema_version", SETTINGS_SCHEMA_VERSION), SETTINGS_SCHEMA_VERSION, minimum=1)
    provider_type = str(raw.get("provider_type", default_provider_type))
    if provider_type == PROVIDER_OLLAMA:
        default_base_url = "http://localhost:11434"
    else:
        default_base_url = "http://localhost:1234"

    record = SettingsRecord(
        schema_version=SETTINGS_SCHEMA_VERSION,
        provider_type=provider_type,
        base_url=str(raw.get("base_url") or default_base_url),
        api_key=str(raw.get("api_key", "")),
        model_path=str(raw.get("model_path", "")),
        api_service=str(raw.get("api_service", default_api_service)),
        debug=_as_bool(raw.get("debug", False)),
        single_voter_enabled=_as_bool(raw.get("single_voter_enabled", False)),
        single_voter_model=str(raw.get("single_voter_model", "")),
        max_concurrency=_as_int(raw.get("max_concurrency", 1), 1, minimum=1, maximum=8),
        roles_enabled=_as_bool(raw.get("roles_enabled", False)),
        personas=[entry for entry in raw.get("personas", []) if isinstance(entry, dict)],
        persona_assignments={
            str(key): str(value)
            for key, value in (raw.get("persona_assignments", {}) or {}).items()
        } if isinstance(raw.get("persona_assignments", {}), dict) else {},
        provider_profiles=[
            normalized
            for normalized in (_normalize_profile(profile) for profile in raw.get("provider_profiles", []) or [])
            if normalized is not None
        ],
        rubric_weights={
            str(key): _as_int(value, 0, minimum=0, maximum=10)
            for key, value in (raw.get("rubric_weights", {}) or {}).items()
        } if isinstance(raw.get("rubric_weights", {}), dict) else {},
        timeout_seconds=_as_int(raw.get("timeout_seconds", 120), 120, minimum=15, maximum=600),
        reduce_motion=_as_bool(raw.get("reduce_motion", False)),
        theme_mode=str(raw.get("theme_mode", "system")),
        window_geometry=str(raw.get("window_geometry", "")),
        window_state=str(raw.get("window_state", "")),
        secure_keyring_available=_as_bool(raw.get("secure_keyring_available", False)),
        secure_storage_ok=_as_bool(raw.get("secure_storage_ok", True), default=True),
    )

    normalized = asdict(record)
    normalized["settings_migrated_from_schema"] = schema_version
    return normalized


def _profile_secret_name(profile_id: str) -> str:
    return f"provider-profile::{profile_id}"


def _persist_api_key(secret_name: str, api_key: str) -> bool:
    if not api_key:
        if set_secret:
            set_secret(secret_name, "")
        return True
    if not set_secret:
        return False
    return bool(set_secret(secret_name, api_key))


def _load_api_key(secret_name: str) -> str:
    if not get_secret:
        return ""
    return get_secret(secret_name)


def _strip_persisted_secrets(settings_data: dict) -> tuple[dict, bool]:
    cleaned = dict(settings_data or {})
    secure_save_ok = True

    if "api_key" in cleaned:
        secure_save_ok = _persist_api_key(ACTIVE_API_KEY_REF, str(cleaned.get("api_key", ""))) and secure_save_ok
        cleaned.pop("api_key", None)

    profiles = []
    for profile in cleaned.get("provider_profiles", []) or []:
        if not isinstance(profile, dict):
            continue
        normalized = dict(profile)
        profile_id = str(normalized.get("id") or uuid.uuid4())
        normalized["id"] = profile_id
        if "api_key" in normalized:
            secure_save_ok = _persist_api_key(_profile_secret_name(profile_id), str(normalized.get("api_key", ""))) and secure_save_ok
            normalized.pop("api_key", None)
        profiles.append(normalized)
    if "provider_profiles" in cleaned:
        cleaned["provider_profiles"] = profiles

    return cleaned, secure_save_ok


def _hydrate_secrets(settings_data: dict) -> tuple[dict, bool]:
    hydrated = dict(settings_data or {})
    secure_load_ok = True

    active_key = _load_api_key(ACTIVE_API_KEY_REF)
    if not active_key and settings_data.get("api_key"):
        secure_load_ok = _persist_api_key(ACTIVE_API_KEY_REF, str(settings_data.get("api_key", ""))) and secure_load_ok
        active_key = str(settings_data.get("api_key", ""))
    hydrated["api_key"] = active_key

    profiles = []
    for profile in hydrated.get("provider_profiles", []) or []:
        if not isinstance(profile, dict):
            continue
        normalized = dict(profile)
        profile_id = str(normalized.get("id") or uuid.uuid4())
        normalized["id"] = profile_id
        profile_key = _load_api_key(_profile_secret_name(profile_id))
        if not profile_key and profile.get("api_key"):
            secure_load_ok = _persist_api_key(_profile_secret_name(profile_id), str(profile.get("api_key", ""))) and secure_load_ok
            profile_key = str(profile.get("api_key", ""))
        normalized["api_key"] = profile_key
        profiles.append(normalized)
    hydrated["provider_profiles"] = profiles
    hydrated["secure_keyring_available"] = bool(secure_store_available and secure_store_available())
    hydrated["secure_storage_ok"] = secure_load_ok
    return hydrated, secure_load_ok


def load_settings(
    *,
    settings_path: Optional[Path] = None,
    default_provider_type: str = PROVIDER_LM_STUDIO,
    default_api_service: str = API_SERVICE_CUSTOM,
) -> dict:
    path = settings_path or SETTINGS_PATH
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            loaded, _ = _hydrate_secrets(loaded)
            return _normalize_settings_record(
                loaded,
                default_provider_type=default_provider_type,
                default_api_service=default_api_service,
            )
        except Exception:
            pass
    defaults = _normalize_settings_record(
        {},
        default_provider_type=default_provider_type,
        default_api_service=default_api_service,
    )
    defaults["secure_keyring_available"] = bool(secure_store_available and secure_store_available())
    defaults["secure_storage_ok"] = True
    return defaults


def save_settings(
    settings_patch: dict,
    *,
    settings_path: Optional[Path] = None,
    warning_hook: Optional[Callable[[str], None]] = None,
) -> bool:
    path = settings_path or SETTINGS_PATH
    with _settings_lock:
        try:
            current = {}
            if path.exists():
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    if warning_hook:
                        warning_hook(f"Settings file corrupted, starting fresh: {exc}")
                    current = {}
            current.update(settings_patch or {})
            current = _normalize_settings_record(
                current,
                default_provider_type=str(current.get("provider_type", PROVIDER_LM_STUDIO)),
                default_api_service=str(current.get("api_service", API_SERVICE_CUSTOM)),
            )
            current.pop("settings_migrated_from_schema", None)
            current, secure_save_ok = _strip_persisted_secrets(current)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
            return secure_save_ok
        except PermissionError as exc:
            if warning_hook:
                warning_hook(f"Cannot write settings (permission denied): {exc}")
        except Exception as exc:
            if warning_hook:
                warning_hook(f"Failed to save settings: {exc}")
    return False
