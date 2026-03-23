from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

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

_settings_lock = threading.Lock()


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
            provider_type = loaded.get("provider_type", default_provider_type)
            if provider_type == PROVIDER_OLLAMA:
                loaded.setdefault("base_url", "http://localhost:11434")
            else:
                loaded.setdefault("base_url", "http://localhost:1234")
            loaded.setdefault("provider_type", provider_type)
            loaded.setdefault("api_key", "")
            loaded.setdefault("model_path", "")
            loaded.setdefault("api_service", default_api_service)
            return loaded
        except Exception:
            pass
    defaults = {
        "provider_type": default_provider_type,
        "base_url": "http://localhost:1234",
        "api_key": "",
        "model_path": "",
        "api_service": default_api_service,
        "debug": False,
        "single_voter_enabled": False,
        "single_voter_model": "",
        "max_concurrency": 1,
        "roles_enabled": False,
        "personas": [],
        "persona_assignments": {},
    }
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
