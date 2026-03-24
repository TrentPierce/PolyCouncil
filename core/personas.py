from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from core.app_state import app_config_dir, legacy_root_dir

DEFAULT_PERSONAS: List[dict] = [
    {"name": "None", "prompt": None, "builtin": True},
    {
        "name": "Meticulous fact-checker",
        "prompt": "You are a meticulous fact-checker. Prefer primary sources and verify each claim.",
        "builtin": True,
    },
    {
        "name": "Pragmatic engineer",
        "prompt": "You are a pragmatic engineer. Focus on feasible steps, tradeoffs, and edge cases.",
        "builtin": True,
    },
    {
        "name": "Cautious risk assessor",
        "prompt": "You are a cautious risk assessor. Identify failure modes and propose mitigations.",
        "builtin": True,
    },
    {
        "name": "Clear teacher",
        "prompt": "You are a clear teacher. Explain concepts simply with short examples where helpful.",
        "builtin": True,
    },
    {
        "name": "Systems thinker",
        "prompt": "You are a systems thinker. Map long-term interactions and consequences.",
        "builtin": True,
    },
]

_ROOT_DIR = legacy_root_dir() if legacy_root_dir else Path(__file__).resolve().parent.parent
_CONFIG_DIR = app_config_dir() if app_config_dir else _ROOT_DIR

DEFAULT_PERSONAS_PATH = _ROOT_DIR / "config" / "default_personas.json"
USER_PERSONAS_PATH = _CONFIG_DIR / "user_personas.json"


def persona_sort_key(persona: dict) -> tuple[int, str]:
    if persona["name"] == "None":
        return (0, "")
    return (
        1 if persona.get("builtin", False) else 2,
        persona["name"].lower(),
    )


def sort_personas_inplace(personas: List[dict]) -> None:
    personas.sort(key=persona_sort_key)


def sort_personas(personas: Iterable[dict]) -> List[dict]:
    result = [dict(persona) for persona in personas]
    sort_personas_inplace(result)
    return result


def persona_names(personas: Iterable[dict]) -> List[str]:
    return [persona["name"] for persona in personas]


def persona_prompt(personas: Iterable[dict], name: str) -> Optional[str]:
    persona = persona_by_name(personas, name)
    return persona.get("prompt") if persona else None


def persona_by_name(personas: Iterable[dict], name: str) -> Optional[dict]:
    for persona in personas:
        if persona["name"] == name:
            return persona
    return None


def assignment_count(assignments: Dict[str, str], name: str) -> int:
    return sum(1 for assigned in assignments.values() if assigned == name)


def cleanup_persona_assignments(personas: Iterable[dict], assignments: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
    valid_names = set(persona_names(personas))
    cleaned = dict(assignments)
    dirty = False
    for model, persona_name in list(cleaned.items()):
        if persona_name not in valid_names:
            cleaned[model] = "None"
            dirty = True
    return cleaned, dirty


def rename_persona_assignments(assignments: Dict[str, str], old_name: str, new_name: str) -> Dict[str, str]:
    renamed = dict(assignments)
    for model, assigned in list(renamed.items()):
        if assigned == old_name:
            renamed[model] = new_name
    return renamed


def clear_persona_assignment(assignments: Dict[str, str], name: str) -> Dict[str, str]:
    updated = dict(assignments)
    for model, assigned in list(updated.items()):
        if assigned == name:
            updated[model] = "None"
    return updated


def _read_persona_entries(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _write_persona_entries(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_persona_library(
    stored: Iterable[dict],
    *,
    default_personas: Optional[Iterable[dict]] = None,
    default_path: Path = DEFAULT_PERSONAS_PATH,
    user_path: Path = USER_PERSONAS_PATH,
) -> List[dict]:
    library: Dict[str, dict] = {}

    for persona in _read_persona_entries(default_path):
        name = persona.get("name")
        if name:
            library[name] = {
                "name": name,
                "prompt": persona.get("prompt_instruction"),
                "builtin": True,
            }

    for persona in _read_persona_entries(user_path):
        name = persona.get("name")
        if name:
            library[name] = {
                "name": name,
                "prompt": persona.get("prompt_instruction"),
                "builtin": False,
            }

    defaults = DEFAULT_PERSONAS if default_personas is None else default_personas
    for persona in defaults:
        name = persona.get("name")
        if name and name not in library:
            library[name] = dict(persona)

    for entry in stored or []:
        name = entry.get("name")
        if not name:
            continue
        if name in library and library[name].get("builtin"):
            continue
        library[name] = {
            "name": str(name),
            "prompt": entry.get("prompt"),
            "builtin": bool(entry.get("builtin", False)),
        }

    personas = list(library.values())
    sort_personas_inplace(personas)
    return personas


def add_user_persona(name: str, prompt: Optional[str], *, user_path: Path = USER_PERSONAS_PATH) -> dict:
    persona_id = f"u_{uuid.uuid4().hex[:8]}"
    entries = _read_persona_entries(user_path)
    entries.append({"id": persona_id, "name": name, "prompt_instruction": prompt or ""})
    _write_persona_entries(user_path, entries)
    return {"id": persona_id, "name": name, "prompt": prompt or None, "builtin": False}


def update_user_persona(existing_name: str, new_name: str, prompt: Optional[str], *, user_path: Path = USER_PERSONAS_PATH) -> bool:
    entries = _read_persona_entries(user_path)
    updated = False
    for entry in entries:
        if entry.get("name") == existing_name:
            entry["name"] = new_name
            entry["prompt_instruction"] = prompt or ""
            updated = True
            break
    if updated:
        _write_persona_entries(user_path, entries)
    return updated


def delete_user_persona(name: str, *, user_path: Path = USER_PERSONAS_PATH) -> bool:
    entries = _read_persona_entries(user_path)
    filtered = [entry for entry in entries if entry.get("name") != name]
    removed = len(filtered) != len(entries)
    if removed:
        _write_persona_entries(user_path, filtered)
    return removed


def build_persona_config(
    persona_name: str,
    personas: Iterable[dict],
    *,
    default_path: Path = DEFAULT_PERSONAS_PATH,
    user_path: Path = USER_PERSONAS_PATH,
) -> Dict[str, Optional[str]]:
    for entry in _read_persona_entries(default_path):
        if entry.get("name") == persona_name:
            return {
                "source": "default",
                "id": entry.get("id"),
                "one_time_prompt": "",
            }

    for entry in _read_persona_entries(user_path):
        if entry.get("name") == persona_name:
            return {
                "source": "user_custom",
                "id": entry.get("id"),
                "one_time_prompt": "",
            }

    prompt = persona_prompt(personas, persona_name)
    if prompt:
        return {
            "source": "one_time",
            "id": None,
            "one_time_prompt": prompt,
        }
    return {
        "source": "default",
        "id": None,
        "one_time_prompt": "",
    }
