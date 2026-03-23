import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.app_state import app_data_dir


def session_dir() -> Path:
    path = app_data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_session(record: Dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = record.get("mode", "session")
    path = session_dir() / f"{timestamp}-{slug}.json"
    payload = dict(record)
    payload.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def list_sessions(limit: int = 25) -> List[Path]:
    files = sorted(session_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[: max(1, int(limit))]


def load_session(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    target = path
    if target is None:
        sessions = list_sessions(limit=1)
        target = sessions[0] if sessions else None
    if not target or not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
