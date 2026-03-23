from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Callable, List, Optional

from core.app_state import app_data_dir

DATA_DIR = app_data_dir()
DB_PATH = DATA_DIR / "council_stats.db"


def ensure_db(*, db_path: Optional[Path] = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            winner TEXT,
            details TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def record_vote(question: str, winner: str, details: dict, *, db_path: Optional[Path] = None, debug_hook: Optional[Callable[[str, str], None]] = None) -> None:
    path = db_path or DB_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO votes (timestamp, question, winner, details) VALUES (?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), question, winner, json.dumps(details)),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        if debug_hook:
            debug_hook("record_vote error", str(exc))


def load_leaderboard(*, db_path: Optional[Path] = None) -> List[tuple[str, int]]:
    path = db_path or DB_PATH
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        rows = list(cur.execute("SELECT winner, COUNT(*) FROM votes GROUP BY winner ORDER BY COUNT(*) DESC"))
        conn.close()
        return [(row[0], int(row[1])) for row in rows]
    except Exception:
        return []


def reset_leaderboard(*, db_path: Optional[Path] = None) -> None:
    path = db_path or DB_PATH
    if path.exists():
        path.unlink()
    ensure_db(db_path=path)
