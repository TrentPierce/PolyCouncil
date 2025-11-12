
# council_gui_qt.py
# GUI: PySide6 (Qt widgets)
# Deps: PySide6, aiohttp, requests
# Optional: qdarktheme (auto dark/light), lmstudio (local-only "loaded models" detection)

import asyncio
import aiohttp
import requests
import sqlite3
import datetime
import json
import threading
import re
import random
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Any, Iterable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# Try optional theming (follows system)
try:
    import qdarktheme  # type: ignore
except Exception:
    qdarktheme = None

# -----------------------
# Debug logging switches
# -----------------------
DEBUG_VOTING = True                 # set False to silence
LOG_TRUNCATE: Optional[int] = None  # e.g., 8000 to cap output, or None for full
LOG_SINK: Optional[Callable[[str, str], None]] = None

DEFAULT_PERSONAS: List[dict] = [
    {"name": "None", "prompt": None, "builtin": True},
    {"name": "Meticulous fact-checker", "prompt": "You are a meticulous fact-checker. Prefer primary sources and verify each claim.", "builtin": True},
    {"name": "Pragmatic engineer", "prompt": "You are a pragmatic engineer. Focus on feasible steps, tradeoffs, and edge cases.", "builtin": True},
    {"name": "Cautious risk assessor", "prompt": "You are a cautious risk assessor. Identify failure modes and propose mitigations.", "builtin": True},
    {"name": "Clear teacher", "prompt": "You are a clear teacher. Explain concepts simply with short examples where helpful.", "builtin": True},
    {"name": "Data analyst", "prompt": "You are a data analyst. Structure answers into bullets, highlight assumptions and limits.", "builtin": True},
    {"name": "Systems thinker", "prompt": "You are a systems thinker. Map long-term interactions and consequences.", "builtin": True},
]


def _dbg(label: str, text: Any):
    if not DEBUG_VOTING:
        return
    try:
        if isinstance(text, (dict, list)):
            s = json.dumps(text, indent=2, ensure_ascii=False)
        else:
            s = "" if text is None else str(text)
    except Exception:
        s = str(text)
    if LOG_TRUNCATE and len(s) > LOG_TRUNCATE:
        s = s[:LOG_TRUNCATE] + f"\n...[truncated {len(s) - LOG_TRUNCATE} chars]"
    if LOG_SINK:
        try:
            LOG_SINK(label, s)
        except Exception:
            pass
    print(f"\n===== {label} =====\n{s}\n", flush=True)


def set_log_sink(callback: Optional[Callable[[str, str], None]]):
    global LOG_SINK
    LOG_SINK = callback

def create_app_icon(size: int = 256) -> QtGui.QIcon:
    size = max(64, int(size))
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    bg_color = QtGui.QColor("#1f80d6")
    radius = size * 0.2
    bg_path = QtGui.QPainterPath()
    bg_path.addRoundedRect(QtCore.QRectF(0, 0, size, size), radius, radius)
    painter.fillPath(bg_path, bg_color)

    stripe_color = QtGui.QColor(255, 255, 255, 220)
    stripe_width = size * 0.08
    stripe_length = size * 0.38
    stripe_start_x = size * 0.14
    stripe_start_y = size * 0.34
    stripe_gap = stripe_width * 0.8
    painter.setBrush(stripe_color)
    painter.setPen(QtCore.Qt.NoPen)
    for i in range(3):
        rect = QtCore.QRectF(
            stripe_start_x,
            stripe_start_y + i * (stripe_width + stripe_gap),
            stripe_length,
            stripe_width,
        )
        painter.drawRoundedRect(rect, stripe_width * 0.5, stripe_width * 0.5)

    bubble_rect = QtCore.QRectF(size * 0.36, size * 0.26, size * 0.48, size * 0.46)
    bubble_path = QtGui.QPainterPath()
    bubble_path.addRoundedRect(bubble_rect, size * 0.12, size * 0.12)

    tail = QtGui.QPolygonF(
        [
            QtCore.QPointF(bubble_rect.left() + bubble_rect.width() * 0.18, bubble_rect.bottom()),
            QtCore.QPointF(bubble_rect.left() + bubble_rect.width() * 0.36, bubble_rect.bottom()),
            QtCore.QPointF(bubble_rect.left() + bubble_rect.width() * 0.26, bubble_rect.bottom() + size * 0.12),
        ]
    )
    bubble_path.addPolygon(tail)
    painter.fillPath(bubble_path, QtGui.QColor("white"))

    check_pen = QtGui.QPen(bg_color, size * 0.085, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
    painter.setPen(check_pen)
    check_points = QtGui.QPolygonF(
        [
            QtCore.QPointF(bubble_rect.left() + bubble_rect.width() * 0.18, bubble_rect.center().y()),
            QtCore.QPointF(bubble_rect.left() + bubble_rect.width() * 0.38, bubble_rect.bottom() - bubble_rect.height() * 0.18),
            QtCore.QPointF(bubble_rect.right() - bubble_rect.width() * 0.18, bubble_rect.top() + bubble_rect.height() * 0.24),
        ]
    )
    painter.drawPolyline(check_points)

    painter.end()
    return QtGui.QIcon(pixmap)

def short_id(mid: str, n: int = 28) -> str:
    try:
        if len(mid) <= n:
            return mid
        head = mid[: n // 2 - 1]
        tail = mid[-(n // 2 - 1):]
        return f"{head}…{tail}"
    except Exception:
        return str(mid)

# -----------------------
# Paths / persistence
# -----------------------
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "council_stats.db"
SETTINGS_PATH = APP_DIR / "council_settings.json"

def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "base_url": "http://localhost:1234",
        "debug": False,
        "single_voter_enabled": False,
        "single_voter_model": "",
        "max_concurrency": 1,
        "roles_enabled": False,
        "personas": [],
        "persona_assignments": {},
    }

def save_settings(s: dict):
    try:
        current = {}
        if SETTINGS_PATH.exists():
            try:
                current = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                current = {}
        current.update(s or {})
        SETTINGS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# -----------------------
# Database (leaderboard)
# -----------------------
def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS votes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        winner TEXT,
        details TEXT
    )
    """)
    conn.commit()
    conn.close()

def record_vote(question: str, winner: str, details: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO votes (timestamp, question, winner, details) VALUES (?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), question, winner, json.dumps(details))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        _dbg("record_vote error", str(e))

def load_leaderboard() -> List[tuple[str, int]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        rows = list(cur.execute("SELECT winner, COUNT(*) FROM votes GROUP BY winner ORDER BY COUNT(*) DESC"))
        conn.close()
        return [(r[0], int(r[1])) for r in rows]
    except Exception:
        return []

# -----------------------
# LM Studio helpers
# -----------------------
def endpoints(base_url: str):
    b = base_url.rstrip("/")
    return {
        "chat":   f"{b}/v1/chat/completions",
        "models": f"{b}/v1/models",
    }

def fetch_models_from_lmstudio(base_url: str) -> List[str]:
    try:
        r = requests.get(endpoints(base_url)["models"], timeout=10)
        r.raise_for_status()
        data = r.json()
        ids = []
        for item in data.get("data", []):
            mid = item.get("id") or item.get("name")
            if mid:
                ids.append(mid)
        return sorted(set(ids))
    except Exception as e:
        print("fetch_models_from_lmstudio error:", e)
        return []

# -----------------------
# Concurrency helper
# -----------------------
async def run_limited(max_concurrency: int, callables: Iterable[Callable[[], Awaitable[Any]]]):
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    tasks = []
    async def runner(fn):
        async with sem:
            return await fn()
    for fn in callables:
        tasks.append(asyncio.create_task(runner(fn)))
    return await asyncio.gather(*tasks)

# -----------------------
# Model invocation
# -----------------------
async def call_model(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float = 0.2,
    sys_prompt: Optional[str] = None,
    json_schema: Optional[dict] = None,
    timeout_sec: int = 120
) -> Any:
    """
    Calls LM Studio's OpenAI-compatible chat API.
    If json_schema is provided, uses response_format={"type":"json_schema","json_schema":{...}}.
    Returns the content string from the first choice.
    """
    url = endpoints(base_url)["chat"]
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if json_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

    _dbg("CALL payload", {"url": url, "payload": payload})
    timeout=aiohttp.ClientTimeout(total=timeout_sec)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        txt = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {model}:\n{txt[:800]}")
        try:
            data = json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"Non-JSON response for {model}:\n{txt[:800]}") from e
        _dbg("CALL raw_response", data)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content

# -----------------------
# Voting schema & parsing
# -----------------------
RUBRIC_WEIGHTS = {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}

VOTE_INSTRUCTIONS = """\
You are voting on the BEST answer to the user's question from several model candidates.

Score EACH candidate (not including yourself) using integer scales:
- correctness(0-5), relevance(0-3), specificity(0-3), safety(0-2), conciseness(0-1).

Return ONLY JSON with this schema:
{
  "scores": {
     "1": {"correctness": int, "relevance": int, "specificity": int, "safety": int, "conciseness": int},
     "2": {...},
     "...": {...}
  },
  "final_pick": int,   // the index of your overall winner
  "reasoning": string  // brief rationale
}
No extra keys, no prose. Use only integers in the specified ranges.
"""

def ballot_json_schema(num_candidates: int) -> dict:
    # Build a strict schema for the response_format
    properties_scores = {}
    for i in range(1, num_candidates + 1):
        properties_scores[str(i)] = {
            "type": "object",
            "properties": {
                "correctness": {"type": "integer", "minimum": 0, "maximum": 5},
                "relevance": {"type": "integer", "minimum": 0, "maximum": 3},
                "specificity": {"type": "integer", "minimum": 0, "maximum": 3},
                "safety": {"type": "integer", "minimum": 0, "maximum": 2},
                "conciseness": {"type": "integer", "minimum": 0, "maximum": 1},
            },
            "required": ["correctness", "relevance", "specificity", "safety", "conciseness"],
            "additionalProperties": False
        }
    schema = {
        "name": "vote_ballot",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": properties_scores,
                    "required": list(properties_scores.keys()),
                    "additionalProperties": False
                },
                "final_pick": {"type": "integer", "minimum": 1, "maximum": num_candidates},
                "reasoning": {"type": "string"}
            },
            "required": ["scores", "final_pick"],
            "additionalProperties": False
        }
    }
    return schema

def safe_load_vote_json(text: str) -> Optional[dict]:
    """
    Try to parse the first JSON object present.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def validate_ballot(idx_map_peer: Dict[int, str], ballot: dict) -> Tuple[bool, str]:
    try:
        scores = ballot.get("scores", {})
        fp = ballot.get("final_pick", None)
        if not isinstance(scores, dict) or not isinstance(fp, int):
            return False, "scores/final_pick missing"

        required = set(idx_map_peer.keys())
        present = set(int(k) for k in scores.keys() if str(k).isdigit())
        if required != present:
            return False, f"indices mismatch required={sorted(required)} present={sorted(present)}"

        for i, sc in scores.items():
            for k, lo, hi in [
                ("correctness", 0, 5), ("relevance", 0, 3), ("specificity", 0, 3),
                ("safety", 0, 2), ("conciseness", 0, 1)
            ]:
                v = sc.get(k, None)
                if not isinstance(v, int) or v < lo or v > hi:
                    return False, f"bad {k} for {i}: {v}"
        if fp not in required:
            return False, f"final_pick {fp} not in {sorted(required)}"
        return True, "ok"
    except Exception as e:
        return False, f"exception {e}"

# -----------------------
# Voting worker
# -----------------------
async def vote_one(
    session: aiohttp.ClientSession,
    base_url: str,
    voter_model_id: str,
    question: str,
    answers: Dict[str, str],
    selected_models: List[str],
) -> Tuple[str, Optional[dict], str]:
    """
    Ask a (voter) model to score peer answers and pick a winner.
    Returns: (voter_id, parsed_ballot, status_message)
    """
    peer_models = [m for m in selected_models if m != voter_model_id]
    idx_map_peer = {i + 1: m for i, m in enumerate(peer_models)}

    if not peer_models:
        return voter_model_id, None, "No peer candidates to score."

    parts = [VOTE_INSTRUCTIONS, "", f"Question:\n{question}", "", "Candidates:"]
    for i, m in idx_map_peer.items():
        ans = answers.get(m, "")
        parts.append(f"[{i}] {short_id(m)}:\n{ans}")
    prompt = "\n\n".join(parts)

    try:
        schema = ballot_json_schema(num_candidates=len(idx_map_peer))
        content = await call_model(
            session, base_url, voter_model_id, prompt,
            temperature=0.0,
            sys_prompt="Be precise. Return only JSON according to the schema.",
            json_schema=schema
        )
        _dbg(f"VOTE raw OUTPUT json_schema (voter={voter_model_id})", content)
        parsed = safe_load_vote_json(content)
        if parsed:
            ok, msg = validate_ballot(idx_map_peer, parsed)
            if ok:
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", ""),
                }
                _dbg(f"VOTE ACCEPTED json_schema (voter={voter_model_id})", normalized)
                return voter_model_id, normalized, f"Valid ballot via json_schema ({msg})"
    except Exception as e:
        _dbg(f"VOTE ERROR json_schema (voter={voter_model_id})", str(e))

    try:
        content = await call_model(
            session, base_url, voter_model_id, prompt,
            temperature=0.0,
            sys_prompt="Return only JSON for the ballot. No extra text.",
            json_schema=None
        )
        _dbg(f"VOTE raw OUTPUT text (voter={voter_model_id})", content)
        parsed = safe_load_vote_json(content)
        if parsed:
            ok, msg = validate_ballot(idx_map_peer, parsed)
            if ok:
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", ""),
                }
                _dbg(f"VOTE ACCEPTED text (voter={voter_model_id})", normalized)
                return voter_model_id, normalized, f"Valid ballot via text ({msg})"
            else:
                _dbg(f"VOTE REJECTED text (voter={voter_model_id})", msg)
    except Exception as e:
        _dbg(f"VOTE ERROR text (voter={voter_model_id})", str(e))

    _dbg(f"VOTE FAILED ALL ATTEMPTS (voter={voter_model_id})", "No valid ballot produced")
    return voter_model_id, None, f"Failed to obtain valid ballot from {short_id(voter_model_id)}"

# -----------------------
# Council orchestration
# -----------------------
async def council_round(
    base_url: str,
    selected_models: List[str],
    question: str,
    roles: Dict[str, Optional[str]],
    status_cb: Callable[[str], None],
    max_concurrency: int = 1,
    voter_override: Optional[List[str]] = None
):
    status_cb("Collecting answers…")
    async with aiohttp.ClientSession() as session:

        async def answer_one(model_id: str) -> tuple[str, str]:
            user_prompt = question
            sys_p = roles.get(model_id) or None
            try:
                ans = await call_model(session, base_url, model_id, user_prompt, temperature=0.5, sys_prompt=sys_p)
                if isinstance(ans, dict):
                    ans = json.dumps(ans, ensure_ascii=False)
                return model_id, ans
            except Exception as e1:
                return model_id, f"[ERROR fetching answer]\n{e1}"

        pairs = await run_limited(max_concurrency, [lambda m=m: answer_one(m) for m in selected_models])
        answers = {m: a for m, a in pairs}
        errors = {m: a for m, a in pairs if isinstance(a, str) and a.startswith("[ERROR")}

        status_cb("Models are voting…")

        voters_to_use = voter_override if voter_override else selected_models
        vote_results = await run_limited(
            max_concurrency,
            [lambda m=m: vote_one(session, base_url, m, question, answers, selected_models) for m in voters_to_use]
        )

        valid_votes: Dict[str, dict] = {}
        invalid_votes: Dict[str, str] = {}
        vote_messages: Dict[str, str] = {}

        for voter_id, ballot, msg in vote_results:
            vote_messages[voter_id] = msg
            if ballot is not None:
                valid_votes[voter_id] = ballot
            else:
                invalid_votes[voter_id] = msg

        if invalid_votes:
            status_cb(f"⚠ {len(invalid_votes)}/{len(voters_to_use)} invalid ballots: {', '.join(short_id(m) for m in invalid_votes)}")

        totals: Dict[str, int] = {mid: 0 for mid in selected_models}
        for ballot in valid_votes.values():
            for candidate_mid, score_dict in ballot["scores"].items():
                weighted = sum(score_dict[k] * RUBRIC_WEIGHTS[k] for k in RUBRIC_WEIGHTS.keys())
                totals[candidate_mid] = totals.get(candidate_mid, 0) + int(weighted)

        if not valid_votes:
            status_cb("⚠ NO VALID VOTES - selecting first model by default")
            winner = selected_models[0]
        else:
            max_score = max(totals.values())
            contenders = [m for m, score in totals.items() if score == max_score]
            if len(contenders) > 1:
                winner = min(contenders, key=lambda m: selected_models.index(m))
                status_cb(f"Tie between {len(contenders)} models, selected by order: {short_id(winner)}")
            else:
                winner = contenders[0]

        tally = totals
        details = {
            "question": question,
            "answers": answers,
            "valid_votes": valid_votes,
            "invalid_votes": invalid_votes,
            "vote_messages": vote_messages,
            "tally": tally,
            "errors": errors,
            "winner": winner,
            "participation_rate": len(valid_votes) / max(1, len(voters_to_use)),
            "voters_used": voters_to_use,
        }

        status_cb(f"Vote complete. Valid: {len(valid_votes)}/{len(voters_to_use)}")
        return answers, winner, details, tally

# -----------------------
# Qt GUI
# -----------------------
class ModelFetchWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url

    @QtCore.Slot()
    def run(self):
        try:
            models = fetch_models_from_lmstudio(self.base_url)
            self.finished.emit(models)
        except Exception as e:
            self.failed.emit(str(e))


class CouncilWindow(QtWidgets.QMainWindow):
    status_signal = QtCore.Signal(str)
    result_signal = QtCore.Signal(object)   # (question, answers, winner, details, tally)
    error_signal  = QtCore.Signal(str)
    log_signal    = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolyCouncil")
        self.resize(1320, 900)
        self.app_icon = create_app_icon()
        self.setWindowIcon(self.app_icon)

        self.use_roles = False
        self.debug_enabled = False
        self._model_thread: Optional[QtCore.QThread] = None
        self._model_worker: Optional[ModelFetchWorker] = None
        self.log_history_limit = 500

        if qdarktheme:
            try:
                qdarktheme.setup_theme()
            except Exception:
                pass

        ensure_db()

        self.models: List[str] = []
        self.model_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.model_tabs: Dict[str, QtWidgets.QWidget] = {}
        self.model_texts: Dict[str, QtWidgets.QPlainTextEdit] = {}
        self.model_persona_combos: Dict[str, QtWidgets.QComboBox] = {}

        self._build_ui()
        self._setup_log_dock()
        self._connect_signals()
        set_log_sink(self._log_sink_dispatch)

        # restore settings
        s = load_settings()
        self.base_edit.setText(s.get("base_url", "http://localhost:1234"))

        self.personas = self._merge_persona_library(s.get("personas", []))
        self.persona_assignments: Dict[str, str] = dict(s.get("persona_assignments", {}) or {})
        self._cleanup_persona_assignments()

        self.debug_enabled = bool(s.get("debug", False))
        self._apply_debug_setting(self.debug_enabled, persist=False, announce=True)

        self.single_voter_check.blockSignals(True)
        self.single_voter_check.setChecked(bool(s.get("single_voter_enabled", False)))
        self.single_voter_check.blockSignals(False)

        sel = s.get("single_voter_model", "") or ""
        if sel and self.single_voter_combo.findText(sel) < 0:
            self.single_voter_combo.addItem(sel)
        if sel:
            self.single_voter_combo.setCurrentText(sel)

        # Load roles_enabled setting (will be shown in settings dialog)
        roles_enabled = bool(s.get("roles_enabled", False))
        self.use_roles = roles_enabled
        # Update visibility if models are already loaded
        if hasattr(self, 'model_persona_combos') and self.model_persona_combos:
            self._update_persona_combo_visibility()

        self.conc_spin.blockSignals(True)
        saved_conc = int(s.get("max_concurrency", 1) or 1)
        self.conc_spin.setValue(max(1, min(saved_conc, self.conc_spin.maximum())))
        self.conc_spin.blockSignals(False)

        # initial connect
        QtCore.QTimer.singleShot(50, self._connect_base)
        self._refresh_leaderboard()

    # ----- UI -----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10,10,10,8)
        layout.setSpacing(8)
        self.setCentralWidget(central)

        # Navigation bar
        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(10)
        nav.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("PolyCouncil")
        t_font = title.font()
        t_font.setPointSize(t_font.pointSize() + 2)
        t_font.setBold(True)
        title.setFont(t_font)
        self.settings_btn = QtWidgets.QPushButton("Settings")
        nav.addWidget(title)
        nav.addStretch(1)
        nav.addWidget(self.settings_btn)
        layout.addLayout(nav)

        # Top Bar (classic look + added single-voter widgets)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)

        self.base_edit = QtWidgets.QLineEdit()
        self.base_edit.setPlaceholderText("http://localhost:1234")
        self.connect_btn = QtWidgets.QPushButton("Connect")

        # Single voter controls
        self.single_voter_check = QtWidgets.QCheckBox("Single-voter")
        self.single_voter_combo = QtWidgets.QComboBox()
        self.single_voter_combo.setMinimumWidth(220)

        # Concurrency
        self.conc_label = QtWidgets.QLabel("Max concurrent jobs:")
        self.conc_spin = QtWidgets.QSpinBox()
        self.conc_spin.setRange(1, 8)
        self.conc_spin.setValue(1)
        self.conc_warn = QtWidgets.QLabel("Warning: higher values can slow the app dramatically on modest hardware.")
        self.conc_warn.setWordWrap(True)
        self.conc_warn.setStyleSheet("color: #a12;")
        conc_row = QtWidgets.QHBoxLayout()
        conc_row.addWidget(self.conc_label)
        conc_row.addWidget(self.conc_spin)
        conc_box = QtWidgets.QVBoxLayout()
        conc_box.addLayout(conc_row)
        conc_box.addWidget(self.conc_warn)

        top.addWidget(QtWidgets.QLabel("LM Studio Base URL:"))
        top.addWidget(self.base_edit, stretch=0)
        top.addWidget(self.connect_btn)
        top.addWidget(self.single_voter_check)
        top.addWidget(self.single_voter_combo)
        top.addStretch(1)
        top.addLayout(conc_box)

        layout.addLayout(top)

        # Main Grid (classic)
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        layout.addLayout(grid, stretch=1)

        # Left pane: Leaderboard + Model list
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        lb_title = QtWidgets.QLabel("Leaderboard")
        lb_title.setStyleSheet("font-weight: 600;")
        self.leader_list = QtWidgets.QListWidget()
        self.reset_btn = QtWidgets.QPushButton("Reset Leaderboard")

        models_title = QtWidgets.QLabel("Models")
        models_title.setStyleSheet("font-weight: 600;")

        self.models_area = QtWidgets.QScrollArea()
        self.models_area.setWidgetResizable(True)
        self.models_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.models_inner = QtWidgets.QWidget()
        self.models_layout = QtWidgets.QVBoxLayout(self.models_inner)
        self.models_layout.setContentsMargins(6,6,6,6)
        self.models_layout.setSpacing(4)
        self.models_layout.addStretch(1)
        self.models_area.setWidget(self.models_inner)

        model_btn_row = QtWidgets.QHBoxLayout()
        self.refresh_models_btn = QtWidgets.QPushButton("Refresh Models")
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        model_btn_row.addWidget(self.refresh_models_btn)
        model_btn_row.addWidget(self.select_all_btn)
        model_btn_row.addWidget(self.clear_btn)

        left.addWidget(lb_title)
        left.addWidget(self.leader_list)
        left.addWidget(self.reset_btn)
        left.addSpacing(8)
        left.addWidget(models_title)
        left.addWidget(self.models_area, stretch=1)
        left.addLayout(model_btn_row)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(420)  # Increased to accommodate persona buttons
        grid.addWidget(left_widget, 0, 0)

        # Center: Chat
        center = QtWidgets.QVBoxLayout()
        center.setSpacing(8)

        chat_title = QtWidgets.QLabel("Chat")
        chat_title.setStyleSheet("font-weight: 600;")
        self.chat_view = QtWidgets.QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("Ask the council…")
        self.send_btn = QtWidgets.QPushButton("Send")

        entry_row = QtWidgets.QHBoxLayout()
        entry_row.addWidget(self.prompt_edit, stretch=1)
        entry_row.addWidget(self.send_btn)

        center.addWidget(chat_title)
        center.addWidget(self.chat_view, stretch=1)
        center.addLayout(entry_row)

        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(center)
        grid.addWidget(center_widget, 0, 1)

        # Right: Tabs
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)

        tabs_title = QtWidgets.QLabel("Per-Model Answers")
        tabs_title.setStyleSheet("font-weight: 600;")
        self.tabs = QtWidgets.QTabWidget()

        right.addWidget(tabs_title)
        right.addWidget(self.tabs, stretch=1)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        grid.addWidget(right_widget, 0, 2)

        # Status/footer
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0,0,0,0)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setTextVisible(False)
        self.busy.setMaximum(0)
        self.busy.setVisible(False)
        self.status_label = QtWidgets.QLabel("Ready.")
        self.footer_link = QtWidgets.QLabel('<a href="https://github.com/TrentPierce">Trent Pierce · GitHub</a>')
        self.footer_link.setOpenExternalLinks(True)

        bottom.addWidget(self.busy, stretch=0)
        bottom.addSpacing(8)
        bottom.addWidget(self.status_label, stretch=1)
        bottom.addWidget(self.footer_link, stretch=0)
        layout.addLayout(bottom)

    def _setup_log_dock(self):
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self.log_history_limit)
        self.log_dock = QtWidgets.QDockWidget("Debug Log", self)
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _connect_signals(self):
        self.connect_btn.clicked.connect(self._connect_base)
        self.refresh_models_btn.clicked.connect(self._refresh_models_clicked)
        self.select_all_btn.clicked.connect(self._select_all_models)
        self.clear_btn.clicked.connect(self._clear_models)
        self.reset_btn.clicked.connect(self._reset_leaderboard_clicked)
        self.send_btn.clicked.connect(self._send)
        self.prompt_edit.returnPressed.connect(self._send)
        self.conc_spin.valueChanged.connect(self._concurrency_changed)
        self.settings_btn.clicked.connect(self._open_settings_dialog)

        # single-voter signals (make sure message shows correct ON/OFF)
        self.single_voter_check.toggled.connect(self._single_voter_toggled_bool)
        self.single_voter_combo.currentTextChanged.connect(self._single_voter_changed)

        self.status_signal.connect(self._set_status)
        self.result_signal.connect(self._handle_result)
        self.error_signal.connect(self._handle_error)
        self.log_signal.connect(self._append_log)

    # ----- actions -----
    def _update_persona_combo_visibility(self):
        """Update visibility and enabled state of persona buttons - show and enable when personas are enabled"""
        if not hasattr(self, 'model_persona_combos') or not self.model_persona_combos:
            return
        
        # Show buttons only when personas are enabled
        enabled = bool(self.use_roles)
        for model, btn in list(self.model_persona_combos.items()):
            if not btn or not isinstance(btn, QtWidgets.QPushButton):
                continue
            # Show buttons when personas are enabled, hide when disabled
            btn.setVisible(enabled)
            btn.setEnabled(enabled)
            # Update button style
            if enabled:
                btn.setStyleSheet("")
            else:
                btn.setStyleSheet("QPushButton { color: #888; background-color: #333; }")
        
        # Force layout update
        if hasattr(self, 'models_inner') and self.models_inner:
            self.models_inner.updateGeometry()
            self.models_area.update()
    
    def _show_persona_menu(self, model_id: str, button: QtWidgets.QPushButton):
        """Show persona selection menu for a model"""
        if not self.use_roles:
            return
        # Create menu dynamically
        menu = QtWidgets.QMenu(self)
        current_assigned = self.persona_assignments.get(model_id, "None")
        persona_list = self._persona_names()
        for persona_name in persona_list:
            action = menu.addAction(persona_name)
            action.setCheckable(True)
            if persona_name == current_assigned:
                action.setChecked(True)
            # Use a closure that captures the values correctly
            def make_action_handler(pn, mid, btn):
                return lambda checked: self._select_persona_for_model(mid, pn, btn)
            action.triggered.connect(make_action_handler(persona_name, model_id, button))
        # Show menu below button
        button_pos = button.mapToGlobal(button.rect().bottomLeft())
        menu.exec(button_pos)
    
    def _select_persona_for_model(self, model: str, persona_name: str, button: QtWidgets.QPushButton):
        """Handle persona selection for a specific model"""
        if persona_name not in self._persona_names():
            persona_name = "None"
        self.persona_assignments[model] = persona_name
        
        # Update button text (truncate if too long)
        display_text = persona_name if persona_name != "None" else "Persona"
        if len(display_text) > 12:
            display_text = display_text[:10] + ".."
        button.setText(display_text)
        button.setToolTip(persona_name if persona_name != "None" else "Select persona")
        
        self._save_persona_state()
        
        # Update the menu to show the selected persona
        menu = button.menu()
        if menu:
            for action in menu.actions():
                action.setChecked(action.text() == persona_name)

    def _debug_toggled(self, checked: bool):
        self._apply_debug_setting(bool(checked), persist=True, announce=True)

    def _apply_debug_setting(self, enabled: bool, *, persist: bool, announce: bool):
        self.debug_enabled = bool(enabled)
        global DEBUG_VOTING
        DEBUG_VOTING = self.debug_enabled
        if persist:
            save_settings({"debug": self.debug_enabled})
        if hasattr(self, "log_dock"):
            if not self.debug_enabled:
                self.log_dock.hide()
            elif self.log_view.blockCount():
                self.log_dock.show()
        if announce:
            self._set_status(f"Debug logs: {'ON' if self.debug_enabled else 'OFF'}")

    def _merge_persona_library(self, stored: Iterable[dict]) -> List[dict]:
        library: Dict[str, dict] = {}
        for persona in DEFAULT_PERSONAS:
            library[persona["name"]] = dict(persona)
        for entry in stored or []:
            name = entry.get("name")
            if not name:
                continue
            entry_prompt = entry.get("prompt")
            entry_builtin = bool(entry.get("builtin", False))
            if name in library and library[name].get("builtin"):
                # keep builtin prompt for defaults
                continue
            library[name] = {
                "name": str(name),
                "prompt": entry_prompt if entry_prompt is not None else None,
                "builtin": entry_builtin,
            }

        personas = list(library.values())

        def sort_key(persona: dict) -> tuple[int, str]:
            if persona["name"] == "None":
                return (0, "")
            return (
                1 if persona.get("builtin", False) else 2,
                persona["name"].lower(),
            )

        personas.sort(key=sort_key)
        return personas

    def _sort_personas_inplace(self):
        def sort_key(persona: dict) -> tuple[int, str]:
            if persona["name"] == "None":
                return (0, "")
            return (
                1 if persona.get("builtin", False) else 2,
                persona["name"].lower(),
            )

        self.personas.sort(key=sort_key)

    def _persona_names(self) -> List[str]:
        return [persona["name"] for persona in self.personas]

    def _persona_prompt(self, name: str) -> Optional[str]:
        for persona in self.personas:
            if persona["name"] == name:
                return persona.get("prompt")
        return None

    def _persona_by_name(self, name: str) -> Optional[dict]:
        for persona in self.personas:
            if persona["name"] == name:
                return persona
        return None

    def _cleanup_persona_assignments(self):
        names = set(self._persona_names())
        dirty = False
        for model, persona_name in list(self.persona_assignments.items()):
            if persona_name not in names:
                self.persona_assignments[model] = "None"
                dirty = True
        if dirty:
            self._save_persona_state()

    def _save_persona_state(self):
        save_settings({
            "personas": self.personas,
            "persona_assignments": self.persona_assignments,
        })

    def _refresh_persona_combos(self):
        """Refresh persona button text for all model buttons"""
        names = self._persona_names()
        if not names:
            # Ensure at least "None" exists
            if not hasattr(self, 'personas') or not self.personas:
                self.personas = [{"name": "None", "prompt": None, "builtin": True}]
            names = ["None"]
        
        self._cleanup_persona_assignments()
        for model, btn in self.model_persona_combos.items():
            if not btn or not isinstance(btn, QtWidgets.QPushButton):
                continue
            assigned = self.persona_assignments.get(model, "None")
            if assigned not in names:
                assigned = "None"
            
            # Update button text
            display_text = assigned if assigned != "None" else "Persona"
            if len(display_text) > 12:
                display_text = display_text[:10] + ".."
            btn.setText(display_text)
            btn.setToolTip(assigned if assigned != "None" else "Select persona")

    def _persona_assigned(self, model: str, persona_name: str):
        if persona_name not in self._persona_names():
            persona_name = "None"
        self.persona_assignments[model] = persona_name
        self._save_persona_state()

    def _concurrency_changed(self, value: int):
        save_settings({"max_concurrency": int(value)})

    def _open_settings_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.resize(700, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Debug section
        debug_checkbox = QtWidgets.QCheckBox("Enable debug logs")
        debug_checkbox.setChecked(self.debug_enabled)
        layout.addWidget(debug_checkbox)
        
        # Enable personas section
        enable_personas_checkbox = QtWidgets.QCheckBox("Enable personas")
        enable_personas_checkbox.setChecked(self.use_roles)
        layout.addWidget(enable_personas_checkbox)

        # Persona management section
        persona_group = QtWidgets.QGroupBox("Persona Management")
        persona_layout = QtWidgets.QVBoxLayout(persona_group)
        persona_layout.setSpacing(8)

        # Persona list
        persona_list_label = QtWidgets.QLabel("Available Personas:")
        persona_layout.addWidget(persona_list_label)

        persona_list = QtWidgets.QListWidget()
        persona_list.setMaximumHeight(200)
        for persona in self.personas:
            item = QtWidgets.QListWidgetItem(persona["name"])
            if persona.get("builtin", False):
                item.setForeground(QtGui.QColor("#666"))
            persona_list.addItem(item)
        persona_layout.addWidget(persona_list)

        # Persona buttons
        persona_btn_layout = QtWidgets.QHBoxLayout()
        add_persona_btn = QtWidgets.QPushButton("Add Custom Persona")
        edit_persona_btn = QtWidgets.QPushButton("Edit Selected")
        delete_persona_btn = QtWidgets.QPushButton("Delete Selected")
        persona_btn_layout.addWidget(add_persona_btn)
        persona_btn_layout.addWidget(edit_persona_btn)
        persona_btn_layout.addWidget(delete_persona_btn)
        persona_layout.addLayout(persona_btn_layout)

        layout.addWidget(persona_group)

        # Report issue button
        report_btn = QtWidgets.QPushButton("Report an Issue")
        layout.addWidget(report_btn)

        layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)

        # Connect signals
        report_btn.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://github.com/TrentPierce/PolyCouncil/issues"))
        )
        debug_checkbox.toggled.connect(self._debug_toggled)
        
        def on_personas_toggled(checked):
            self.use_roles = checked
            save_settings({"roles_enabled": self.use_roles})
            self._update_persona_combo_visibility()
            QtWidgets.QApplication.processEvents()
        
        enable_personas_checkbox.toggled.connect(on_personas_toggled)
        buttons.rejected.connect(dialog.reject)

        def refresh_persona_list():
            persona_list.clear()
            for persona in self.personas:
                item = QtWidgets.QListWidgetItem(persona["name"])
                if persona.get("builtin", False):
                    item.setForeground(QtGui.QColor("#666"))
                persona_list.addItem(item)

        def get_persona_prompt(current_prompt: str = "") -> Optional[str]:
            prompt_dialog = QtWidgets.QDialog(dialog)
            prompt_dialog.setWindowTitle("Persona System Prompt")
            prompt_dialog.resize(500, 300)
            layout = QtWidgets.QVBoxLayout(prompt_dialog)
            layout.addWidget(QtWidgets.QLabel("Enter the system prompt for this persona:"))
            text_edit = QtWidgets.QPlainTextEdit()
            text_edit.setPlainText(current_prompt)
            layout.addWidget(text_edit)
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(prompt_dialog.accept)
            buttons.rejected.connect(prompt_dialog.reject)
            layout.addWidget(buttons)
            if prompt_dialog.exec() != QtWidgets.QDialog.Accepted:
                return None
            return text_edit.toPlainText().strip()

        def add_persona():
            name_dialog = QtWidgets.QInputDialog(dialog)
            name_dialog.setWindowTitle("New Persona")
            name_dialog.setLabelText("Persona Name:")
            name_dialog.setTextValue("")
            if name_dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            name = name_dialog.textValue().strip()
            if not name:
                return
            if name in self._persona_names():
                QtWidgets.QMessageBox.warning(dialog, "Duplicate Name", f"Persona '{name}' already exists.")
                return

            prompt = get_persona_prompt()
            if prompt is None:
                return

            self.personas.append({
                "name": name,
                "prompt": prompt if prompt else None,
                "builtin": False,
            })
            self._sort_personas_inplace()
            self._save_persona_state()
            self._refresh_persona_combos()
            refresh_persona_list()

        def edit_persona():
            current = persona_list.currentItem()
            if not current:
                QtWidgets.QMessageBox.information(dialog, "No Selection", "Please select a persona to edit.")
                return
            name = current.text()
            persona = self._persona_by_name(name)
            if not persona:
                return
            if persona.get("builtin", False):
                QtWidgets.QMessageBox.information(dialog, "Built-in Persona", "Built-in personas cannot be edited. Create a custom persona instead.")
                return

            name_dialog = QtWidgets.QInputDialog(dialog)
            name_dialog.setWindowTitle("Edit Persona Name")
            name_dialog.setLabelText("Persona Name:")
            name_dialog.setTextValue(name)
            if name_dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            new_name = name_dialog.textValue().strip()
            if not new_name:
                return
            if new_name != name and new_name in self._persona_names():
                QtWidgets.QMessageBox.warning(dialog, "Duplicate Name", f"Persona '{new_name}' already exists.")
                return

            prompt = get_persona_prompt(persona.get("prompt") or "")
            if prompt is None:
                return

            persona["name"] = new_name
            persona["prompt"] = prompt if prompt else None
            if name != new_name:
                for model, assigned in list(self.persona_assignments.items()):
                    if assigned == name:
                        self.persona_assignments[model] = new_name
            self._sort_personas_inplace()
            self._save_persona_state()
            self._refresh_persona_combos()
            refresh_persona_list()

        def delete_persona():
            current = persona_list.currentItem()
            if not current:
                QtWidgets.QMessageBox.information(dialog, "No Selection", "Please select a persona to delete.")
                return
            name = current.text()
            persona = self._persona_by_name(name)
            if not persona:
                return
            if persona.get("builtin", False):
                QtWidgets.QMessageBox.information(dialog, "Built-in Persona", "Built-in personas cannot be deleted.")
                return
            if name == "None":
                QtWidgets.QMessageBox.information(dialog, "Cannot Delete", "The 'None' persona cannot be deleted.")
                return

            reply = QtWidgets.QMessageBox.question(
                dialog, "Delete Persona", f"Are you sure you want to delete '{name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

            self.personas = [p for p in self.personas if p["name"] != name]
            for model, assigned in list(self.persona_assignments.items()):
                if assigned == name:
                    self.persona_assignments[model] = "None"
            self._save_persona_state()
            self._refresh_persona_combos()
            refresh_persona_list()

        add_persona_btn.clicked.connect(add_persona)
        edit_persona_btn.clicked.connect(edit_persona)
        delete_persona_btn.clicked.connect(delete_persona)

        dialog.exec()

    def _single_voter_toggled_state(self, state: int):
        # 'state' is a Qt.CheckState enum; compare directly
        try:
            enabled = (state == QtCore.Qt.CheckState.Checked)
        except Exception:
            enabled = bool(state)
        save_settings({"single_voter_enabled": enabled, "single_voter_model": self.single_voter_combo.currentText()})
        self._set_status("Single-voter mode: ON" if enabled else "Single-voter mode: OFF")

    def _single_voter_toggled_bool(self, checked: bool):
        save_settings({"single_voter_enabled": bool(checked), "single_voter_model": self.single_voter_combo.currentText()})
        self._set_status("Single-voter mode: ON" if checked else "Single-voter mode: OFF")

    def _single_voter_changed(self, text: str):
        save_settings({"single_voter_model": text})

    def _connect_base(self):
        base = self.base_edit.text().strip() or "http://localhost:1234"
        save_settings({"base_url": base})
        self._set_status(f"Connecting to {base} …")
        self._busy(True)
        QtCore.QTimer.singleShot(50, self._refresh_models)

    def _refresh_models_clicked(self):
        self._set_status("Refreshing models …")
        self._busy(True)
        QtCore.QTimer.singleShot(50, self._refresh_models)

    def _refresh_models(self):
        base = self.base_edit.text().strip() or "http://localhost:1234"
        if self._model_thread and self._model_thread.isRunning():
            return

        self._model_worker = ModelFetchWorker(base)
        self._model_thread = QtCore.QThread(self)
        self._model_worker.moveToThread(self._model_thread)

        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._models_fetched)
        self._model_worker.failed.connect(self._models_fetch_failed)

        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.failed.connect(self._model_thread.quit)
        self._model_thread.finished.connect(self._model_thread.deleteLater)
        self._model_worker.finished.connect(self._model_worker.deleteLater)
        self._model_worker.failed.connect(self._model_worker.deleteLater)

        self._model_thread.start()

    def _models_fetched(self, models: List[str]):
        self._model_thread = None
        self._model_worker = None
        self.models = models

        self._populate_models()

        self.single_voter_combo.blockSignals(True)
        self.single_voter_combo.clear()
        for m in self.models:
            self.single_voter_combo.addItem(m)
        last = load_settings().get("single_voter_model", "")
        if last:
            ix = self.single_voter_combo.findText(last)
            if ix >= 0:
                self.single_voter_combo.setCurrentIndex(ix)
        self.single_voter_combo.blockSignals(False)

        self._busy(False)
        if not self.models:
            self._set_status("No models found. Check base URL or LM Studio.")
        else:
            self._set_status(f"Found {len(self.models)} models.")

    def _models_fetch_failed(self, error: str):
        self._model_thread = None
        self._model_worker = None
        self._busy(False)
        self._set_status(f"Model refresh failed: {error}")

    def _populate_models(self):
        # clear UI - remove ALL items including stretches
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.spacerItem():
                # Remove spacer items too
                pass
            del item
        self.model_checks.clear()
        self.model_persona_combos.clear()
        # Add stretch at the end
        self.models_layout.addStretch(1)

        # Ensure personas are initialized
        if not hasattr(self, 'personas') or not self.personas:
            s = load_settings()
            self.personas = self._merge_persona_library(s.get("personas", []))
            if not hasattr(self, 'persona_assignments'):
                self.persona_assignments = dict(s.get("persona_assignments", {}) or {})

        persona_names = self._persona_names()
        if not persona_names:
            # Fallback: ensure at least "None" exists
            self.personas = [{"name": "None", "prompt": None, "builtin": True}]
            persona_names = ["None"]

        for m in self.models:
            row_widget = QtWidgets.QWidget()
            row_widget.setMinimumHeight(28)  # Ensure enough height for button
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(8)

            cb = QtWidgets.QCheckBox(m)
            cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            
            # Create a button for persona selection instead of dropdown
            persona_btn = QtWidgets.QPushButton("Persona")
            persona_btn.setFixedWidth(110)  # Fixed width to prevent layout issues
            persona_btn.setFixedHeight(24)  # Fixed height to match checkbox
            persona_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            
            assigned = self.persona_assignments.get(m, "None")
            if assigned not in persona_names:
                assigned = "None"
            
            # Update button text to show current persona (truncate if too long)
            display_text = assigned if assigned != "None" else "Persona"
            if len(display_text) > 12:
                display_text = display_text[:10] + ".."
            persona_btn.setText(display_text)
            persona_btn.setToolTip(assigned if assigned != "None" else "Select persona")
            
            # Show buttons only when personas are enabled
            persona_btn.setVisible(self.use_roles)
            persona_btn.setEnabled(self.use_roles)
            # Ensure button accepts mouse events and is clickable
            persona_btn.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
            persona_btn.setFocusPolicy(QtCore.Qt.StrongFocus)
            persona_btn.setAutoDefault(False)
            persona_btn.setDefault(False)
            persona_btn.raise_()  # Ensure button is on top
            
            # Connect button click to show persona menu - use direct lambda with explicit capture
            # Capture model_id and button in the lambda closure
            model_id_capture = m  # Capture in outer scope
            button_capture = persona_btn  # Capture in outer scope
            persona_btn.clicked.connect(
                lambda checked=False, mid=model_id_capture, btn=button_capture: self._show_persona_menu(mid, btn)
            )
            
            # Add widgets to layout - checkbox gets remaining space, button gets fixed width
            row_layout.addWidget(cb, stretch=1)
            row_layout.addWidget(persona_btn, stretch=0)
            row_layout.setAlignment(persona_btn, QtCore.Qt.AlignRight)

            self.models_layout.insertWidget(self.models_layout.count() - 1, row_widget)
            self.model_checks[m] = cb
            self.model_persona_combos[m] = persona_btn  # Store button instead of combo

        # Ensure visibility is correct after all buttons are created
        self._update_persona_combo_visibility()

    def _select_all_models(self):
        for cb in self.model_checks.values():
            cb.setChecked(True)

    def _clear_models(self):
        for cb in self.model_checks.values():
            cb.setChecked(False)

    def _reset_leaderboard_clicked(self):
        try:
            if DB_PATH.exists():
                DB_PATH.unlink()
            ensure_db()
            self._refresh_leaderboard()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reset Leaderboard", str(e))

    def _refresh_leaderboard(self):
        self.leader_list.clear()
        for mid, count in load_leaderboard():
            self.leader_list.addItem(f"{short_id(mid)} — {count} wins")

    # ----- UI helpers -----
    def _prepare_tabs(self, selected_models: List[str]):
        # remove any stale tabs except results
        keep = set(selected_models)
        for mid in list(self.model_tabs.keys()):
            if mid == "_RESULTS_":
                continue
            if mid not in keep:
                idx = self.tabs.indexOf(self.model_tabs[mid])
                if idx >= 0:
                    self.tabs.removeTab(idx)
                self.model_tabs.pop(mid, None)
                self.model_texts.pop(mid, None)

        for m in selected_models:
            if m not in self.model_tabs:
                page = QtWidgets.QWidget()
                v = QtWidgets.QVBoxLayout(page)
                txt = QtWidgets.QPlainTextEdit()
                txt.setReadOnly(True)
                v.addWidget(txt)
                self.tabs.addTab(page, short_id(m))
                self.model_tabs[m] = page
                self.model_texts[m] = txt
            self.model_texts[m].setPlainText("Thinking …")

        # Results tab
        if "_RESULTS_" not in self.model_tabs:
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            txt = QtWidgets.QPlainTextEdit()
            txt.setReadOnly(True)
            v.addWidget(txt)
            self.tabs.addTab(page, "Results")
            self.model_tabs["_RESULTS_"] = page
            self.model_texts["_RESULTS_"] = txt
        self.model_texts["_RESULTS_"].setPlainText("")

    def _append_chat(self, text: str):
        self.chat_view.appendPlainText(text)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def _append_log(self, text: str):
        if not text:
            return
        if self.debug_enabled and not self.log_dock.isVisible():
            self.log_dock.show()
        self.log_view.appendPlainText(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _log_sink_dispatch(self, label: str, message: str):
        if not message:
            return
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        formatted = f"{stamp} [{label}] {message}"
        self.log_signal.emit(formatted)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _busy(self, on: bool):
        self.busy.setVisible(on)
        self.busy.setMaximum(0 if on else 1)

    def _send(self):
        base = self.base_edit.text().strip() or "http://localhost:1234"
        selected = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            self._set_status("Select at least one model.")
            return
        question = self.prompt_edit.text().strip()
        if not question:
            self._set_status("Ask something first.")
            return

        # Use self.use_roles instead of removed roles_check
        if self.use_roles:
            roles_map = {}
            for m in selected:
                # Get persona from assignments (buttons don't have currentText)
                persona_name = self.persona_assignments.get(m, "None")
                if persona_name not in self._persona_names():
                    persona_name = "None"
                persona_prompt = self._persona_prompt(persona_name)
                roles_map[m] = persona_prompt
                self.persona_assignments[m] = persona_name
            self._save_persona_state()
        else:
            roles_map = {m: None for m in selected}

        # Prepare UI
        self._prepare_tabs(selected)
        self._append_chat(f"You: {question}")
        self.prompt_edit.clear()
        self._set_status("Starting council …")

        conc =  int(self.conc_spin.value())
        single_voter_enabled = self.single_voter_check.isChecked()
        single_voter_model   = self.single_voter_combo.currentText().strip()

        def worker():
            try:
                voters = [single_voter_model] if (single_voter_enabled and single_voter_model) else None
                answers, winner, details, tally = asyncio.run(
                    council_round(
                        base, selected, question, roles_map, self.status_signal.emit,
                        max_concurrency=conc, voter_override=voters
                    )
                )
                # record leaderboard
                try:
                    record_vote(question, winner, details)
                except Exception:
                    pass
                self.result_signal.emit((question, answers, winner, details, tally))
            except Exception as e:
                self.error_signal.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()
        self._busy(True)

    def _handle_error(self, msg: str):
        self._busy(False)
        self._set_status(f"Error: {msg}")
        self._append_chat(f"[Error] {msg}")

    def _handle_result(self, payload: object):
        self._busy(False)
        question, answers, winner, details, tally = payload

        # Fill each model tab with its answer
        for mid, txtw in self.model_texts.items():
            if mid == "_RESULTS_":
                continue
            txtw.setPlainText(answers.get(mid, ""))

        # Refresh leaderboard list
        self._refresh_leaderboard()

        # Compose results summary
        valid_votes = details.get("valid_votes", {})
        invalid_votes = details.get("invalid_votes", {})
        vote_messages = details.get("vote_messages", {})
        voters_used = details.get("voters_used", [])

        lines = []
        lines.append(f"Winner: {short_id(winner)}  (score={tally.get(winner, 0)})")
        lines.append("")
        lines.append("Totals:")
        for m in sorted(tally.keys(), key=lambda k: (-tally[k], k)):
            lines.append(f"  {short_id(m):28} → {tally[m]}")

        lines.append("")
        lines.append(f"Voters used: {', '.join(short_id(v) for v in voters_used)}")
        lines.append(f"Valid ballots: {len(valid_votes)}; Invalid: {len(invalid_votes)}")

        if valid_votes:
            lines.append("")
            lines.append("Per-candidate received scores (weighted sums):")
            for candidate in answers.keys():
                received = []
                for voter, ballot in valid_votes.items():
                    sc = ballot["scores"].get(candidate)
                    if sc:
                        weighted = sum(sc[k] * RUBRIC_WEIGHTS[k] for k in RUBRIC_WEIGHTS.keys())
                        received.append(f"{short_id(voter)}: {sc} → {weighted}")
                if received:
                    lines.append(f"- {short_id(candidate)}:\n  " + "\n  ".join(received))

        if vote_messages:
            lines.append("")
            lines.append("Ballot notes:")
            for voter, msg in vote_messages.items():
                lines.append(f"  {short_id(voter)} → {msg}")

        self.model_texts["_RESULTS_"].setPlainText("\n".join(lines))

        # Show winning answer in chat
        ans = answers.get(winner, "")
        self._append_chat(f"Council → {short_id(winner)}:\n{ans}")
        self._set_status("Done.")

# -----------------------
# Entry point
# -----------------------
def main():
    import sys
    import platform
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("PolyCouncil")
    app.setApplicationDisplayName("PolyCouncil")
    
    # Create and set icon
    icon = create_app_icon()
    app.setWindowIcon(icon)
    
    # On Windows, also set the taskbar icon
    if platform.system() == "Windows":
        try:
            import ctypes
            # Get the app user model ID for proper taskbar grouping
            myappid = 'TrentPierce.PolyCouncil.1.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
    
    w = CouncilWindow()
    w.setWindowIcon(icon)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
