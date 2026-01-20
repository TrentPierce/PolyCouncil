
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
import uuid
import markdown  # type: ignore
import base64
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Any, Iterable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# Import new core modules
try:
    from core.tool_manager import FileParser, ModelCapabilityDetector
    from core.discussion_manager import DiscussionManager, CORE_SYSTEM_INSTRUCTION
except ImportError:
    # Fallback if modules not found
    FileParser = None
    ModelCapabilityDetector = None
    DiscussionManager = None
    CORE_SYSTEM_INSTRUCTION = ""

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
    {"name": "Structured Data Analyst", "prompt": "You are a data analyst. Structure answers into bullets, highlight assumptions and limits.", "builtin": True},
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
DEFAULT_PERSONAS_PATH = APP_DIR / "config" / "default_personas.json"
USER_PERSONAS_PATH = APP_DIR / "config" / "user_personas.json"

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

# Thread lock for settings to prevent race conditions
_settings_lock = threading.Lock()

def save_settings(s: dict):
    """Thread-safe settings persistence with proper error handling."""
    with _settings_lock:
        try:
            current = {}
            if SETTINGS_PATH.exists():
                try:
                    current = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    print(f"Warning: Settings file corrupted, starting fresh: {e}")
                    current = {}
            current.update(s or {})
            SETTINGS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
        except PermissionError as e:
            print(f"Error: Cannot write settings (permission denied): {e}")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

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
    timeout_sec: int = 120,
    images: List[str] = [],
    web_search: bool = False
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
    
    # Handle multimodal content
    if images:
        content_list = [{"type": "text", "text": user_prompt}]
        for img_b64 in images:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        messages.append({"role": "user", "content": content_list})
    else:
        messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if json_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

    # Inject Web Search Tools if enabled
    if web_search:
        payload["tools"] = [{
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }]

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
    voter_override: Optional[List[str]] = None,
    images: List[str] = [],
    web_search: bool = False,
    temperature: float = 0.7,
    is_cancelled: Optional[Callable[[], bool]] = None
):
    status_cb("Collecting answers…")
    async with aiohttp.ClientSession() as session:

        async def answer_one(model_id: str) -> tuple[str, str]:
            if is_cancelled and is_cancelled():
                return model_id, "[Cancelled]"
            
            user_prompt = question
            sys_p = roles.get(model_id) or None
            try:
                ans = await call_model(
                    session, base_url, model_id, user_prompt, 
                    temperature=temperature, sys_prompt=sys_p, 
                    images=images, web_search=web_search
                )
                if isinstance(ans, dict):
                    ans = json.dumps(ans, ensure_ascii=False)
                return model_id, ans
            except Exception as e1:
                return model_id, f"[ERROR fetching answer]\n{e1}"

        # Check cancellation before starting
        if is_cancelled and is_cancelled():
            raise RuntimeError("Process cancelled by user")

        pairs = await run_limited(max_concurrency, [lambda m=m: answer_one(m) for m in selected_models])
        
        # Check cancellation after answers
        if is_cancelled and is_cancelled():
            raise RuntimeError("Process cancelled by user")
            
        answers = {m: a for m, a in pairs}
        errors = {m: a for m, a in pairs if isinstance(a, str) and a.startswith("[ERROR")}

        status_cb("Models are voting…")

        voters_to_use = voter_override if voter_override else selected_models
        # Note: Voting phase does not use images or web search, typically.
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
            status_cb("⚠ NO VALID VOTES - selecting first model by default (consider checking model compatibility)")
            winner = selected_models[0]
        else:
            max_score = max(totals.values())
            contenders = [m for m, score in totals.items() if score == max_score]
            if len(contenders) > 1:
                # Use random selection instead of positional bias to ensure fairness
                winner = random.choice(contenders)
                status_cb(f"Tie between {len(contenders)} models, randomly selected: {short_id(winner)}")
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
    discussion_update_signal = QtCore.Signal(object)  # (entry_dict) for real-time updates
    capability_update_signal = QtCore.Signal()  # Signal to update capability UI

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
        
        # New features
        self.mode = "deliberation"  # "deliberation" or "discussion"
        self.uploaded_files: List[Path] = []
        self.model_capabilities: Dict[str, Dict[str, bool]] = {}  # model -> {web_search: bool, visual: bool}
        self.web_search_enabled = False

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
        
        # Mode selection
        mode_label = QtWidgets.QLabel("Mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Deliberation Mode", "Collaborative Discussion Mode"])
        self.mode_combo.setCurrentIndex(0)
        
        self.settings_btn = QtWidgets.QPushButton("Settings")
        nav.addWidget(title)
        nav.addSpacing(20)
        nav.addWidget(mode_label)
        nav.addWidget(self.mode_combo)
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
        self.conc_warn = QtWidgets.QLabel("⚠ Higher values may slow the app on modest hardware")
        self.conc_warn.setWordWrap(True)
        self.conc_warn.setStyleSheet("color: #a12;")
        self.conc_warn.setVisible(False)  # Only show when concurrency > 2
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

        self.lb_title = QtWidgets.QLabel("Leaderboard")
        self.lb_title.setStyleSheet("font-weight: 600;")
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

        left.addWidget(self.lb_title)
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

        # Center: Chat with file upload and tool controls
        center = QtWidgets.QVBoxLayout()
        center.setSpacing(8)

        chat_title = QtWidgets.QLabel("Chat")
        chat_title.setStyleSheet("font-weight: 600;")
        
        # File upload area
        file_group = QtWidgets.QGroupBox("File Upload")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        file_layout.setSpacing(4)
        
        file_btn_row = QtWidgets.QHBoxLayout()
        self.upload_btn = QtWidgets.QPushButton("Upload File")
        self.clear_files_btn = QtWidgets.QPushButton("Clear Files")
        file_btn_row.addWidget(self.upload_btn)
        file_btn_row.addWidget(self.clear_files_btn)
        file_btn_row.addStretch(1)
        
        self.files_list = QtWidgets.QListWidget()
        self.files_list.setMaximumHeight(80)
        self.files_list.setToolTip("Uploaded files will be parsed and included as context")
        
        # Visual/image support indicator (moved into file upload box)
        self.visual_status = QtWidgets.QLabel("Visual/Image Support: Not detected")
        self.visual_status.setStyleSheet("font-size: 10pt;")
        
        file_layout.addLayout(file_btn_row)
        file_layout.addWidget(self.files_list)
        file_layout.addWidget(self.visual_status)
        
        # Tool status indicators (web search only now)
        tool_group = QtWidgets.QGroupBox("Model Capabilities")
        tool_layout = QtWidgets.QVBoxLayout(tool_group)
        tool_layout.setSpacing(4)
        
        self.web_search_check = QtWidgets.QCheckBox("Enable Web Search")
        self.web_search_check.setEnabled(False)
        self.web_search_check.setToolTip("Enable if your model supports web search tools")
        
        tool_layout.addWidget(self.web_search_check)
        
        # Advanced parameters
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QHBoxLayout(param_group)
        
        param_layout.addWidget(QtWidgets.QLabel("Temp:"))
        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)  # 0.7 default
        self.temp_value_label = QtWidgets.QLabel("0.7")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_value_label.setText(f"{v/100:.1f}"))
        
        param_layout.addWidget(self.temp_slider)
        param_layout.addWidget(self.temp_value_label)
        
        center.addWidget(chat_title)
        center.addWidget(file_group)
        center.addWidget(tool_group)
        center.addWidget(param_group)
        
        # Chat view - switch to QTextBrowser for HTML/Markdown
        self.chat_view = QtWidgets.QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setReadOnly(True)
        
        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("Ask the council…")
        self.send_btn = QtWidgets.QPushButton("Send")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #ffcccc; color: black;")

        entry_row = QtWidgets.QHBoxLayout()
        entry_row.addWidget(self.prompt_edit, stretch=1)
        entry_row.addWidget(self.send_btn)
        entry_row.addWidget(self.stop_btn)

        center.addWidget(self.chat_view, stretch=1)
        center.addLayout(entry_row)

        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(center)
        grid.addWidget(center_widget, 0, 1)

        # Right: Tabs
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)

        self.tabs_title = QtWidgets.QLabel("Per-Model Answers")
        self.tabs_title.setStyleSheet("font-weight: 600;")
        self.tabs = QtWidgets.QTabWidget()

        right.addWidget(self.tabs_title)
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
        self.stop_btn.clicked.connect(self._stop_process)
        self.conc_spin.valueChanged.connect(self._concurrency_changed)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        
        # New signal connections
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.upload_btn.clicked.connect(self._upload_file)
        self.clear_files_btn.clicked.connect(self._clear_files)
        self.web_search_check.toggled.connect(self._web_search_toggled)

        # single-voter signals (make sure message shows correct ON/OFF)
        self.single_voter_check.toggled.connect(self._single_voter_toggled_bool)
        self.single_voter_combo.currentTextChanged.connect(self._single_voter_changed)

        self.status_signal.connect(self._set_status)
        self.result_signal.connect(self._handle_result)
        self.error_signal.connect(self._handle_error)
        self.log_signal.connect(self._append_log)
        self.discussion_update_signal.connect(self._update_discussion_view)
        self.capability_update_signal.connect(self._update_capability_ui)
        
        # Keyboard shortcuts
        self._setup_keyboard_shortcuts()
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions."""
        # Ctrl+Enter to send (alternative to Enter in prompt)
        send_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        send_shortcut.activated.connect(self._send)
        
        # Ctrl+Shift+A to select all models (Ctrl+A is used by text fields)
        select_all_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+A"), self)
        select_all_shortcut.activated.connect(self._select_all_models)
        
        # Ctrl+R to refresh models
        refresh_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        refresh_shortcut.activated.connect(self._refresh_models_clicked)
        
        # Escape to stop current operation
        stop_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        stop_shortcut.activated.connect(self._stop_process)

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
        
        # Load default personas from config/default_personas.json
        if DEFAULT_PERSONAS_PATH.exists():
            try:
                with open(DEFAULT_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                    default_personas = json.load(f)
                    for persona in default_personas:
                        name = persona.get("name")
                        if name:
                            library[name] = {
                                "name": name,
                                "prompt": persona.get("prompt_instruction"),
                                "builtin": True,
                            }
            except Exception as e:
                print(f"Error loading default personas: {e}")
        
        # Load user personas from config/user_personas.json
        if USER_PERSONAS_PATH.exists():
            try:
                with open(USER_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                    user_personas = json.load(f)
                    for persona in user_personas:
                        name = persona.get("name")
                        if name:
                            library[name] = {
                                "name": name,
                                "prompt": persona.get("prompt_instruction"),
                                "builtin": False,
                            }
            except Exception as e:
                print(f"Error loading user personas: {e}")
        
        # Also merge from legacy DEFAULT_PERSONAS
        for persona in DEFAULT_PERSONAS:
            name = persona.get("name")
            if name and name not in library:
                library[name] = dict(persona)
        
        # Merge from stored (legacy council_settings.json personas)
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
        # Show warning only when concurrency is high enough to potentially cause issues
        self.conc_warn.setVisible(value > 2)

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

            # Generate unique ID for user persona
            persona_id = f"u_{uuid.uuid4().hex[:8]}"
            
            # Add to local personas list
            self.personas.append({
                "name": name,
                "prompt": prompt if prompt else None,
                "builtin": False,
            })
            
            # Save to user_personas.json
            try:
                if USER_PERSONAS_PATH.exists():
                    with open(USER_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                        user_personas = json.load(f)
                else:
                    user_personas = []
                
                user_personas.append({
                    "id": persona_id,
                    "name": name,
                    "prompt_instruction": prompt if prompt else ""
                })
                
                USER_PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(USER_PERSONAS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(user_personas, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving user persona: {e}")
            
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
            
            # Update user_personas.json
            try:
                if USER_PERSONAS_PATH.exists():
                    with open(USER_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                        user_personas = json.load(f)
                    for p in user_personas:
                        if p.get("name") == name:
                            p["name"] = new_name
                            p["prompt_instruction"] = prompt if prompt else ""
                            break
                    with open(USER_PERSONAS_PATH, 'w', encoding='utf-8') as f:
                        json.dump(user_personas, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error updating user persona: {e}")
            
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

            # Remove from local list
            self.personas = [p for p in self.personas if p["name"] != name]
            
            # Remove from user_personas.json
            try:
                if USER_PERSONAS_PATH.exists():
                    with open(USER_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                        user_personas = json.load(f)
                    user_personas = [p for p in user_personas if p.get("name") != name]
                    with open(USER_PERSONAS_PATH, 'w', encoding='utf-8') as f:
                        json.dump(user_personas, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error deleting user persona: {e}")
            
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

        # Detect model capabilities
        if ModelCapabilityDetector and models:
            self._detect_model_capabilities()

        self._busy(False)
        if not self.models:
            self._set_status("No models found. Check base URL or LM Studio.")
        else:
            self._set_status(f"Found {len(self.models)} models.")
    
    def _detect_model_capabilities(self):
        """Detect capabilities for loaded models."""
        if not ModelCapabilityDetector:
            return
        
        base = self.base_edit.text().strip() or "http://localhost:1234"
        
        async def detect_all():
            async with aiohttp.ClientSession() as session:
                # Check all models, not just first 3
                for model in self.models:
                    try:
                        # Quick check: if model name contains "vl" or "VL", it likely supports visual
                        model_lower = model.lower()
                        has_vl = "vl" in model_lower
                        
                        # Use API detection, but also check model name
                        web_search = await ModelCapabilityDetector.detect_web_search(session, base, model)
                        visual_api = await ModelCapabilityDetector.detect_visual(session, base, model)
                        visual = visual_api or has_vl  # Combine API detection with name check
                        
                        self.model_capabilities[model] = {
                            "web_search": web_search,
                            "visual": visual
                        }
                    except Exception:
                        # Fallback: check model name for "vl" pattern
                        model_lower = model.lower()
                        has_vl = "vl" in model_lower
                        self.model_capabilities[model] = {
                            "web_search": False,
                            "visual": has_vl
                        }
        
        # Run detection in background
        def worker():
            try:
                asyncio.run(detect_all())
                # Update UI using signal (thread-safe)
                self.status_signal.emit("Capability detection complete")
                # Emit signal to update UI on main thread
                self.capability_update_signal.emit()
            except Exception:
                pass
        
        threading.Thread(target=worker, daemon=True).start()
    
    @QtCore.Slot()
    def _update_capability_ui(self):
        """Update UI to reflect detected capabilities."""
        # Use signal to update from background thread
        self.status_signal.emit("Updating capability indicators...")
        
        # Check selected models for capabilities
        selected_models = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        
        # Check if any SELECTED model supports visual
        has_visual = False
        if selected_models:
            has_visual = any(
                self.model_capabilities.get(model, {}).get("visual", False) 
                for model in selected_models
            )
            # Also check model names for "vl" pattern as fallback
            if not has_visual:
                has_visual = any("vl" in model.lower() for model in selected_models)
        else:
            # If no models selected, check all models
            has_visual = any(caps.get("visual", False) for caps in self.model_capabilities.values())
            if not has_visual:
                has_visual = any("vl" in model.lower() for model in self.models)
        
        if has_visual:
            self.visual_status.setText("Visual/Image Support: ✓ Detected")
            self.visual_status.setStyleSheet("color: green;")
        else:
            self.visual_status.setText("Visual/Image Support: Not detected")
            self.visual_status.setStyleSheet("")
        
        # Check if any SELECTED model supports web search
        has_web_search = False
        if selected_models:
            has_web_search = any(
                self.model_capabilities.get(model, {}).get("web_search", False) 
                for model in selected_models
            )
        else:
            has_web_search = any(caps.get("web_search", False) for caps in self.model_capabilities.values())
        
        if has_web_search:
            self.web_search_check.setEnabled(True)
            self.web_search_check.setToolTip("Web search capability detected. Enable to use.")
        else:
            self.web_search_check.setEnabled(False)
            self.web_search_check.setToolTip("No web search capability detected.")
        
        # Update file upload button based on visual support
        self._update_file_upload_capabilities(has_visual)

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
            
            # Connect model selection change to update capabilities
            cb.toggled.connect(self._on_model_selection_changed)

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
        leaderboard = load_leaderboard()
        total_wins = sum(count for _, count in leaderboard)
        for mid, count in leaderboard:
            pct = (count / total_wins * 100) if total_wins > 0 else 0
            self.leader_list.addItem(f"{short_id(mid)} — {count} wins ({pct:.0f}%)")

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
        # Convert simple text to HTML/Markdown for QTextBrowser
        if not text:
            return
        
        # Simple styling
        style = "margin: 5px; padding: 8px; border-radius: 8px;"
        
        if text.startswith("You:"):
            # User bubble: Aligned right (simulated), distinct color
            style += "background-color: #e6f3ff; color: #000000;"
            formatted_text = text.replace("You:", "<b>You:</b>")
        elif "→" in text:
            # System/Council bubble
            style += "background-color: #f0f0f0; color: #000000;"
            parts = text.split(":", 1)
            if len(parts) == 2:
                formatted_text = f"<b>{parts[0]}:</b>{parts[1]}"
            else:
                formatted_text = text
        elif text.startswith("<i>"):
            # Info message
            style += "background-color: transparent; color: #666666;"
            formatted_text = text
        else:
            # Generic
            style += "background-color: #ffffff; border: 1px solid #ddd; color: #000000;"
            formatted_text = text
            
        # Convert content to markdown (excluding the headers if possible, but mixed is tricky)
        # We will just markdown the message part if we split it, but for simplicity:
        html_content = markdown.markdown(formatted_text)
        
        full_html = f"<div style='{style}'>{html_content}</div>"
        self.chat_view.append(full_html)
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
        if hasattr(self, 'stop_btn'):
            self.stop_btn.setEnabled(on)
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(not on)

    def _mode_changed(self, index: int):
        """Handle mode selection change."""
        self.mode = "discussion" if index == 1 else "deliberation"
        # Update UI based on mode
        if self.mode == "discussion":
            self.tabs_title.setText("Discussion View")
            # Hide single voter controls in discussion mode
            self.single_voter_check.setVisible(False)
            self.single_voter_combo.setVisible(False)
            # Hide leaderboard in discussion mode
            self.lb_title.setVisible(False)
            self.leader_list.setVisible(False)
            self.reset_btn.setVisible(False)
        else:
            self.tabs_title.setText("Per-Model Answers")
            self.single_voter_check.setVisible(True)
            self.single_voter_combo.setVisible(True)
            # Show leaderboard in deliberation mode
            self.lb_title.setVisible(True)
            self.leader_list.setVisible(True)
            self.reset_btn.setVisible(True)
    
    def _update_file_upload_capabilities(self, has_visual: bool):
        """Update file upload button and tooltip based on visual support."""
        if has_visual:
            self.upload_btn.setToolTip("Upload documents or images (PDF, TXT, DOCX, JPG, PNG, etc.)")
        else:
            self.upload_btn.setToolTip("Upload documents only (PDF, TXT, DOCX) - No visual models selected")
    
    def _upload_file(self):
        """Handle file upload."""
        if FileParser is None:
            QtWidgets.QMessageBox.warning(self, "File Parsing Unavailable", 
                "File parsing libraries not installed. Install PyPDF2 and python-docx.")
            return
        
        # Check if any selected model supports visual
        selected_models = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        has_visual = False
        if selected_models:
            has_visual = any(
                self.model_capabilities.get(model, {}).get("visual", False) 
                for model in selected_models
            )
            # Also check model names for "vl" pattern
            if not has_visual:
                has_visual = any("vl" in model.lower() for model in selected_models)
        
        # Build file filter based on visual support
        if has_visual:
            file_filter = (
                "All Supported (*.txt *.pdf *.docx *.doc *.jpg *.jpeg *.png *.gif *.webp);;"
                "Documents (*.txt *.pdf *.docx *.doc);;"
                "Images (*.jpg *.jpeg *.png *.gif *.webp);;"
                "Text Files (*.txt);;PDF Files (*.pdf);;Word Documents (*.docx *.doc)"
            )
        else:
            file_filter = (
                "Documents (*.txt *.pdf *.docx *.doc);;"
                "Text Files (*.txt);;PDF Files (*.pdf);;Word Documents (*.docx *.doc)"
            )
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", "", file_filter
        )
        
        if file_path:
            path = Path(file_path)
            
            # Check if it's an image file
            is_image = path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            
            if is_image and not has_visual:
                QtWidgets.QMessageBox.warning(
                    self, "Image Upload Not Available",
                    "No visual models are currently selected. Please select a model with visual capabilities (name contains 'vl' or 'VL') to upload images."
                )
                return
            
            if path not in self.uploaded_files:
                self.uploaded_files.append(path)
                self.files_list.addItem(path.name)
                file_type = "image" if is_image else "document"
                self._set_status(f"{file_type.capitalize()} uploaded: {path.name}")
    
    def _on_model_selection_changed(self):
        """Update UI when model selection changes."""
        # Update capability indicators based on selected models
        self._update_capability_ui()
    
    def _clear_files(self):
        """Clear uploaded files."""
        self.uploaded_files.clear()
        self.files_list.clear()
        self._set_status("Files cleared")
    
    def _web_search_toggled(self, checked: bool):
        """Handle web search toggle."""
        self.web_search_enabled = checked
    
    def _stop_process(self):
        """Handle stop button click."""
        self._stop_requested = True
        self._set_status("Stopping...")
        self._busy(False)
        self._append_chat("[Stopped] Process cancelled by user.")
        self._set_status("Stopped.")
    
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

        # Parse uploaded files for context and images
        context_block = ""
        images_b64 = []
        
        if self.uploaded_files:
            context_parts = []
            max_file_size = 50000  # Limit each file to 50k chars before parsing
            for file_path in self.uploaded_files:
                suffix = file_path.suffix.lower()
                # Check for images
                if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    try:
                        with open(file_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                            images_b64.append(b64_data)
                    except Exception as e:
                        self._set_status(f"Error processing image {file_path.name}: {e}")
                    continue
                
                if FileParser:
                    parsed = FileParser.parse_file(file_path)
                    if parsed:
                        # Limit individual file content size
                        if len(parsed) > max_file_size:
                            # Truncate at sentence boundary
                            truncated = parsed[:max_file_size]
                            last_period = truncated.rfind('.')
                            last_newline = truncated.rfind('\n')
                            cutoff = max(last_period, last_newline)
                            if cutoff > max_file_size * 0.8:
                                parsed = parsed[:cutoff + 1] + "\n\n[... file content truncated for processing ...]"
                            else:
                                parsed = truncated + "\n\n[... file content truncated for processing ...]"
                        
                        context_parts.append(FileParser.format_context_block(parsed, file_path.name))
            context_block = "\n".join(context_parts)
            
            # Final safety check: if total context is still huge, truncate aggressively
            if len(context_block) > 100000:  # 100k chars = ~25k tokens, way too much
                self._set_status(f"Warning: File context very large ({len(context_block)} chars), will be summarized...")
                # Truncate to 20k chars before passing to DiscussionManager for summarization
                context_block = context_block[:20000] + "\n\n[... additional content truncated ...]"

        # Handle mode selection
        if self.mode == "discussion":
            self._send_discussion_mode(base, selected, question, context_block, images_b64)
        else:
            self._send_deliberation_mode(base, selected, question, context_block, images_b64)
    
    def _send_deliberation_mode(self, base: str, selected: List[str], question: str, context_block: str, images: List[str] = []):
        """Send in Deliberation Mode (existing functionality)."""
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

        # Inject context block into prompts if available
        if context_block:
            for m in roles_map:
                if roles_map[m]:
                    roles_map[m] = context_block + "\n\n" + roles_map[m]
                else:
                    roles_map[m] = context_block

        # Prepare UI
        self._prepare_tabs(selected)
        self._append_chat(f"You: {question}")
        if images:
            self._append_chat(f"<i>[Attached {len(images)} image(s)]</i>")
        self.prompt_edit.clear()
        self._set_status("Starting council …")

        conc =  int(self.conc_spin.value())
        single_voter_enabled = self.single_voter_check.isChecked()
        single_voter_model   = self.single_voter_combo.currentText().strip()
        
        # Get temperature from slider (convert 0-100 to 0.0-1.0)
        temp = self.temp_slider.value() / 100.0
        
        # Web search status
        web_search = self.web_search_check.isChecked()

        def worker():
            try:
                self._stop_requested = False
                voters = None
                if single_voter_enabled and single_voter_model:
                    # Validate that single voter model is available
                    if single_voter_model in self.models:
                        voters = [single_voter_model]
                    else:
                        self.status_signal.emit(f"⚠ Single voter '{short_id(single_voter_model)}' not available, using all models as voters")
                answers, winner, details, tally = asyncio.run(
                    council_round(
                        base, selected, question, roles_map, self.status_signal.emit,
                        max_concurrency=conc, voter_override=voters,
                        images=images, web_search=web_search, temperature=temp,
                        is_cancelled=lambda: self._stop_requested
                    )
                )
                if not self._stop_requested:
                    # record leaderboard
                    try:
                        record_vote(question, winner, details)
                    except Exception:
                        pass
                    self.result_signal.emit((question, answers, winner, details, tally))
            except Exception as e:
                if not self._stop_requested:
                    self.error_signal.emit(str(e))

        self._current_worker_thread = threading.Thread(target=worker, daemon=True)
        self._current_worker_thread.start()
        self._busy(True)
    
    def _send_discussion_mode(self, base: str, selected: List[str], question: str, context_block: str):
        """Send in Collaborative Discussion Mode."""
        if DiscussionManager is None:
            QtWidgets.QMessageBox.warning(self, "Discussion Mode Unavailable", 
                "Discussion mode requires core modules. Please check installation.")
            return
        
        # Build agent configurations
        agents = []
        for model in selected:
            persona_name = self.persona_assignments.get(model, "None")
            persona_config = self._build_persona_config(model, persona_name)
            
            agent = {
                "name": f"Agent {model[:20]}",
                "model": model,
                "is_active": True,
                "persona_config": persona_config,
                "persona_name": persona_name if persona_name != "None" else None
            }
            agents.append(agent)
        
        # Prepare UI for discussion mode
        self._prepare_discussion_tabs(selected)
        self._append_chat(f"You: {question}")
        self.prompt_edit.clear()
        self._set_status("Starting collaborative discussion…")

        conc = int(self.conc_spin.value())

        def worker():
            try:
                self._stop_requested = False
                manager = DiscussionManager(
                    base_url=base,
                    agents=agents,
                    user_prompt=question,
                    context_block=context_block,
                    status_callback=self.status_signal.emit,
                    update_callback=lambda entry: self.discussion_update_signal.emit(entry),
                    max_turns=10,
                    max_concurrency=conc
                )
                transcript, synthesis = asyncio.run(manager.run_discussion())
                if not self._stop_requested:
                    self.result_signal.emit(("discussion", question, transcript, synthesis))
            except Exception as e:
                if not self._stop_requested:
                    self.error_signal.emit(str(e))

        self._current_worker_thread = threading.Thread(target=worker, daemon=True)
        self._current_worker_thread.start()
        self._busy(True)
    
    def _build_persona_config(self, model: str, persona_name: str) -> Dict:
        """Build persona_config structure for an agent."""
        # Try to match persona_name to default or user personas
        persona_id = None
        source = "default"
        
        # Check default personas
        if DEFAULT_PERSONAS_PATH.exists():
            try:
                with open(DEFAULT_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                    default_personas = json.load(f)
                    for p in default_personas:
                        if p.get("name") == persona_name:
                            persona_id = p.get("id")
                            source = "default"
                            break
            except Exception:
                pass
        
        # Check user personas if not found in defaults
        if not persona_id and USER_PERSONAS_PATH.exists():
            try:
                with open(USER_PERSONAS_PATH, 'r', encoding='utf-8') as f:
                    user_personas = json.load(f)
                    for p in user_personas:
                        if p.get("name") == persona_name:
                            persona_id = p.get("id")
                            source = "user_custom"
                            break
            except Exception:
                pass
        
        # Fallback to legacy persona system
        if not persona_id:
            persona_prompt = self._persona_prompt(persona_name)
            if persona_prompt:
                return {
                    "source": "one_time",
                    "id": None,
                    "one_time_prompt": persona_prompt
                }
            else:
                return {
                    "source": "default",
                    "id": None,
                    "one_time_prompt": ""
                }
        
        return {
            "source": source,
            "id": persona_id,
            "one_time_prompt": ""
        }
    
    def _prepare_discussion_tabs(self, selected: List[str]):
        """Prepare UI tabs for discussion mode."""
        # Clear existing tabs
        self.tabs.clear()
        self.model_tabs.clear()
        self.model_texts.clear()
        
        # Create discussion view tab
        discussion_page = QtWidgets.QWidget()
        discussion_layout = QtWidgets.QVBoxLayout(discussion_page)
        
        # Switch to QTextBrowser for markdown/HTML
        self.discussion_view = QtWidgets.QTextBrowser()
        self.discussion_view.setOpenExternalLinks(True)
        self.discussion_view.setReadOnly(True)
        discussion_layout.addWidget(self.discussion_view)
        
        # Export button for discussion
        export_layout = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export Discussion")
        export_btn.clicked.connect(self._export_discussion)
        export_layout.addStretch(1)
        export_layout.addWidget(export_btn)
        discussion_layout.addLayout(export_layout)
        
        self.tabs.addTab(discussion_page, "Live Discussion")
        self.model_tabs["_DISCUSSION_"] = discussion_page
        self.model_texts["_DISCUSSION_"] = self.discussion_view
        
        # Initialize discussion transcript storage
        self.discussion_transcript = []
        
        # Create final report tab
        report_page = QtWidgets.QWidget()
        report_layout = QtWidgets.QVBoxLayout(report_page)
        # Switch to QTextBrowser
        self.report_view = QtWidgets.QTextBrowser()
        self.report_view.setOpenExternalLinks(True)
        self.report_view.setReadOnly(True)
        report_layout.addWidget(self.report_view)
        self.tabs.addTab(report_page, "Final Report")
        self.model_tabs["_REPORT_"] = report_page
        self.model_texts["_REPORT_"] = self.report_view
    
    def _update_discussion_view(self, entry: Dict):
        """Update the discussion view in real-time as each agent responds."""
        if not hasattr(self, 'discussion_view') or not self.discussion_view:
            return
        
        # Add entry to transcript
        self.discussion_transcript.append(entry)
        
        # Format and update the view
        agent = entry.get("agent", "Unknown")
        persona = entry.get("persona")
        message = entry.get("message", "")
        turn = entry.get("turn", 0)
        
        # Include persona label if available
        if persona:
            agent_label = f"{agent} <span style='color: #666; font-size: 0.9em;'>[{persona}]</span>"
        else:
            agent_label = agent
        
        # Style for discussion bubbles
        style = "margin: 5px 0; padding: 10px; border-radius: 8px; background-color: #f9f9f9; border: 1px solid #eee; color: #000000;"
        
        # Convert message to markdown
        msg_html = markdown.markdown(message)
        
        new_html = f"""
        <div style="{style}">
            <div style="font-weight: bold; margin-bottom: 5px; color: #333;">[Turn {turn}] {agent_label}</div>
            <div>{msg_html}</div>
        </div>
        """
        
        # Append to existing content
        self.discussion_view.append(new_html)
        
        # Auto-scroll to bottom
        self.discussion_view.verticalScrollBar().setValue(
            self.discussion_view.verticalScrollBar().maximum()
        )
    
    def _export_discussion(self):
        """Export the current discussion to a Markdown file."""
        if not hasattr(self, 'discussion_transcript') or not self.discussion_transcript:
            QtWidgets.QMessageBox.information(self, "No Data", "No discussion data to export.")
            return
            
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Discussion", "discussion_report.md", "Markdown Files (*.md);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# PolyCouncil Discussion Report\n\n")
                f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Question:** {self.prompt_edit.text() if hasattr(self, 'prompt_edit') and self.prompt_edit.text() else 'N/A'}\n\n")
                
                f.write("## Transcript\n\n")
                for entry in self.discussion_transcript:
                    agent = entry.get("agent", "Unknown")
                    persona = entry.get("persona")
                    message = entry.get("message", "")
                    turn = entry.get("turn", 0)
                    
                    agent_label = f"{agent} [{persona}]" if persona else agent
                    
                    f.write(f"### Turn {turn}: {agent_label}\n\n")
                    f.write(f"{message}\n\n")
                    f.write("---\n\n")
                
                # Include synthesis if available (extract from report view text)
                if hasattr(self, 'report_view') and self.report_view:
                    report_content = self.report_view.toPlainText()
                    if "=== FINAL SYNTHESIS ===" in report_content:
                        synthesis_part = report_content.split("=== FINAL SYNTHESIS ===")[1].strip()
                        f.write("## Final Synthesis\n\n")
                        f.write(f"{synthesis_part}\n")
            
            self._set_status(f"Discussion exported to {Path(file_path).name}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export discussion: {str(e)}")

    def _handle_error(self, msg: str):
        self._busy(False)
        self._set_status(f"Error: {msg}")
        self._append_chat(f"[Error] {msg}")

    def _handle_result(self, payload: object):
        self._busy(False)
        
        # Check if this is a discussion mode result
        if isinstance(payload, tuple) and len(payload) > 0 and payload[0] == "discussion":
            self._handle_discussion_result(payload)
        else:
            self._handle_deliberation_result(payload)
    
    def _handle_deliberation_result(self, payload: object):
        """Handle result from Deliberation Mode."""
        question, answers, winner, details, tally = payload

        # Fill each model tab with its answer
        for mid, txtw in self.model_texts.items():
            if mid == "_RESULTS_":
                continue
            # If it's a QPlainTextEdit (legacy tabs), update text. If QTextBrowser, use setHtml
            if isinstance(txtw, QtWidgets.QTextBrowser):
                html = markdown.markdown(answers.get(mid, ""))
                txtw.setHtml(html)
            else:
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

        # Results tab is QPlainTextEdit, not QTextBrowser, so use setPlainText
        self.model_texts["_RESULTS_"].setPlainText("\n".join(lines))

        # Show winning answer in chat
        ans = answers.get(winner, "")
        self._append_chat(f"Council → {short_id(winner)}:\n{ans}")
        self._set_status("Done.")
    
    def _handle_discussion_result(self, payload: object):
        """Handle result from Collaborative Discussion Mode."""
        mode, question, transcript, synthesis = payload
        
        # Display transcript in discussion view
        if hasattr(self, 'discussion_view') and self.discussion_view:
            transcript_text = []
            for entry in transcript:
                agent = entry.get("agent", "Unknown")
                persona = entry.get("persona")
                message = entry.get("message", "")
                turn = entry.get("turn", 0)
                
                # Include persona label if available
                if persona:
                    agent_label = f"{agent} [{persona}]"
                else:
                    agent_label = agent
                
                transcript_text.append(f"[Turn {turn}] {agent_label}:\n{message}\n")
            
            self.discussion_view.setPlainText("\n".join(transcript_text))
        
        # Display synthesis in report view
        if hasattr(self, 'report_view') and self.report_view:
            report_text = f"""=== COLLABORATIVE DISCUSSION REPORT ===

Question: {question}

=== DISCUSSION TRANSCRIPT ===

"""
            for entry in transcript:
                agent = entry.get("agent", "Unknown")
                persona = entry.get("persona")
                message = entry.get("message", "")
                turn = entry.get("turn", 0)
                
                # Include persona label if available
                if persona:
                    agent_label = f"{agent} [{persona}]"
                else:
                    agent_label = agent
                
                report_text += f"[Turn {turn}] {agent_label}:\n{message}\n\n"
            
            report_text += f"""
=== FINAL SYNTHESIS ===

{synthesis if synthesis and synthesis.strip() else "Synthesis not available. The synthesis may have failed to generate or was empty."}

"""
            self.report_view.setPlainText(report_text)
        
        # Show summary in chat
        self._append_chat(f"Discussion complete. {len(transcript)} turns recorded.")
        if synthesis and synthesis.strip():
            # Show full synthesis in chat (no truncation)
            self._append_chat(f"Final Synthesis:\n{synthesis}")
        elif synthesis is None:
            self._append_chat("Note: Synthesis generation failed. Check console for details.")
        
        self._set_status("Discussion complete.")

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
