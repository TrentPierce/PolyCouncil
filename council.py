
# council_gui_qt.py
# GUI: PySide6 (Qt widgets)
# Deps: PySide6, aiohttp
# Optional: qdarktheme (auto dark/light), lmstudio (local-only "loaded models" detection)

import asyncio
import aiohttp
import sqlite3
import datetime
import json
import threading
import re
import random
import uuid
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Any, Iterable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# Import new core modules
try:
    from core.tool_manager import FileParser, ModelCapabilityDetector
    from core.api_client import ModelFetchError, call_model, fetch_models
    from core.discussion_manager import DiscussionManager
    from core import result_presenter
    from core.provider_config import (
        API_SERVICE_CUSTOM,
        API_SERVICE_GEMINI,
        API_SERVICE_LABELS,
        API_SERVICE_OPENAI,
        API_SERVICE_OPENROUTER,
        PROVIDER_LABELS,
        PROVIDER_LM_STUDIO,
        PROVIDER_OLLAMA,
        PROVIDER_OPENAI_COMPAT,
        ProviderConfig,
        api_service_label,
        canonicalize_base_url,
        make_provider_config,
        normalize_api_service,
        normalize_provider_type,
        provider_defaults,
        provider_label,
        service_preset,
    )
    from core.settings_store import load_settings, save_settings
    from core.app_state import app_config_dir, app_data_dir, app_log_dir, legacy_root_dir, migrate_legacy_state
    from core.rendering import escape_text, markdown_to_safe_html
    from core.session_history import load_session, save_session
except ImportError:
    # Fallback if modules not found
    FileParser = None
    ModelCapabilityDetector = None
    ModelFetchError = None
    call_model = None
    fetch_models = None
    DiscussionManager = None
    result_presenter = None
    API_SERVICE_CUSTOM = "custom"
    API_SERVICE_OPENAI = "openai"
    API_SERVICE_OPENROUTER = "openrouter"
    API_SERVICE_GEMINI = "gemini"
    API_SERVICE_LABELS = {}
    PROVIDER_LM_STUDIO = "lm_studio"
    PROVIDER_OPENAI_COMPAT = "openai_compatible"
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_LABELS = {}
    ProviderConfig = None
    api_service_label = None
    canonicalize_base_url = None
    make_provider_config = None
    normalize_api_service = None
    normalize_provider_type = None
    provider_defaults = None
    provider_label = None
    service_preset = None
    load_settings = None
    save_settings = None
    app_config_dir = None
    app_data_dir = None
    app_log_dir = None
    legacy_root_dir = None
    migrate_legacy_state = None
    escape_text = None
    markdown_to_safe_html = None
    load_session = None
    save_session = None

# Try optional theming (follows system)
try:
    import qdarktheme  # type: ignore
except Exception:
    qdarktheme = None

# Import new UI package
try:
    from ui.theme import ThemeEngine
    from ui.animations import FadeIn
    from ui.components import (
        CollapsibleGroupBox,
        ModelCard,
        ToastNotification,
        EnhancedPromptEditor,
        AnimatedStatusBar,
        OnboardingOverlay,
        KeyboardShortcutOverlay,
    )
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False

try:
    from gui import build_provider_profile_row, build_workspace_panel, clear_layout
except ImportError:
    build_provider_profile_row = None
    build_workspace_panel = None
    clear_layout = None

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


def human_file_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(size)} B"


class PromptEditor(QtWidgets.QPlainTextEdit):
    submitRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._min_editor_height = 74
        self._max_editor_height = 180
        self.setTabChangesFocus(True)
        self.setPlaceholderText("Ask the council a question. Press Enter to send, Shift+Enter for a new line.")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.document().documentLayout().documentSizeChanged.connect(self._sync_height)
        self._sync_height()

    def _sync_height(self, *_args):
        doc_height = self.document().size().height()
        frame = self.frameWidth() * 2
        padding = 14
        target = int(doc_height + frame + padding)
        target = max(self._min_editor_height, min(target, self._max_editor_height))
        self.setFixedHeight(target)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if not (event.modifiers() & QtCore.Qt.ShiftModifier):
                self.submitRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_height)

    def clear(self):
        super().clear()
        self._sync_height()


class AttachmentListWidget(QtWidgets.QListWidget):
    filesDropped = QtCore.Signal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.empty_text = "Drop documents or images here, or use Upload Files."
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if self._extract_local_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if self._extract_local_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        paths = self._extract_local_paths(event.mimeData())
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if self.count() or not self.empty_text:
            return
        painter = QtGui.QPainter(self.viewport())
        color = self.palette().color(QtGui.QPalette.Mid)
        painter.setPen(color)
        painter.drawText(
            self.viewport().rect().adjusted(18, 12, -18, -12),
            QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
            self.empty_text,
        )
        painter.end()

    @staticmethod
    def _extract_local_paths(mime: Optional[QtCore.QMimeData]) -> List[str]:
        if not mime or not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if local_path:
                    paths.append(local_path)
        return paths


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
LEGACY_ROOT = legacy_root_dir() if legacy_root_dir else APP_DIR
if migrate_legacy_state:
    migrate_legacy_state()
CONFIG_DIR = app_config_dir() if app_config_dir else APP_DIR
DATA_DIR = app_data_dir() if app_data_dir else APP_DIR
LOG_DIR = app_log_dir() if app_log_dir else APP_DIR
DB_PATH = DATA_DIR / "council_stats.db"
DEFAULT_PERSONAS_PATH = APP_DIR / "config" / "default_personas.json"
USER_PERSONAS_PATH = CONFIG_DIR / "user_personas.json"
def log_unhandled_exception(exc_type, exc_value, exc_tb):
    try:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_path = LOG_DIR / "polycouncil_crash.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"[{ts}] Unhandled exception\n{tb_text}\n", encoding="utf-8")
    except Exception:
        pass

# -----------------------
# Database (leaderboard)
# -----------------------
def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
# Voting schema & parsing
# -----------------------
DEFAULT_RUBRIC_WEIGHTS = {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}
RUBRIC_PRESETS = {
    "Balanced": dict(DEFAULT_RUBRIC_WEIGHTS),
    "Accuracy First": {"correctness": 6, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1},
    "Safety First": {"correctness": 4, "relevance": 2, "specificity": 2, "safety": 5, "conciseness": 1},
    "Concise": {"correctness": 4, "relevance": 3, "specificity": 2, "safety": 2, "conciseness": 3},
}

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


def normalize_rubric_weights(weights: Optional[Dict[str, Any]]) -> Dict[str, int]:
    normalized = dict(DEFAULT_RUBRIC_WEIGHTS)
    if not isinstance(weights, dict):
        return normalized
    for key, default_value in DEFAULT_RUBRIC_WEIGHTS.items():
        value = weights.get(key, default_value)
        try:
            normalized[key] = max(0, int(value))
        except Exception:
            normalized[key] = default_value
    return normalized

# -----------------------
# Voting worker
# -----------------------
async def vote_one(
    session: aiohttp.ClientSession,
    voter_entry: Dict[str, Any],
    question: str,
    answers: Dict[str, str],
    model_entries: List[Dict[str, Any]],
) -> Tuple[str, Optional[dict], str]:
    """
    Ask a (voter) model to score peer answers and pick a winner.
    Returns: (voter_id, parsed_ballot, status_message)
    """
    voter_id = str(voter_entry["id"])
    voter_raw_model = str(voter_entry["model"])
    voter_provider = voter_entry["provider"]
    peer_entries = [m for m in model_entries if str(m["id"]) != voter_id]
    idx_map_peer = {i + 1: str(m["id"]) for i, m in enumerate(peer_entries)}

    if not peer_entries:
        return voter_id, None, "No peer candidates to score."

    parts = [VOTE_INSTRUCTIONS, "", f"Question:\n{question}", "", "Candidates:"]
    for i, m in idx_map_peer.items():
        ans = answers.get(m, "")
        parts.append(f"[{i}] {short_id(m)}:\n{ans}")
    prompt = "\n\n".join(parts)

    try:
        schema = ballot_json_schema(num_candidates=len(idx_map_peer))
        content = await call_model(
            session, voter_provider, voter_raw_model, prompt,
            debug_hook=_dbg,
            sys_prompt="Be precise. Return only JSON according to the schema.",
            json_schema=schema
        )
        _dbg(f"VOTE raw OUTPUT json_schema (voter={voter_id})", content)
        parsed = safe_load_vote_json(content)
        if parsed:
            ok, msg = validate_ballot(idx_map_peer, parsed)
            if ok:
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", ""),
                }
                _dbg(f"VOTE ACCEPTED json_schema (voter={voter_id})", normalized)
                return voter_id, normalized, f"Valid ballot via json_schema ({msg})"
    except Exception as e:
        _dbg(f"VOTE ERROR json_schema (voter={voter_id})", str(e))

    try:
        content = await call_model(
            session, voter_provider, voter_raw_model, prompt,
            debug_hook=_dbg,
            sys_prompt="Return only JSON for the ballot. No extra text.",
            json_schema=None
        )
        _dbg(f"VOTE raw OUTPUT text (voter={voter_id})", content)
        parsed = safe_load_vote_json(content)
        if parsed:
            ok, msg = validate_ballot(idx_map_peer, parsed)
            if ok:
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", ""),
                }
                _dbg(f"VOTE ACCEPTED text (voter={voter_id})", normalized)
                return voter_id, normalized, f"Valid ballot via text ({msg})"
            else:
                _dbg(f"VOTE REJECTED text (voter={voter_id})", msg)
    except Exception as e:
        _dbg(f"VOTE ERROR text (voter={voter_id})", str(e))

    _dbg(f"VOTE FAILED ALL ATTEMPTS (voter={voter_id})", "No valid ballot produced")
    return voter_id, None, f"Failed to obtain valid ballot from {short_id(voter_id)}"

# -----------------------
# Council orchestration
# -----------------------
async def council_round(
    model_entries: List[Dict[str, Any]],
    question: str,
    roles: Dict[str, Optional[str]],
    status_cb: Callable[[str], None],
    max_concurrency: int = 1,
    voter_override: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
    web_search: bool = False,
    temperature: Optional[float] = None,
    rubric_weights: Optional[Dict[str, int]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None
):
    status_cb("Collecting answers…")
    async with aiohttp.ClientSession() as session:
        weights = normalize_rubric_weights(rubric_weights)

        images = images or []
        async def answer_one(entry: Dict[str, Any]) -> tuple[str, str, int]:
            model_id = str(entry["id"])
            raw_model = str(entry["model"])
            provider = entry["provider"]
            if is_cancelled and is_cancelled():
                return model_id, "[Cancelled]", 0
            
            user_prompt = question
            sys_p = roles.get(model_id) or None
            started_at = datetime.datetime.now()
            try:
                ans = await call_model(
                    session, provider, raw_model, user_prompt,
                    debug_hook=_dbg,
                    temperature=temperature, sys_prompt=sys_p,
                    images=images, web_search=web_search
                )
                if isinstance(ans, dict):
                    ans = json.dumps(ans, ensure_ascii=False)
                elapsed_ms = int((datetime.datetime.now() - started_at).total_seconds() * 1000)
                return model_id, ans, elapsed_ms
            except Exception as e1:
                elapsed_ms = int((datetime.datetime.now() - started_at).total_seconds() * 1000)
                return model_id, f"[ERROR fetching answer]\n{e1}", elapsed_ms

        # Check cancellation before starting
        if is_cancelled and is_cancelled():
            raise RuntimeError("Process cancelled by user")

        pairs = await run_limited(max_concurrency, [lambda m=m: answer_one(m) for m in model_entries])
        
        # Check cancellation after answers
        if is_cancelled and is_cancelled():
            raise RuntimeError("Process cancelled by user")
            
        answers = {m: a for m, a, _ in pairs}
        timings_ms = {m: elapsed for m, _, elapsed in pairs}
        errors = {m: a for m, a, _ in pairs if isinstance(a, str) and a.startswith("[ERROR")}

        status_cb("Models are voting…")

        model_entry_by_id = {str(m["id"]): m for m in model_entries}
        selected_model_ids = list(model_entry_by_id.keys())
        voters_to_use = voter_override if voter_override else selected_model_ids
        # Note: Voting phase does not use images or web search, typically.
        vote_results = await run_limited(
            max_concurrency,
            [
                lambda m=m: vote_one(session, model_entry_by_id[m], question, answers, model_entries)
                for m in voters_to_use if m in model_entry_by_id
            ]
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

        totals: Dict[str, int] = {mid: 0 for mid in selected_model_ids}
        for ballot in valid_votes.values():
            for candidate_mid, score_dict in ballot["scores"].items():
                weighted = sum(score_dict[k] * weights[k] for k in weights.keys())
                totals[candidate_mid] = totals.get(candidate_mid, 0) + int(weighted)

        if not valid_votes:
            status_cb("⚠ NO VALID VOTES - selecting first model by default (consider checking model compatibility)")
            winner = selected_model_ids[0]
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
            "timings_ms": timings_ms,
            "rubric_weights": weights,
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

    def __init__(self, provider: ProviderConfig):
        super().__init__()
        self.provider = provider

    @QtCore.Slot()
    def run(self):
        try:
            models = asyncio.run(fetch_models(self.provider, provider_label=provider_label))
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

        # Theme engine (replaces qdarktheme)
        self._theme_engine: Optional[ThemeEngine] = None
        if _UI_AVAILABLE:
            app = QtWidgets.QApplication.instance()
            if app:
                self._theme_engine = ThemeEngine(app, self)
        elif qdarktheme:
            try:
                qdarktheme.setup_theme()
            except Exception:
                pass

        ensure_db()

        self.models: List[str] = []
        self.model_actual_ids: Dict[str, str] = {}
        self.model_provider_map: Dict[str, ProviderConfig] = {}
        self.model_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.model_tabs: Dict[str, QtWidgets.QWidget] = {}
        self.model_texts: Dict[str, QtWidgets.QWidget] = {}
        self.model_persona_combos: Dict[str, QtWidgets.QPushButton] = {}
        self.model_rows: Dict[str, QtWidgets.QWidget] = {}
        self.provider_type = PROVIDER_LM_STUDIO
        self.api_key = ""
        self.model_path = ""
        self.api_service = API_SERVICE_CUSTOM
        self.last_submitted_question = ""
        self.provider_profiles: List[Dict[str, str]] = []
        self.provider_profile_rows: Dict[str, QtWidgets.QWidget] = {}
        self.last_session_record: Optional[Dict[str, Any]] = None
        self.model_meta_labels: Dict[str, QtWidgets.QLabel] = {}
        
        # New features
        self.mode = "deliberation"  # "deliberation" or "discussion"
        self.uploaded_files: List[Path] = []
        self.model_capabilities: Dict[str, Dict[str, bool]] = {}  # model -> {web_search: bool, visual: bool}
        self.web_search_enabled = False
        self._fetch_append = False
        self._fetch_provider: Optional[ProviderConfig] = None
        self._status_tone = "neutral"

        self._build_ui()
        self._apply_window_style()
        self.chat_view.setHtml(
            self._placeholder_html(
                "Welcome",
                "Choose a provider, load some models, and send a prompt to compare answers or run a collaborative discussion.",
            )
        )
        self._setup_log_dock()
        self._setup_settings_dock()
        self._connect_signals()
        set_log_sink(self._log_sink_dispatch)

        # Keyboard shortcut overlay
        if _UI_AVAILABLE:
            self._shortcut_overlay = KeyboardShortcutOverlay(self)
        else:
            self._shortcut_overlay = None

        # restore settings
        s = load_settings()
        self.secure_keyring_available = bool(s.get("secure_keyring_available", False))
        self.secure_storage_ok = bool(s.get("secure_storage_ok", True))
        self.provider_type = normalize_provider_type(s.get("provider_type", PROVIDER_LM_STUDIO))
        self.api_key = s.get("api_key", "") or ""
        self.model_path = s.get("model_path", "") or ""
        self.api_service = normalize_api_service(s.get("api_service", API_SERVICE_CUSTOM))
        self.rubric_weights = normalize_rubric_weights(s.get("rubric_weights"))
        default_base, _ = provider_defaults(self.provider_type)
        self.base_edit.setText(s.get("base_url", default_base))
        self.api_key_inline.setText(self.api_key)
        self.provider_combo.setCurrentText(provider_label(self.provider_type))
        self.api_service_combo.setCurrentText(api_service_label(self.api_service))
        self._provider_changed(self.provider_combo.currentIndex())
        loaded_profiles = s.get("provider_profiles", [])
        if isinstance(loaded_profiles, list):
            normalized_profiles: List[Dict[str, str]] = []
            for p in loaded_profiles:
                if not isinstance(p, dict):
                    continue
                normalized_profiles.append({
                    "id": p.get("id") or str(uuid.uuid4()),
                    "provider_type": normalize_provider_type(p.get("provider_type", PROVIDER_LM_STUDIO)),
                    "api_service": normalize_api_service(p.get("api_service", API_SERVICE_CUSTOM)),
                    "base_url": str(p.get("base_url", "")),
                    "api_key": str(p.get("api_key", "")),
                    "model_path": str(p.get("model_path", "")),
                })
            self.provider_profiles = normalized_profiles
        if not self.provider_profiles:
            self.provider_profiles = [self._current_profile_payload()]
        self._render_provider_profiles()

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

        self._refresh_leaderboard()
        self._refresh_selection_summary()
        self._refresh_attachment_summary()
        self._refresh_header_badges()
        self._set_results_empty_state()
        self._refresh_settings_panel()
        self._refresh_persona_library_panel()
        if not self.secure_keyring_available:
            self._set_status("Secure keychain unavailable. Hosted API keys will only persist for the current session.")
        else:
            self._set_status("Ready. Choose a provider, then load models.")

        # Show onboarding on first launch
        if _UI_AVAILABLE and not s.get("onboarding_complete", False):
            QtCore.QTimer.singleShot(500, self._show_onboarding)

    # ----- UI -----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        central.setObjectName("Root")
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        hero = QtWidgets.QFrame()
        hero.setObjectName("HeroCard")
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 18, 18, 18)
        hero_layout.setSpacing(10)

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(14)
        nav.setContentsMargins(0, 0, 0, 0)
        title_block = QtWidgets.QVBoxLayout()
        title_block.setSpacing(2)
        title = QtWidgets.QLabel("PolyCouncil")
        title.setObjectName("HeroTitle")
        t_font = title.font()
        t_font.setPointSize(t_font.pointSize() + 6)
        t_font.setBold(True)
        title.setFont(t_font)
        subtitle = QtWidgets.QLabel(
            "Coordinate multiple models, compare their reasoning, and keep the workflow readable while runs are in flight."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        mode_label = QtWidgets.QLabel("Mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Deliberation Mode", "Collaborative Discussion Mode"])
        self.mode_combo.setCurrentIndex(0)
        self.settings_btn = QtWidgets.QPushButton("&Settings")
        self.settings_btn.setObjectName("SecondaryButton")
        self.settings_btn.setMinimumWidth(120)
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip("Open or close the workspace settings panel.")

        nav.addLayout(title_block, stretch=1)
        nav.addWidget(mode_label)
        nav.addWidget(self.mode_combo)
        nav.addWidget(self.settings_btn)
        hero_layout.addLayout(nav)

        badge_row = QtWidgets.QHBoxLayout()
        badge_row.setSpacing(8)
        self.mode_badge = QtWidgets.QLabel("Deliberation")
        self.mode_badge.setObjectName("InfoBadge")
        self.selection_badge = QtWidgets.QLabel("0 models selected")
        self.selection_badge.setObjectName("InfoBadge")
        self.attachment_badge = QtWidgets.QLabel("No attachments")
        self.attachment_badge.setObjectName("InfoBadge")
        self.tool_badge = QtWidgets.QLabel("Web off")
        self.tool_badge.setObjectName("InfoBadge")
        badge_row.addWidget(self.mode_badge)
        badge_row.addWidget(self.selection_badge)
        badge_row.addWidget(self.attachment_badge)
        badge_row.addWidget(self.tool_badge)
        badge_row.addStretch(1)
        hero_layout.addLayout(badge_row)
        layout.addWidget(hero)

        # --- Provider Connection (collapsible) ---
        if _UI_AVAILABLE:
            self._provider_collapsible = CollapsibleGroupBox("Provider Connection")
            provider_inner = QtWidgets.QWidget()
            provider_layout = QtWidgets.QGridLayout(provider_inner)
        else:
            provider_group = QtWidgets.QGroupBox("Provider Connection")
            provider_layout = QtWidgets.QGridLayout(provider_group)
        provider_layout.setContentsMargins(12, 10, 12, 12)
        provider_layout.setHorizontalSpacing(10)
        provider_layout.setVerticalSpacing(8)

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems([
            provider_label(PROVIDER_LM_STUDIO),
            provider_label(PROVIDER_OPENAI_COMPAT),
            provider_label(PROVIDER_OLLAMA),
        ])
        self.provider_combo.setMinimumWidth(220)

        self.api_service_combo = QtWidgets.QComboBox()
        self.api_service_combo.addItems([
            api_service_label(API_SERVICE_CUSTOM),
            api_service_label(API_SERVICE_OPENAI),
            api_service_label(API_SERVICE_OPENROUTER),
            api_service_label(API_SERVICE_GEMINI),
        ])
        self.api_service_combo.setMinimumWidth(170)
        self.api_service_label = QtWidgets.QLabel("API Service")

        self.base_edit = QtWidgets.QLineEdit()
        self.base_edit.setPlaceholderText("http://localhost:1234")
        self.base_label = QtWidgets.QLabel("Base URL")
        self.api_key_inline_label = QtWidgets.QLabel("API Key")
        self.api_key_inline = QtWidgets.QLineEdit()
        self.api_key_inline.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_inline.setPlaceholderText("Paste API key")
        self.show_key_check = QtWidgets.QCheckBox("Show")

        self.connect_btn = QtWidgets.QPushButton("&Load models")
        self.connect_btn.setObjectName("PrimaryButton")
        self.replace_models_btn = QtWidgets.QPushButton("Replace List")
        self.replace_models_btn.setToolTip("Clear the current model list and replace it with the selected provider's models.")
        self.connect_btn.setToolTip("Load models from the provider currently shown here and append them to the model list.")
        self.add_provider_btn = QtWidgets.QPushButton("&Save profile")
        self.add_provider_btn.setObjectName("SecondaryButton")

        provider_layout.addWidget(
            self._create_form_field("Provider", self.provider_combo),
            0, 0
        )
        self.api_service_field = self._create_form_field(
            "API Service",
            self.api_service_combo,
            helper_text="Choose a hosted service preset or keep Custom for your own compatible endpoint.",
        )
        provider_layout.addWidget(self.api_service_field, 0, 1)
        provider_layout.addWidget(
            self._create_form_field(
                "Base URL",
                self.base_edit,
                helper_text="Use the local or hosted API base URL for the current provider.",
            ),
            1, 0, 1, 2
        )
        provider_layout.addWidget(
            self._create_form_field(
                "API Key",
                self.api_key_inline,
                helper_text="Keys are stored securely when a Windows keychain backend is available.",
                trailing=self.show_key_check,
            ),
            2, 0, 1, 2
        )
        provider_actions = QtWidgets.QHBoxLayout()
        provider_actions.setContentsMargins(0, 0, 0, 0)
        provider_actions.setSpacing(8)
        provider_actions.addWidget(self.connect_btn)
        provider_actions.addWidget(self.replace_models_btn)
        provider_actions.addWidget(self.add_provider_btn)
        provider_actions.addStretch(1)
        provider_layout.addLayout(provider_actions, 3, 0, 1, 2)
        provider_layout.setColumnStretch(0, 1)
        provider_layout.setColumnStretch(1, 1)
        if _UI_AVAILABLE:
            self._provider_collapsible.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        else:
            provider_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        # --- Run Settings (collapsible) ---
        if _UI_AVAILABLE:
            self._session_collapsible = CollapsibleGroupBox("Run Settings")
            session_inner = QtWidgets.QWidget()
            session_layout = QtWidgets.QGridLayout(session_inner)
        else:
            session_group = QtWidgets.QGroupBox("Run Settings")
            session_layout = QtWidgets.QGridLayout(session_group)
        session_layout.setContentsMargins(12, 10, 12, 12)
        session_layout.setHorizontalSpacing(10)
        session_layout.setVerticalSpacing(8)

        self.single_voter_check = QtWidgets.QCheckBox("Single-voter")
        self.single_voter_combo = QtWidgets.QComboBox()
        self.single_voter_combo.setMinimumWidth(220)

        self.conc_label = QtWidgets.QLabel("Max concurrent jobs")
        self.conc_spin = QtWidgets.QSpinBox()
        self.conc_spin.setRange(1, 8)
        self.conc_spin.setValue(1)
        self.web_search_check = QtWidgets.QCheckBox("Enable Web Search")
        self.web_search_check.setEnabled(False)
        self.web_search_check.setToolTip("Enable only when a selected model exposes web tools.")
        self.mode_help_label = QtWidgets.QLabel(
            "Use the header mode switch to choose weighted deliberation or collaborative discussion."
        )
        self.mode_help_label.setObjectName("HintLabel")
        self.mode_help_label.setWordWrap(True)

        session_layout.addWidget(self.mode_help_label, 0, 0, 1, 2)
        self.single_voter_field = self._create_form_field(
            "Voting Mode",
            self.single_voter_combo,
            helper_text="Enable single-voter mode to let one selected model judge all candidates.",
            trailing=self.single_voter_check,
        )
        session_layout.addWidget(self.single_voter_field, 1, 0, 1, 2)
        session_layout.addWidget(
            self._create_form_field(
                "Concurrency",
                self.conc_spin,
                helper_text="Keep this low on local hardware for steadier runs.",
            ),
            2, 0
        )
        session_layout.addWidget(
            self._create_form_field(
                "Tools",
                self.web_search_check,
                helper_text="Turn web tools on only when the selected models support them.",
            ),
            2, 1
        )
        session_layout.setColumnStretch(0, 1)
        session_layout.setColumnStretch(1, 1)
        if _UI_AVAILABLE:
            self._session_collapsible.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        else:
            session_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        # Assemble the controls area
        controls_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        controls_splitter.setChildrenCollapsible(False)
        controls_splitter.setHandleWidth(1)
        controls_splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        if _UI_AVAILABLE:
            self._provider_collapsible.add_widget(provider_inner)
            self._session_collapsible.add_widget(session_inner)
            controls_splitter.addWidget(self._provider_collapsible)
            controls_splitter.addWidget(self._session_collapsible)
        else:
            controls_splitter.addWidget(provider_group)
            controls_splitter.addWidget(session_group)
        controls_splitter.setSizes([700, 420])
        layout.addWidget(controls_splitter)

        # --- Saved Provider Profiles (collapsible, starts collapsed) ---
        if _UI_AVAILABLE:
            self._profiles_collapsible = CollapsibleGroupBox("Saved Provider Profiles", start_collapsed=True)
            profiles_content = QtWidgets.QWidget()
            providers_group_layout = QtWidgets.QVBoxLayout(profiles_content)
        else:
            providers_group = QtWidgets.QGroupBox("Saved Provider Profiles")
            providers_group_layout = QtWidgets.QVBoxLayout(providers_group)
        providers_group_layout.setContentsMargins(14, 12, 14, 14)
        providers_group_layout.setSpacing(8)
        providers_help = QtWidgets.QLabel(
            "Keep reusable endpoints here. Use loads the profile into the form, and Load fetches models from that saved provider."
        )
        providers_help.setObjectName("HintLabel")
        providers_help.setWordWrap(True)
        providers_group_layout.addWidget(providers_help)
        self.providers_scroll = QtWidgets.QScrollArea()
        self.providers_scroll.setWidgetResizable(True)
        self.providers_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.providers_inner = QtWidgets.QWidget()
        self.providers_layout = QtWidgets.QVBoxLayout(self.providers_inner)
        self.providers_layout.setContentsMargins(0, 0, 0, 0)
        self.providers_layout.setSpacing(6)
        self.providers_layout.addStretch(1)
        self.providers_scroll.setWidget(self.providers_inner)
        providers_group_layout.addWidget(self.providers_scroll)
        if _UI_AVAILABLE:
            self._profiles_collapsible.add_widget(profiles_content)
            layout.addWidget(self._profiles_collapsible)
        else:
            providers_group.setMaximumHeight(220)
            layout.addWidget(providers_group)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        sidebar_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        sidebar_splitter.setChildrenCollapsible(False)

        models_group = QtWidgets.QGroupBox("Model Selection")
        models_layout = QtWidgets.QVBoxLayout(models_group)
        models_layout.setContentsMargins(12, 16, 12, 12)
        models_layout.setSpacing(10)
        self.model_filter_edit = QtWidgets.QLineEdit()
        self.model_filter_edit.setPlaceholderText("Filter models by provider, capability, or name")
        self.model_selection_label = QtWidgets.QLabel("0 selected of 0 loaded")
        self.model_selection_label.setObjectName("HintLabel")
        model_btn_row = QtWidgets.QHBoxLayout()
        self.refresh_models_btn = QtWidgets.QPushButton("&Reload provider")
        self.select_all_btn = QtWidgets.QPushButton("Select &All")
        self.clear_btn = QtWidgets.QPushButton("Select &None")
        self.clear_model_list_btn = QtWidgets.QPushButton("Clear Models")
        self.refresh_models_btn.setToolTip("Reload models from the provider currently shown in the provider card.")
        self.clear_model_list_btn.setToolTip("Remove all loaded models from all providers.")
        model_btn_row.addWidget(self.refresh_models_btn)
        model_btn_row.addWidget(self.select_all_btn)
        model_btn_row.addWidget(self.clear_btn)
        model_btn_row.addWidget(self.clear_model_list_btn)
        self.models_area = QtWidgets.QScrollArea()
        self.models_area.setWidgetResizable(True)
        self.models_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.models_inner = QtWidgets.QWidget()
        self.models_layout = QtWidgets.QVBoxLayout(self.models_inner)
        self.models_layout.setContentsMargins(6, 6, 6, 6)
        self.models_layout.setSpacing(4)
        self.models_layout.addStretch(1)
        self.models_area.setWidget(self.models_inner)
        models_layout.addWidget(self.model_filter_edit)
        models_layout.addWidget(self.model_selection_label)
        models_layout.addLayout(model_btn_row)
        models_layout.addWidget(self.models_area, stretch=1)

        self.leaderboard_group = QtWidgets.QGroupBox("Leaderboard")
        leaderboard_layout = QtWidgets.QVBoxLayout(self.leaderboard_group)
        leaderboard_layout.setContentsMargins(12, 16, 12, 12)
        leaderboard_layout.setSpacing(10)
        lb_header = QtWidgets.QHBoxLayout()
        self.lb_title = QtWidgets.QLabel("Council performance over time")
        self.lb_title.setObjectName("HintLabel")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        lb_header.addWidget(self.lb_title)
        lb_header.addStretch(1)
        lb_header.addWidget(self.reset_btn)
        self.leader_list = QtWidgets.QListWidget()
        leaderboard_layout.addLayout(lb_header)
        leaderboard_layout.addWidget(self.leader_list, stretch=1)

        sidebar_splitter.addWidget(models_group)
        sidebar_splitter.addWidget(self.leaderboard_group)
        sidebar_splitter.setSizes([620, 240])
        main_splitter.addWidget(sidebar_splitter)

        workspace_group = QtWidgets.QGroupBox("Workspace")
        workspace_layout = QtWidgets.QVBoxLayout(workspace_group)
        workspace_layout.setContentsMargins(12, 16, 12, 12)
        workspace_layout.setSpacing(10)

        attachment_group = QtWidgets.QGroupBox("Context & Attachments")
        attachment_layout = QtWidgets.QVBoxLayout(attachment_group)
        attachment_layout.setContentsMargins(12, 12, 12, 12)
        attachment_layout.setSpacing(8)
        file_btn_row = QtWidgets.QHBoxLayout()
        self.upload_btn = QtWidgets.QPushButton("&Upload files")
        self.upload_btn.setObjectName("SecondaryButton")
        self.remove_file_btn = QtWidgets.QPushButton("Remove Selected")
        self.clear_files_btn = QtWidgets.QPushButton("Clear All")
        self.visual_status = QtWidgets.QLabel("Visual/Image Support: Not detected")
        self.visual_status.setObjectName("HintLabel")
        file_btn_row.addWidget(self.upload_btn)
        file_btn_row.addWidget(self.remove_file_btn)
        file_btn_row.addWidget(self.clear_files_btn)
        file_btn_row.addStretch(1)
        file_btn_row.addWidget(self.visual_status)
        self.files_list = AttachmentListWidget()
        self.files_list.setMaximumHeight(96)
        self.files_list.setToolTip("Uploaded files are parsed and added to the council context. Double-click an item to remove it.")
        attachment_layout.addLayout(file_btn_row)
        attachment_layout.addWidget(self.files_list)

        workspace_header = QtWidgets.QHBoxLayout()
        chat_title = QtWidgets.QLabel("Council Feed")
        chat_title.setObjectName("SectionTitle")
        self.attachment_help_label = QtWidgets.QLabel("Upload supporting files or drag them into the attachment area before you send.")
        self.attachment_help_label.setObjectName("HintLabel")
        self.attachment_help_label.setWordWrap(True)
        workspace_header.addWidget(chat_title)
        workspace_header.addStretch(1)
        workspace_layout.addWidget(attachment_group)
        workspace_layout.addLayout(workspace_header)
        workspace_layout.addWidget(self.attachment_help_label)

        self.run_banner = QtWidgets.QFrame()
        self.run_banner.setObjectName("ComposerCard")
        run_banner_layout = QtWidgets.QHBoxLayout(self.run_banner)
        run_banner_layout.setContentsMargins(12, 10, 12, 10)
        run_banner_layout.setSpacing(10)
        self.run_banner_badge = QtWidgets.QLabel("Ready")
        self.run_banner_badge.setObjectName("StatusBadge")
        self.run_banner_text = QtWidgets.QLabel(
            "Workflow: connect a provider, load and select models, compose your prompt, then run the council."
        )
        self.run_banner_text.setObjectName("StatusText")
        self.run_banner_text.setWordWrap(True)
        run_banner_layout.addWidget(self.run_banner_badge, 0)
        run_banner_layout.addWidget(self.run_banner_text, 1)
        workspace_layout.addWidget(self.run_banner)

        self.chat_view = self._create_output_view()
        workspace_layout.addWidget(self.chat_view, stretch=1)

        composer = QtWidgets.QFrame()
        composer.setObjectName("ComposerCard")
        composer_layout = QtWidgets.QVBoxLayout(composer)
        composer_layout.setContentsMargins(12, 12, 12, 12)
        composer_layout.setSpacing(8)
        if _UI_AVAILABLE:
            self.prompt_edit = EnhancedPromptEditor()
        else:
            self.prompt_edit = PromptEditor()
        self.composer_hint_label = QtWidgets.QLabel(
            "Enter sends immediately. Shift+Enter adds a new line. Use Ctrl+Shift+A to select every loaded model."
        )
        self.composer_hint_label.setObjectName("HintLabel")
        if _UI_AVAILABLE:
            self.composer_hint_label.hide()  # EnhancedPromptEditor has its own hint
        composer_actions = QtWidgets.QHBoxLayout()
        composer_actions.setSpacing(8)
        composer_actions.addStretch(1)
        self.send_btn = QtWidgets.QPushButton("&Run council")
        self.send_btn.setObjectName("PrimaryButton")
        self.stop_btn = QtWidgets.QPushButton("S&top run")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        composer_actions.addWidget(self.send_btn)
        composer_actions.addWidget(self.stop_btn)
        composer_layout.addWidget(self.prompt_edit)
        composer_layout.addWidget(self.composer_hint_label)
        composer_layout.addLayout(composer_actions)
        workspace_layout.addWidget(composer)
        main_splitter.addWidget(workspace_group)

        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        results_layout.setContentsMargins(12, 16, 12, 12)
        results_layout.setSpacing(10)
        right_header = QtWidgets.QHBoxLayout()
        self.tabs_title = QtWidgets.QLabel("Per-Model Answers")
        self.tabs_title.setObjectName("SectionTitle")
        self.results_hint_label = QtWidgets.QLabel("Run a council round to populate model outputs, scoring, and exports.")
        self.results_hint_label.setObjectName("HintLabel")
        self.results_stage_label = QtWidgets.QLabel("Idle")
        self.results_stage_label.setObjectName("StatusBadge")
        right_header.addWidget(self.tabs_title)
        right_header.addStretch(1)
        right_header.addWidget(self.results_stage_label)
        self.replay_last_btn = QtWidgets.QPushButton("&Replay last")
        self.export_json_btn = QtWidgets.QPushButton("Export &JSON")
        right_header.addWidget(self.replay_last_btn)
        right_header.addWidget(self.export_json_btn)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        results_layout.addLayout(right_header)
        results_layout.addWidget(self.results_hint_label)
        results_layout.addWidget(self.tabs, stretch=1)
        self._init_results_tabs()
        main_splitter.addWidget(results_group)

        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 3)
        main_splitter.setSizes([360, 520, 520])
        layout.addWidget(main_splitter, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(10)
        if _UI_AVAILABLE:
            self._animated_status = AnimatedStatusBar()
            self.busy = self._animated_status.progress
            self.status_badge = self._animated_status.badge
            self.status_label = self._animated_status.message
            self.footer_link = self._animated_status.footer_link
            bottom.addWidget(self._animated_status, 1)
        else:
            self._animated_status = None
            self.busy = QtWidgets.QProgressBar()
            self.busy.setTextVisible(False)
            self.busy.setMaximum(0)
            self.busy.setFixedWidth(120)
            self.busy.setVisible(False)
            self.status_badge = QtWidgets.QLabel("Idle")
            self.status_badge.setObjectName("StatusBadge")
            self.status_label = QtWidgets.QLabel("Ready.")
            self.status_label.setObjectName("StatusText")
            self.status_label.setWordWrap(True)
            self.footer_link = QtWidgets.QLabel('<a href="https://github.com/TrentPierce">Trent Pierce · GitHub</a>')
            self.footer_link.setOpenExternalLinks(True)
            bottom.addWidget(self.busy, stretch=0)
            bottom.addWidget(self.status_badge, stretch=0)
            bottom.addWidget(self.status_label, stretch=1)
            bottom.addWidget(self.footer_link, stretch=0)
        layout.addLayout(bottom)

    def _create_output_view(self) -> QtWidgets.QTextBrowser:
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setReadOnly(True)
        view.setUndoRedoEnabled(False)
        view.setFrameShape(QtWidgets.QFrame.NoFrame)
        view.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        view.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        view.document().setDocumentMargin(12)
        self._apply_output_view_theme(view)
        return view

    def _create_form_field(
        self,
        label_text: str,
        control: QtWidgets.QWidget,
        *,
        helper_text: str = "",
        trailing: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        if trailing is None:
            layout.addWidget(control)
        else:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(control, 1)
            row.addWidget(trailing, 0)
            layout.addLayout(row)
        if helper_text:
            helper = QtWidgets.QLabel(helper_text)
            helper.setObjectName("HintLabel")
            helper.setWordWrap(True)
            layout.addWidget(helper)
        return wrapper

    def _init_results_tabs(self):
        self.tabs.clear()
        self.model_tabs.clear()
        self.model_texts.clear()

        self.results_overview_view = self._create_output_view()
        self.results_winner_view = self._create_output_view()
        self.results_ballots_view = self._create_output_view()
        self.results_discussion_view = self._create_output_view()

        self.selected_model_list = QtWidgets.QListWidget()
        self.selected_model_list.setMaximumWidth(240)
        self.selected_model_list.currentTextChanged.connect(self._refresh_selected_model_detail)
        self.selected_model_detail_view = self._create_output_view()
        selected_model_page = QtWidgets.QWidget()
        selected_model_layout = QtWidgets.QHBoxLayout(selected_model_page)
        selected_model_layout.setContentsMargins(0, 0, 0, 0)
        selected_model_layout.setSpacing(10)
        selected_model_layout.addWidget(self.selected_model_list, 0)
        selected_model_layout.addWidget(self.selected_model_detail_view, 1)

        self.inline_log_view = QtWidgets.QPlainTextEdit()
        self.inline_log_view.setReadOnly(True)
        self.inline_log_view.setMaximumBlockCount(self.log_history_limit)

        self.tabs.addTab(self.results_overview_view, "Overview")
        self.tabs.addTab(self.results_winner_view, "Winner")
        self.tabs.addTab(self.results_ballots_view, "Ballots")
        self.tabs.addTab(selected_model_page, "Selected Model")
        self.tabs.addTab(self.results_discussion_view, "Discussion")
        self.tabs.addTab(self.inline_log_view, "Logs")

        self._current_answers: Dict[str, str] = {}
        self._current_tally: Dict[str, Any] = {}
        self._current_details: Dict[str, Any] = {}
        self._current_winner: str = ""
        self._current_question: str = ""

    def _apply_window_style(self):
        if self._theme_engine:
            self._theme_engine.apply(self)
            self._refresh_output_document_styles()
            self._set_badge_tone(self.status_badge, "neutral")
            return
        # Fallback: original hardcoded QSS
        self.setStyleSheet(
            """
            QWidget#Root {
                background: palette(window);
            }
            QFrame#HeroCard, QFrame#ComposerCard, QGroupBox {
                background: palette(base);
                border: 1px solid palette(midlight);
                border-radius: 14px;
            }
            QGroupBox {
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 4px;
            }
            QLabel#HeroTitle {
                letter-spacing: 0.4px;
            }
            QLabel#HeroSubtitle, QLabel#HintLabel, QLabel#StatusText {
                color: palette(mid);
            }
            QLabel#SectionTitle {
                font-weight: 700;
            }
            QLabel#MetricValue {
                font-weight: 700;
                min-width: 40px;
            }
            QLabel#InfoBadge, QLabel#StatusBadge {
                border-radius: 11px;
                padding: 5px 10px;
                font-weight: 600;
                background: #243447;
                color: #f6fbff;
                border: 1px solid #31485f;
            }
            QLabel#StatusBadge[tone="busy"] {
                background: #1f3e63;
                border: 1px solid #3f73af;
            }
            QLabel#StatusBadge[tone="success"] {
                background: #1f5133;
                border: 1px solid #3a8f57;
            }
            QLabel#StatusBadge[tone="warn"] {
                background: #5f4918;
                border: 1px solid #af8a35;
            }
            QLabel#StatusBadge[tone="error"] {
                background: #612626;
                border: 1px solid #b94b4b;
            }
            QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QScrollArea, QComboBox, QSpinBox {
                border-radius: 10px;
                border: 1px solid palette(midlight);
                padding: 6px 8px;
                background: palette(base);
            }
            QListWidget {
                padding: 8px;
            }
            QPlainTextEdit {
                padding: 10px;
            }
            QPushButton {
                border-radius: 10px;
                padding: 7px 12px;
                min-height: 34px;
            }
            QPushButton#PrimaryButton {
                background: #1463a0;
                color: white;
                border: 1px solid #0d4e7f;
                font-weight: 700;
            }
            QPushButton#SecondaryButton {
                background: palette(alternate-base);
            }
            QPushButton#DangerButton {
                background: #7c2d2d;
                color: white;
                border: 1px solid #602020;
                font-weight: 700;
            }
            QPushButton:hover {
                border-color: #5a8ec8;
            }
            QTabWidget::pane {
                border: 1px solid palette(midlight);
                border-radius: 12px;
                top: -1px;
            }
            QTabBar::tab {
                padding: 8px 12px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QTabBar::tab:selected {
                background: palette(base);
                font-weight: 700;
            }
            QProgressBar {
                border-radius: 8px;
                border: 1px solid palette(midlight);
                background: palette(alternate-base);
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: #2d7dd2;
            }
            QSplitter::handle {
                background: transparent;
            }
            """
        )
        self._refresh_output_document_styles()
        self._set_badge_tone(self.status_badge, "neutral")

    def _set_badge_tone(self, badge: QtWidgets.QLabel, tone: str):
        badge.setProperty("tone", tone)
        badge.style().unpolish(badge)
        badge.style().polish(badge)
        badge.update()

    def _palette_hex(self, role: QtGui.QPalette.ColorRole) -> str:
        return self.palette().color(role).name()

    def _is_dark_palette(self) -> bool:
        return self.palette().color(QtGui.QPalette.Window).lightness() < 128

    def _surface_tokens(self) -> Dict[str, str]:
        if self._theme_engine:
            t = self._theme_engine.tokens
            return {
                "text_primary": t.text_primary,
                "text_secondary": t.text_secondary,
                "text_muted": t.text_muted,
                "border": t.border,
                "panel": t.bg_elevated,
                "panel_alt": t.bg_tertiary,
                "panel_subtle": t.bg_secondary,
                "accent": t.accent,
                "accent_muted": t.accent_muted,
                "danger": t.danger,
                "danger_bg": t.danger_bg,
                "success": t.success,
                "success_bg": t.success_bg,
            }
        return {
            "text_primary": self._palette_hex(QtGui.QPalette.Text),
            "text_secondary": self._palette_hex(QtGui.QPalette.Mid),
            "text_muted": self._palette_hex(QtGui.QPalette.Mid),
            "border": self._palette_hex(QtGui.QPalette.Midlight),
            "panel": self._palette_hex(QtGui.QPalette.Base),
            "panel_alt": self._palette_hex(QtGui.QPalette.AlternateBase),
            "panel_subtle": self._palette_hex(QtGui.QPalette.Window),
            "accent": "#2d7dd2",
            "accent_muted": self._palette_hex(QtGui.QPalette.AlternateBase),
            "danger": "#b94b4b",
            "danger_bg": "#ffe8e8",
            "success": "#2e7d32",
            "success_bg": "#e8f5e9",
        }

    def _output_document_css(self) -> str:
        c = self._surface_tokens()
        return f"""
        body {{
            background: transparent;
            color: {c["text_primary"]};
            font-family: "Segoe UI";
            font-size: 13px;
            line-height: 1.5;
        }}
        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 10px;
            color: {c["text_primary"]};
        }}
        li {{
            margin-bottom: 4px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {c["text_primary"]};
            margin-top: 0;
            margin-bottom: 8px;
        }}
        a {{
            color: {c["accent"]};
        }}
        code {{
            color: {c["text_primary"]};
            background: {c["panel_subtle"]};
            border-radius: 6px;
            padding: 1px 4px;
        }}
        pre {{
            white-space: pre-wrap;
            color: {c["text_primary"]};
            background: {c["panel_subtle"]};
            border: 1px solid {c["border"]};
            border-radius: 10px;
            padding: 10px 12px;
            margin: 0 0 12px 0;
        }}
        blockquote {{
            margin: 0 0 12px 0;
            padding: 0 0 0 12px;
            border-left: 3px solid {c["border"]};
            color: {c["text_secondary"]};
        }}
        table {{
            width: 100%;
        }}
        td {{
            vertical-align: top;
        }}
        """

    def _apply_output_view_theme(self, view: Optional[QtWidgets.QTextBrowser]):
        if view is None:
            return
        view.document().setDefaultStyleSheet(self._output_document_css())

    def _refresh_output_document_styles(self):
        for attr in (
            "chat_view",
            "results_overview_view",
            "results_winner_view",
            "results_ballots_view",
            "results_discussion_view",
            "selected_model_detail_view",
            "persona_preview",
        ):
            widget = getattr(self, attr, None)
            if isinstance(widget, QtWidgets.QTextBrowser):
                self._apply_output_view_theme(widget)

    def _safe_markdown_html(self, text: str) -> str:
        if markdown_to_safe_html:
            return markdown_to_safe_html(text)
        safe = escape_text(text) if escape_text else text
        return f"<p>{safe}</p>"

    def _placeholder_html(self, title: str, message: str) -> str:
        colors = self._surface_tokens()
        return f"""
        <div style="border:1px solid {colors['border']}; border-radius:16px; padding:18px; background:{colors['panel_subtle']};">
            <div style="font-size:16px; font-weight:700; color:{colors['text_primary']}; margin-bottom:6px;">{title}</div>
            <div style="color:{colors['text_secondary']}; line-height:1.5;">{message}</div>
        </div>
        """

    def _refresh_header_badges(self):
        selected_count = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        file_count = len(self.uploaded_files)
        self.mode_badge.setText("Discussion" if self.mode == "discussion" else "Deliberation")
        self.selection_badge.setText(
            f"{selected_count} model{'s' if selected_count != 1 else ''} selected"
        )
        self.attachment_badge.setText(
            "No attachments" if file_count == 0 else f"{file_count} attachment{'s' if file_count != 1 else ''}"
        )
        web_text = "Web on" if self.web_search_check.isChecked() else "Web off"
        self.tool_badge.setText(web_text)

    def _refresh_selection_summary(self):
        total_models = len(self.models)
        selected_models = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        visible_models = sum(
            1 for widget in self.model_rows.values()
            if widget.isVisible()
        ) if self.model_rows else total_models
        self.model_selection_label.setText(
            f"{selected_models} selected of {total_models} loaded"
            + (f" · {visible_models} visible" if total_models else "")
        )
        self._refresh_header_badges()

    def _refresh_attachment_summary(self):
        count = len(self.uploaded_files)
        if count == 0:
            self.attachment_help_label.setText(
                "Upload supporting files or drag them into the attachment area before you send."
            )
        else:
            self.attachment_help_label.setText(
                f"{count} attachment{'s' if count != 1 else ''} ready to include in the next run."
            )
        self.remove_file_btn.setEnabled(count > 0)
        self._refresh_header_badges()

    def _set_context_status(self, text: str, tone: Optional[str] = None):
        self.run_banner_text.setText(text)
        self.results_stage_label.setText(text if len(text) <= 28 else text[:25] + "...")
        inferred = tone or self._status_tone or "neutral"
        self._set_badge_tone(self.run_banner_badge, inferred)
        self._set_badge_tone(self.results_stage_label, inferred)
        badge_text = {
            "busy": "Running",
            "success": "Ready",
            "warn": "Attention",
            "error": "Error",
            "neutral": "Ready",
        }.get(inferred, "Ready")
        self.run_banner_badge.setText(badge_text)

    def _refresh_selected_model_detail(self):
        model_id = self.selected_model_list.currentItem().text() if self.selected_model_list.currentItem() else ""
        if not model_id:
            self.selected_model_detail_view.setHtml(
                self._placeholder_html(
                    "Selected Model",
                    "Choose a model from the list to inspect its provider, latency, score, persona assignment, and answer.",
                )
            )
            return
        answer = self._current_answers.get(model_id, "")
        score = self._current_tally.get(model_id, "n/a")
        persona = self.persona_assignments.get(model_id, "None")
        provider_text = self._model_badge_text(model_id)
        esc = escape_text if escape_text else (lambda value: str(value))
        colors = self._surface_tokens()
        detail_html = f"""
        <div style="font-size:20px; font-weight:700; margin-bottom:6px;">{esc(short_id(model_id))}</div>
        <div style="margin-bottom:10px; color:{colors['text_secondary']};">{esc(provider_text)}</div>
        <div style="margin-bottom:16px;"><strong>Score:</strong> {esc(str(score))}<br><strong>Persona:</strong> {esc(persona)}</div>
        {self._safe_markdown_html(answer or "_No response returned._")}
        """
        self.selected_model_detail_view.setHtml(detail_html)

    def _set_results_empty_state(self):
        self.results_hint_label.setText("Run a council round to populate model outputs, scoring, and exports.")
        self._init_results_tabs()
        placeholder = self._placeholder_html(
            "No Results Yet",
            "Load models, select the ones you want to compare, then send a prompt to generate answers and voting details.",
        )
        self.results_overview_view.setHtml(placeholder)
        self.results_winner_view.setHtml(
            self._placeholder_html("Winner", "The winning answer appears here after a deliberation run.")
        )
        self.results_ballots_view.setHtml(
            self._placeholder_html("Ballots", "Weighted voting and ballot notes appear here after a deliberation run.")
        )
        self.results_discussion_view.setHtml(
            self._placeholder_html("Discussion", "Collaborative discussion transcripts appear here in discussion mode.")
        )
        self.inline_log_view.clear()
        self.selected_model_list.clear()
        self._current_answers.clear()
        self._current_tally.clear()
        self._current_details.clear()
        self._current_winner = ""
        self._current_question = ""
        self._refresh_selected_model_detail()

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

    def _setup_settings_dock(self):
        if build_workspace_panel:
            panel = build_workspace_panel(self, create_output_view=self._create_output_view)
            self.utility_dock = panel.dock
            self.utility_tabs = panel.tabs
            self.settings_debug_check = panel.settings_debug_check
            self.settings_personas_check = panel.settings_personas_check
            self.settings_storage_label = panel.settings_storage_label
            self.settings_shortcuts_btn = panel.settings_shortcuts_btn
            self.settings_issue_btn = panel.settings_issue_btn
            self.persona_search_edit = panel.persona_search_edit
            self.persona_library_list = panel.persona_library_list
            self.persona_preview = panel.persona_preview
            self.persona_add_btn = panel.persona_add_btn
            self.persona_edit_btn = panel.persona_edit_btn
            self.persona_delete_btn = panel.persona_delete_btn
        else:
            self.utility_dock = QtWidgets.QDockWidget("Workspace Panel", self)
            self.utility_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
            self.utility_dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable
            )

            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            self.utility_tabs = QtWidgets.QTabWidget()
            self.utility_tabs.setDocumentMode(True)

            settings_page = QtWidgets.QWidget()
            settings_layout = QtWidgets.QVBoxLayout(settings_page)
            settings_layout.setContentsMargins(8, 8, 8, 8)
            settings_layout.setSpacing(12)
            settings_intro = QtWidgets.QLabel(
                "Settings are applied immediately. Use this panel for stable app preferences and support links."
            )
            settings_intro.setObjectName("HintLabel")
            settings_intro.setWordWrap(True)
            settings_layout.addWidget(settings_intro)

            self.settings_debug_check = QtWidgets.QCheckBox("Enable debug logs")
            self.settings_personas_check = QtWidgets.QCheckBox("Show persona controls in the workflow")
            self.settings_storage_label = QtWidgets.QLabel("")
            self.settings_storage_label.setObjectName("HintLabel")
            self.settings_storage_label.setWordWrap(True)
            self.settings_shortcuts_btn = QtWidgets.QPushButton("Keyboard Shortcuts")
            self.settings_shortcuts_btn.setObjectName("SecondaryButton")
            self.settings_issue_btn = QtWidgets.QPushButton("Report an Issue")
            self.settings_issue_btn.setObjectName("SecondaryButton")

            settings_layout.addWidget(self.settings_debug_check)
            settings_layout.addWidget(self.settings_personas_check)
            settings_layout.addWidget(self.settings_storage_label)
            settings_layout.addWidget(self.settings_shortcuts_btn)
            settings_layout.addWidget(self.settings_issue_btn)
            settings_layout.addStretch(1)

            personas_page = QtWidgets.QWidget()
            personas_layout = QtWidgets.QVBoxLayout(personas_page)
            personas_layout.setContentsMargins(8, 8, 8, 8)
            personas_layout.setSpacing(10)
            personas_intro = QtWidgets.QLabel(
                "Manage the persona library here. Assign personas from the model list in the workflow."
            )
            personas_intro.setObjectName("HintLabel")
            personas_intro.setWordWrap(True)
            self.persona_search_edit = QtWidgets.QLineEdit()
            self.persona_search_edit.setPlaceholderText("Filter persona library")
            self.persona_library_list = QtWidgets.QListWidget()
            self.persona_preview = self._create_output_view()
            persona_actions = QtWidgets.QHBoxLayout()
            self.persona_add_btn = QtWidgets.QPushButton("Add Persona")
            self.persona_edit_btn = QtWidgets.QPushButton("Edit Persona")
            self.persona_delete_btn = QtWidgets.QPushButton("Delete Persona")
            persona_actions.addWidget(self.persona_add_btn)
            persona_actions.addWidget(self.persona_edit_btn)
            persona_actions.addWidget(self.persona_delete_btn)
            persona_actions.addStretch(1)
            personas_layout.addWidget(personas_intro)
            personas_layout.addWidget(self.persona_search_edit)
            personas_layout.addWidget(self.persona_library_list, 1)
            personas_layout.addLayout(persona_actions)
            personas_layout.addWidget(self.persona_preview, 1)

            self.utility_tabs.addTab(settings_page, "Settings")
            self.utility_tabs.addTab(personas_page, "Personas")
            layout.addWidget(self.utility_tabs)
            self.utility_dock.setWidget(container)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.utility_dock)
        self.utility_dock.hide()
        self.utility_dock.visibilityChanged.connect(self._settings_dock_visibility_changed)

        self.settings_debug_check.toggled.connect(self._debug_toggled)
        self.settings_personas_check.toggled.connect(self._settings_personas_toggled)
        self.settings_shortcuts_btn.clicked.connect(self._toggle_shortcut_overlay)
        self.settings_issue_btn.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://github.com/TrentPierce/PolyCouncil/issues"))
        )
        self.persona_search_edit.textChanged.connect(self._refresh_persona_library_panel)
        self.persona_library_list.currentTextChanged.connect(self._update_persona_preview_panel)
        self.persona_add_btn.clicked.connect(self._add_persona_from_panel)
        self.persona_edit_btn.clicked.connect(self._edit_selected_persona_from_panel)
        self.persona_delete_btn.clicked.connect(self._delete_selected_persona_from_panel)

    def _connect_signals(self):
        self.connect_btn.clicked.connect(self._connect_base)
        self.replace_models_btn.clicked.connect(self._replace_models_clicked)
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.api_service_combo.currentIndexChanged.connect(self._api_service_changed)
        self.api_key_inline.editingFinished.connect(self._api_key_changed)
        self.show_key_check.toggled.connect(
            lambda checked: self.api_key_inline.setEchoMode(QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password)
        )
        self.add_provider_btn.clicked.connect(self._add_current_provider_profile)
        self.refresh_models_btn.clicked.connect(self._refresh_models_clicked)
        self.select_all_btn.clicked.connect(self._select_all_models)
        self.clear_btn.clicked.connect(self._clear_models)
        self.clear_model_list_btn.clicked.connect(self._clear_model_list)
        self.reset_btn.clicked.connect(self._reset_leaderboard_clicked)
        self.send_btn.clicked.connect(self._send)
        self.prompt_edit.submitRequested.connect(self._send)
        self.stop_btn.clicked.connect(self._stop_process)
        self.conc_spin.valueChanged.connect(self._concurrency_changed)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        self.replay_last_btn.clicked.connect(self._replay_last_session)
        self.export_json_btn.clicked.connect(self._export_session_json)
        
        # New signal connections
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.upload_btn.clicked.connect(self._upload_file)
        self.remove_file_btn.clicked.connect(self._remove_selected_files)
        self.clear_files_btn.clicked.connect(self._clear_files)
        self.web_search_check.toggled.connect(self._web_search_toggled)
        self.model_filter_edit.textChanged.connect(self._filter_model_rows)
        self.files_list.filesDropped.connect(self._handle_dropped_files)
        self.files_list.itemDoubleClicked.connect(self._remove_file_item)

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
        # Ctrl+Enter to send from anywhere in the window.
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

        focus_prompt_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        focus_prompt_shortcut.activated.connect(self.prompt_edit.setFocus)

        focus_filter_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        focus_filter_shortcut.activated.connect(self.model_filter_edit.setFocus)

        # Ctrl+? to toggle keyboard shortcut overlay
        if _UI_AVAILABLE:
            help_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+/"), self)
            help_shortcut.activated.connect(self._toggle_shortcut_overlay)

    # --- New UI helpers ---

    def _show_onboarding(self):
        """Show the first-run onboarding overlay."""
        if not _UI_AVAILABLE:
            return
        overlay = OnboardingOverlay(self)
        overlay.dismissed.connect(self._onboarding_dismissed)
        overlay.show()
        overlay.raise_()

    def _onboarding_dismissed(self):
        """Mark onboarding as complete in settings."""
        s = load_settings()
        s["onboarding_complete"] = True
        save_settings(s)

    def _toggle_shortcut_overlay(self):
        """Toggle the keyboard shortcut help overlay."""
        if self._shortcut_overlay:
            self._shortcut_overlay.toggle()

    def _show_toast(self, message: str, tone: str = "info", duration_ms: int = 3500):
        """Show a non-blocking toast notification over the main window."""
        if not _UI_AVAILABLE:
            return
        toast = ToastNotification(self, message, tone=tone, duration_ms=duration_ms)
        toast.show_toast()

    def _settings_personas_toggled(self, checked: bool):
        self.use_roles = bool(checked)
        save_settings({"roles_enabled": self.use_roles})
        self._update_persona_combo_visibility()
        self._refresh_settings_panel()
        self._refresh_header_badges()

    def _refresh_settings_panel(self):
        if not hasattr(self, "settings_debug_check"):
            return
        self.settings_debug_check.blockSignals(True)
        self.settings_personas_check.blockSignals(True)
        self.settings_debug_check.setChecked(self.debug_enabled)
        self.settings_personas_check.setChecked(self.use_roles)
        self.settings_debug_check.blockSignals(False)
        self.settings_personas_check.blockSignals(False)
        if getattr(self, "secure_keyring_available", False):
            self.settings_storage_label.setText("Secure storage is available for hosted API keys on this machine.")
        else:
            self.settings_storage_label.setText(
                "Secure storage is unavailable. Hosted API keys persist only for the current session."
            )

    def _refresh_persona_library_panel(self):
        if not hasattr(self, "persona_library_list"):
            return
        query = self.persona_search_edit.text().strip().lower() if hasattr(self, "persona_search_edit") else ""
        current_name = self.persona_library_list.currentItem().text() if self.persona_library_list.currentItem() else ""
        self.persona_library_list.clear()
        for persona in self.personas:
            name = persona["name"]
            prompt = persona.get("prompt") or ""
            if query and query not in name.lower() and query not in prompt.lower():
                continue
            item = QtWidgets.QListWidgetItem(name)
            if persona.get("builtin", False):
                item.setForeground(QtGui.QColor("#666"))
            self.persona_library_list.addItem(item)
        if current_name:
            matches = self.persona_library_list.findItems(current_name, QtCore.Qt.MatchExactly)
            if matches:
                self.persona_library_list.setCurrentItem(matches[0])
        if not self.persona_library_list.currentItem() and self.persona_library_list.count():
            self.persona_library_list.setCurrentRow(0)
        self._update_persona_preview_panel()

    def _update_persona_preview_panel(self):
        if not hasattr(self, "persona_preview"):
            return
        item = self.persona_library_list.currentItem() if hasattr(self, "persona_library_list") else None
        if not item:
            self.persona_preview.setHtml(
                self._placeholder_html("Persona Preview", "Select a persona to inspect its prompt and assignment behavior.")
            )
            return
        persona = self._persona_by_name(item.text())
        if not persona:
            self.persona_preview.setHtml(self._placeholder_html("Persona Preview", "Persona not found."))
            return
        prompt = persona.get("prompt") or "No prompt configured."
        assignment_count = sum(1 for assigned in self.persona_assignments.values() if assigned == persona["name"])
        colors = self._surface_tokens()
        self.persona_preview.setHtml(
            f"<div style='font-size:18px; font-weight:700; margin-bottom:6px;'>{escape_text(persona['name']) if escape_text else persona['name']}</div>"
            f"<div style='margin-bottom:10px; color:{colors['text_secondary']};'>"
            f"{'Built-in' if persona.get('builtin', False) else 'Custom'} persona · Assigned to {assignment_count} model(s)</div>"
            f"{self._safe_markdown_html(prompt)}"
        )

    def _prompt_for_persona_text(self, title: str, current_prompt: str = "") -> Optional[str]:
        prompt_dialog = QtWidgets.QDialog(self)
        prompt_dialog.setWindowTitle(title)
        prompt_dialog.resize(560, 360)
        layout = QtWidgets.QVBoxLayout(prompt_dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(QtWidgets.QLabel("System prompt"))
        text_edit = QtWidgets.QPlainTextEdit()
        text_edit.setPlainText(current_prompt)
        layout.addWidget(text_edit)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(prompt_dialog.accept)
        buttons.rejected.connect(prompt_dialog.reject)
        layout.addWidget(buttons)
        if prompt_dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return text_edit.toPlainText().strip()

    def _add_persona_from_panel(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New Persona", "Persona name:")
        name = name.strip()
        if not ok or not name:
            return
        if name in self._persona_names():
            QtWidgets.QMessageBox.warning(self, "Duplicate Name", f"Persona '{name}' already exists.")
            return
        prompt = self._prompt_for_persona_text("Persona System Prompt")
        if prompt is None:
            return
        persona_id = f"u_{uuid.uuid4().hex[:8]}"
        self.personas.append({"name": name, "prompt": prompt if prompt else None, "builtin": False})
        try:
            user_personas = json.loads(USER_PERSONAS_PATH.read_text(encoding="utf-8")) if USER_PERSONAS_PATH.exists() else []
            user_personas.append({"id": persona_id, "name": name, "prompt_instruction": prompt if prompt else ""})
            USER_PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
            USER_PERSONAS_PATH.write_text(json.dumps(user_personas, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"Error saving user persona: {e}")
        self._sort_personas_inplace()
        self._save_persona_state()
        self._refresh_persona_combos()
        self._refresh_persona_library_panel()

    def _edit_selected_persona_from_panel(self):
        item = self.persona_library_list.currentItem() if hasattr(self, "persona_library_list") else None
        if not item:
            QtWidgets.QMessageBox.information(self, "No Selection", "Select a persona to edit.")
            return
        name = item.text()
        persona = self._persona_by_name(name)
        if not persona:
            return
        if persona.get("builtin", False):
            QtWidgets.QMessageBox.information(self, "Built-in Persona", "Built-in personas cannot be edited.")
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Edit Persona", "Persona name:", text=name)
        new_name = new_name.strip()
        if not ok or not new_name:
            return
        if new_name != name and new_name in self._persona_names():
            QtWidgets.QMessageBox.warning(self, "Duplicate Name", f"Persona '{new_name}' already exists.")
            return
        prompt = self._prompt_for_persona_text("Persona System Prompt", persona.get("prompt") or "")
        if prompt is None:
            return
        persona["name"] = new_name
        persona["prompt"] = prompt if prompt else None
        try:
            if USER_PERSONAS_PATH.exists():
                user_personas = json.loads(USER_PERSONAS_PATH.read_text(encoding="utf-8"))
                for entry in user_personas:
                    if entry.get("name") == name:
                        entry["name"] = new_name
                        entry["prompt_instruction"] = prompt if prompt else ""
                        break
                USER_PERSONAS_PATH.write_text(json.dumps(user_personas, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"Error updating user persona: {e}")
        if name != new_name:
            for model, assigned in list(self.persona_assignments.items()):
                if assigned == name:
                    self.persona_assignments[model] = new_name
        self._sort_personas_inplace()
        self._save_persona_state()
        self._refresh_persona_combos()
        self._refresh_persona_library_panel()

    def _delete_selected_persona_from_panel(self):
        item = self.persona_library_list.currentItem() if hasattr(self, "persona_library_list") else None
        if not item:
            QtWidgets.QMessageBox.information(self, "No Selection", "Select a persona to delete.")
            return
        name = item.text()
        persona = self._persona_by_name(name)
        if not persona:
            return
        if persona.get("builtin", False) or name == "None":
            QtWidgets.QMessageBox.information(self, "Cannot Delete", "Built-in personas cannot be deleted.")
            return
        if QtWidgets.QMessageBox.question(
            self, "Delete Persona", f"Delete persona '{name}'?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.personas = [p for p in self.personas if p["name"] != name]
        try:
            if USER_PERSONAS_PATH.exists():
                user_personas = json.loads(USER_PERSONAS_PATH.read_text(encoding="utf-8"))
                user_personas = [entry for entry in user_personas if entry.get("name") != name]
                USER_PERSONAS_PATH.write_text(json.dumps(user_personas, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"Error deleting user persona: {e}")
        for model, assigned in list(self.persona_assignments.items()):
            if assigned == name:
                self.persona_assignments[model] = "None"
        self._save_persona_state()
        self._refresh_persona_combos()
        self._refresh_persona_library_panel()

    def _provider_type_from_ui(self) -> str:
        label = self.provider_combo.currentText().strip()
        for provider_type, provider_name in PROVIDER_LABELS.items():
            if provider_name == label:
                return provider_type
        return PROVIDER_LM_STUDIO

    def _api_service_from_ui(self) -> str:
        label = self.api_service_combo.currentText().strip()
        for service, service_name in API_SERVICE_LABELS.items():
            if service_name == label:
                return service
        return API_SERVICE_CUSTOM

    def _current_provider_config(self) -> ProviderConfig:
        self.api_key = self.api_key_inline.text().strip() if hasattr(self, "api_key_inline") else self.api_key
        return make_provider_config(
            provider_type=self.provider_type,
            base_url=self.base_edit.text().strip(),
            api_key=self.api_key,
            model_path=self.model_path,
            api_service=self.api_service,
        )

    def _provider_changed(self, _index: int):
        self.api_key = self.api_key_inline.text().strip() if hasattr(self, "api_key_inline") else self.api_key
        new_provider = self._provider_type_from_ui()
        old_provider = getattr(self, "provider_type", PROVIDER_LM_STUDIO)
        self.provider_type = new_provider
        default_base, _ = provider_defaults(new_provider)
        if not self.base_edit.text().strip():
            self.base_edit.setText(default_base)
        elif old_provider != new_provider:
            old_default, _ = provider_defaults(old_provider)
            if self.base_edit.text().strip() == old_default:
                self.base_edit.setText(default_base)
        if self.provider_type != PROVIDER_OPENAI_COMPAT:
            self.api_service = API_SERVICE_CUSTOM
            _, default_model_path = provider_defaults(self.provider_type)
            self.model_path = default_model_path
            self.api_service_combo.blockSignals(True)
            self.api_service_combo.setCurrentText(api_service_label(self.api_service))
            self.api_service_combo.blockSignals(False)
        else:
            if self.api_service == API_SERVICE_CUSTOM:
                self.api_service = API_SERVICE_OPENAI
                self.api_service_combo.blockSignals(True)
                self.api_service_combo.setCurrentText(api_service_label(self.api_service))
                self.api_service_combo.blockSignals(False)
            self._apply_api_service_preset_if_needed(force=False)
        self._update_provider_ui()
        save_settings({
            "provider_type": self.provider_type,
            "base_url": self.base_edit.text().strip(),
            "api_key": self.api_key,
            "model_path": self.model_path,
            "api_service": self.api_service,
        })
        self._set_status(f"Provider set to {provider_label(self.provider_type)}")

    def _apply_api_service_preset_if_needed(self, force: bool):
        if self.provider_type != PROVIDER_OPENAI_COMPAT:
            return
        service = normalize_api_service(self.api_service)
        if service == API_SERVICE_CUSTOM:
            return
        preset = service_preset(service)
        current_base = self.base_edit.text().strip()
        current_model_path = (self.model_path or "").strip()
        if force or not current_base:
            self.base_edit.setText(preset["base_url"])
        if force or not current_model_path:
            self.model_path = preset["model_path"]

    def _api_service_changed(self, _index: int):
        self.api_key = self.api_key_inline.text().strip() if hasattr(self, "api_key_inline") else self.api_key
        selected_service = self._api_service_from_ui()
        self.api_service = selected_service
        if selected_service != API_SERVICE_CUSTOM:
            if self.provider_type != PROVIDER_OPENAI_COMPAT:
                self.provider_combo.blockSignals(True)
                self.provider_combo.setCurrentText(provider_label(PROVIDER_OPENAI_COMPAT))
                self.provider_combo.blockSignals(False)
                self.provider_type = PROVIDER_OPENAI_COMPAT
            self._apply_api_service_preset_if_needed(force=True)
        self._update_provider_ui()
        save_settings({
            "provider_type": self.provider_type,
            "base_url": self.base_edit.text().strip(),
            "api_key": self.api_key,
            "model_path": self.model_path,
            "api_service": self.api_service,
        })

    def _api_key_changed(self):
        self.api_key = self.api_key_inline.text().strip()
        persisted = save_settings({"api_key": self.api_key})
        if self.api_key and not persisted:
            self._set_status("API key updated for this session. Install/configure a secure keychain backend to persist it.")

    def _update_provider_ui(self):
        hosted = self.provider_type == PROVIDER_OPENAI_COMPAT
        custom_hosted = hosted and self.api_service == API_SERVICE_CUSTOM

        if hasattr(self, "api_service_field"):
            self.api_service_field.setVisible(hosted)
        self.api_service_label.setVisible(hosted)
        self.api_service_combo.setVisible(hosted)
        self.api_service_combo.setEnabled(hosted)

        if hasattr(self, "api_key_inline") and self.api_key_inline.parentWidget():
            self.api_key_inline.parentWidget().setVisible(hosted)
        self.api_key_inline_label.setVisible(hosted)
        self.api_key_inline.setVisible(hosted)
        self.show_key_check.setVisible(hosted)

        if hosted:
            self.base_label.setText("API Base URL:")
            self.base_edit.setEnabled(custom_hosted)
            self.base_edit.setToolTip("Set custom endpoint only when API Service is Custom.")
        else:
            self.base_label.setText("Base URL:")
            self.base_edit.setEnabled(True)
            self.base_edit.setToolTip("")

    def _current_profile_payload(self) -> Dict[str, str]:
        provider = self._current_provider_config()
        return {
            "id": str(uuid.uuid4()),
            "provider_type": provider.provider_type,
            "api_service": provider.api_service,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "model_path": provider.model_path,
        }

    def _profile_summary(self, profile: Dict[str, str]) -> str:
        provider_text = provider_label(profile.get("provider_type", PROVIDER_LM_STUDIO))
        service = normalize_api_service(profile.get("api_service", API_SERVICE_CUSTOM))
        if profile.get("provider_type") == PROVIDER_OPENAI_COMPAT and service != API_SERVICE_CUSTOM:
            provider_text = f"{provider_text} / {api_service_label(service)}"
        base = profile.get("base_url", "")
        has_key = "key set" if profile.get("api_key", "") else "no key"
        return f"{provider_text} - {base} ({has_key})"

    def _save_provider_profiles(self):
        save_settings({"provider_profiles": self.provider_profiles})

    def _load_profile_into_controls(self, profile: Dict[str, str]):
        self.provider_type = normalize_provider_type(profile.get("provider_type", PROVIDER_LM_STUDIO))
        self.api_service = normalize_api_service(profile.get("api_service", API_SERVICE_CUSTOM))
        self.model_path = profile.get("model_path", "")
        self.api_key = profile.get("api_key", "")
        self.provider_combo.blockSignals(True)
        self.provider_combo.setCurrentText(provider_label(self.provider_type))
        self.provider_combo.blockSignals(False)
        self.api_service_combo.blockSignals(True)
        self.api_service_combo.setCurrentText(api_service_label(self.api_service))
        self.api_service_combo.blockSignals(False)
        self.base_edit.setText(profile.get("base_url", ""))
        self.api_key_inline.setText(self.api_key)
        self._provider_changed(self.provider_combo.currentIndex())
        self._api_service_changed(self.api_service_combo.currentIndex())

    def _profile_use_clicked(self, profile_id: str):
        for profile in self.provider_profiles:
            if profile.get("id") == profile_id:
                self._load_profile_into_controls(profile)
                self._set_status("Provider loaded from saved list.")
                return

    def _profile_add_models_clicked(self, profile_id: str):
        for profile in self.provider_profiles:
            if profile.get("id") == profile_id:
                self._load_profile_into_controls(profile)
                self._add_provider_models_clicked()
                return

    def _profile_remove_clicked(self, profile_id: str):
        self.provider_profiles = [p for p in self.provider_profiles if p.get("id") != profile_id]
        if not self.provider_profiles:
            self.provider_profiles = [self._current_profile_payload()]
        self._save_provider_profiles()
        self._render_provider_profiles()

    def _add_current_provider_profile(self):
        payload = self._current_profile_payload()
        for existing in self.provider_profiles:
            same = (
                existing.get("provider_type") == payload["provider_type"]
                and existing.get("api_service") == payload["api_service"]
                and existing.get("base_url") == payload["base_url"]
                and existing.get("model_path") == payload["model_path"]
                and existing.get("api_key", "") == payload["api_key"]
            )
            if same:
                self._set_status("Provider already saved.")
                return
        self.provider_profiles.append(payload)
        self._save_provider_profiles()
        self._render_provider_profiles()
        self._set_status("Provider saved.")

    def _render_provider_profiles(self):
        if not hasattr(self, "providers_layout"):
            return
        if clear_layout:
            clear_layout(self.providers_layout)
        else:
            while self.providers_layout.count():
                item = self.providers_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                del item
        self.provider_profile_rows.clear()
        for profile in self.provider_profiles:
            pid = profile.get("id", "")
            if build_provider_profile_row:
                row = build_provider_profile_row(
                    summary=self._profile_summary(profile),
                    profile_id=pid,
                    on_use=self._profile_use_clicked,
                    on_load=self._profile_add_models_clicked,
                    on_remove=self._profile_remove_clicked,
                )
            else:
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(2, 2, 2, 2)
                row_layout.setSpacing(6)
                label = QtWidgets.QLabel(self._profile_summary(profile))
                use_btn = QtWidgets.QPushButton("Use")
                add_btn = QtWidgets.QPushButton("Load")
                rm_btn = QtWidgets.QPushButton("Remove")
                use_btn.clicked.connect(lambda _=False, x=pid: self._profile_use_clicked(x))
                add_btn.clicked.connect(lambda _=False, x=pid: self._profile_add_models_clicked(x))
                rm_btn.clicked.connect(lambda _=False, x=pid: self._profile_remove_clicked(x))
                row_layout.addWidget(label, 1)
                row_layout.addWidget(use_btn)
                row_layout.addWidget(add_btn)
                row_layout.addWidget(rm_btn)
            self.providers_layout.addWidget(row)
            self.provider_profile_rows[pid] = row
        self.providers_layout.addStretch(1)

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
        self._refresh_settings_panel()
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
        self._refresh_header_badges()

    def _open_settings_dialog(self):
        if self.utility_dock.isVisible():
            self.utility_dock.hide()
            return
        self._refresh_settings_panel()
        self._refresh_persona_library_panel()
        self.utility_tabs.setCurrentIndex(0)
        self.utility_dock.show()
        self.utility_dock.raise_()

    def _settings_dock_visibility_changed(self, visible: bool):
        self.settings_btn.blockSignals(True)
        self.settings_btn.setChecked(bool(visible))
        self.settings_btn.blockSignals(False)

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
        provider = self._current_provider_config()
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and not provider.api_key:
            self._set_status("Add an API key before loading hosted provider models.")
            return
        save_settings({
            "provider_type": provider.provider_type,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "model_path": provider.model_path,
            "api_service": provider.api_service,
        })
        self._set_status(f"Loading models from {provider_label(provider.provider_type)} at {provider.base_url} …")
        self._busy(True)
        self._refresh_models(append=True)

    def _replace_models_clicked(self):
        provider = self._current_provider_config()
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and not provider.api_key:
            self._set_status("Add an API key before loading hosted provider models.")
            return
        self._set_status("Replacing model list from selected provider …")
        self._busy(True)
        self._refresh_models(append=False)

    def _refresh_models_clicked(self):
        self._set_status("Reloading models from the selected provider …")
        self._busy(True)
        self._refresh_models(append=True)

    def _add_provider_models_clicked(self):
        provider = self._current_provider_config()
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and not provider.api_key:
            self._set_status("Add an API key before loading hosted provider models.")
            return
        self._set_status("Loading models from saved provider …")
        self._busy(True)
        self._refresh_models(append=True)

    def _refresh_models(self, append: bool = False):
        try:
            provider = self._current_provider_config()
            if self._model_thread and self._model_thread.isRunning():
                self._set_status("A model load is already in progress. Wait for it to finish or stop it first.")
                return
            self._fetch_append = bool(append)
            self._fetch_provider = provider

            self._model_worker = ModelFetchWorker(provider)
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
        except Exception as e:
            self._busy(False)
            self._set_status(f"Failed to start model refresh: {e}")

    def _provider_tag(self, provider: ProviderConfig) -> str:
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and provider.api_service != API_SERVICE_CUSTOM:
            return api_service_label(provider.api_service)
        return provider_label(provider.provider_type)

    def _make_display_model_name(self, provider: ProviderConfig, raw_model: str) -> str:
        base_name = f"{self._provider_tag(provider)} :: {raw_model}"
        if base_name not in self.model_actual_ids:
            return base_name
        i = 2
        while True:
            candidate = f"{base_name} ({i})"
            if candidate not in self.model_actual_ids:
                return candidate
            i += 1

    def _models_fetched(self, models: List[str]):
        try:
            self._model_thread = None
            self._model_worker = None
            fetched_provider = self._fetch_provider or self._current_provider_config()
            append_mode = bool(self._fetch_append)
            self._fetch_provider = None
            self._fetch_append = False

            if not append_mode:
                self.models = []
                self.model_actual_ids.clear()
                self.model_provider_map.clear()

            added = 0
            for raw_model in models:
                display_name = self._make_display_model_name(fetched_provider, raw_model)
                if display_name in self.model_actual_ids:
                    continue
                self.models.append(display_name)
                self.model_actual_ids[display_name] = raw_model
                self.model_provider_map[display_name] = fetched_provider
                added += 1

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

            if ModelCapabilityDetector and self.models:
                self._detect_model_capabilities()

            self._busy(False)
            self._refresh_selection_summary()
            if not self.models:
                self._set_status("No models found. Check provider settings and connection.")
            else:
                if append_mode:
                    self._set_status(f"Added {added} models. Total available: {len(self.models)}.")
                else:
                    self._set_status(f"Found {len(self.models)} models.")
        except Exception as e:
            self._busy(False)
            self._set_status(f"Failed to merge models: {e}")
    
    def _detect_model_capabilities(self):
        """Detect capabilities for loaded models."""
        if not ModelCapabilityDetector:
            return

        async def detect_all():
            async with aiohttp.ClientSession() as session:
                model_data_cache: Dict[str, dict] = {}
                for model in self.models:
                    provider = self.model_provider_map.get(model, self._current_provider_config())
                    raw_model = self.model_actual_ids.get(model, model)
                    try:
                        model_lower = model.lower()
                        has_vl = "vl" in model_lower
                        if provider.provider_type == PROVIDER_OLLAMA:
                            self.model_capabilities[model] = {
                                "web_search": False,
                                "visual": any(k in raw_model.lower() for k in ["vision", "vl", "llava"]),
                            }
                        else:
                            cache_key = f"{provider.provider_type}|{provider.base_url}|{provider.model_path}|{provider.api_service}|{bool(provider.api_key)}"
                            if cache_key not in model_data_cache:
                                model_data_cache[cache_key] = await ModelCapabilityDetector.fetch_models_data(
                                    session, provider.base_url, provider.api_key, provider.model_path
                                )
                            model_data = model_data_cache[cache_key]
                            web_search = ModelCapabilityDetector.detect_web_search_from_data(model_data, raw_model)
                            visual_api = ModelCapabilityDetector.detect_visual_from_data(model_data, raw_model)
                            visual = visual_api or has_vl
                            self.model_capabilities[model] = {"web_search": web_search, "visual": visual}
                    except Exception:
                        model_lower = model.lower()
                        has_vl = "vl" in model_lower
                        self.model_capabilities[model] = {"web_search": False, "visual": has_vl}

        # Run detection in background
        def worker():
            try:
                asyncio.run(detect_all())
                self.capability_update_signal.emit()
            except Exception:
                pass
        
        threading.Thread(target=worker, daemon=True).start()
    
    @QtCore.Slot()
    def _update_capability_ui(self):
        """Update UI to reflect detected capabilities."""
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
            if self.web_search_check.isChecked():
                self.web_search_check.blockSignals(True)
                self.web_search_check.setChecked(False)
                self.web_search_check.blockSignals(False)
                self.web_search_enabled = False
            self.web_search_check.setEnabled(False)
            self.web_search_check.setToolTip("No web search capability detected.")
        
        # Update file upload button based on visual support
        self._update_file_upload_capabilities(has_visual)
        for model, label in self.model_meta_labels.items():
            label.setText(self._model_badge_text(model))
        self._refresh_header_badges()

    def _models_fetch_failed(self, error: str):
        self._model_thread = None
        self._model_worker = None
        self._busy(False)
        self._refresh_selection_summary()
        self._set_status(f"Model refresh failed: {error}")

    def _model_badge_text(self, model: str) -> str:
        provider = self.model_provider_map.get(model)
        provider_badge = self._provider_tag(provider) if provider else "Unknown"
        caps = self.model_capabilities.get(model, {})
        cap_parts = []
        if caps.get("visual"):
            cap_parts.append("vision")
        if caps.get("web_search"):
            cap_parts.append("web")
        session_timings = (self.last_session_record or {}).get("timings_ms", {}) or {}
        latency_ms = session_timings.get(model)
        details = [provider_badge]
        if cap_parts:
            details.append(", ".join(cap_parts))
        if isinstance(latency_ms, (int, float)):
            details.append(f"{int(latency_ms)} ms")
        return " | ".join(details)

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
        self.model_meta_labels.clear()
        self.model_rows.clear()
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
            assigned = self.persona_assignments.get(m, "None")
            if assigned not in persona_names:
                assigned = "None"
            
            pconfig = self.model_provider_map.get(m)
            pname = provider_label(pconfig.provider_type) if pconfig else "Unknown"
            mtext = self._model_badge_text(m)

            if _UI_AVAILABLE:
                row_widget = ModelCard(m, pname, mtext)
                cb = row_widget.checkbox
                persona_btn = row_widget.persona_btn
                meta_label = row_widget.meta_label
                row_widget.setPersonaText(assigned)
                row_widget.showPersonaButton(self.use_roles)
            else:
                row_widget = QtWidgets.QWidget()
                row_widget.setMinimumHeight(34)
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(6, 4, 6, 4)
                row_layout.setSpacing(8)
    
                cb = QtWidgets.QCheckBox(m)
                cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
                cb.setAccessibleName(f"Model selector {m}")
                
                persona_btn = QtWidgets.QPushButton("Persona")
                persona_btn.setFixedWidth(118)
                persona_btn.setFixedHeight(28)
                persona_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                
                display_text = assigned if assigned != "None" else "Persona"
                if len(display_text) > 12:
                    display_text = display_text[:10] + ".."
                persona_btn.setText(display_text)
                persona_btn.setToolTip(assigned if assigned != "None" else "Select persona")
                
                persona_btn.setVisible(self.use_roles)
                persona_btn.setEnabled(self.use_roles)
                
                row_layout.addWidget(cb, stretch=1)
    
                meta_label = QtWidgets.QLabel(mtext)
                meta_label.setObjectName("HintLabel")
                meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                meta_label.setMinimumWidth(170)
                row_layout.addWidget(meta_label, stretch=0)
    
                row_layout.addWidget(persona_btn, stretch=0)
                row_layout.setAlignment(persona_btn, QtCore.Qt.AlignRight)

            # Ensure button accepts mouse events and is clickable
            persona_btn.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
            persona_btn.setFocusPolicy(QtCore.Qt.StrongFocus)
            persona_btn.setAutoDefault(False)
            persona_btn.setDefault(False)
            persona_btn.raise_()  # Ensure button is on top
            
            # Connect button click to show persona menu - use direct lambda with explicit capture
            model_id_capture = m  
            button_capture = persona_btn  
            persona_btn.clicked.connect(
                lambda checked=False, mid=model_id_capture, btn=button_capture: self._show_persona_menu(mid, btn)
            )

            self.models_layout.insertWidget(self.models_layout.count() - 1, row_widget)
            self.model_checks[m] = cb
            self.model_persona_combos[m] = persona_btn
            self.model_meta_labels[m] = meta_label
            self.model_rows[m] = row_widget
            
            # Connect model selection change to update capabilities
            cb.toggled.connect(self._on_model_selection_changed)

        # Ensure visibility is correct after all buttons are created
        self._update_persona_combo_visibility()
        self._filter_model_rows()
        self._refresh_selection_summary()

    def _select_all_models(self):
        for cb in self.model_checks.values():
            if cb.parentWidget() is None or cb.parentWidget().isVisible():
                cb.setChecked(True)
        self._refresh_selection_summary()

    def _clear_models(self):
        for cb in self.model_checks.values():
            cb.setChecked(False)
        self._refresh_selection_summary()

    def _clear_model_list(self):
        self.models = []
        self.model_actual_ids.clear()
        self.model_provider_map.clear()
        self.model_capabilities.clear()
        self._populate_models()
        self.single_voter_combo.clear()
        self.model_filter_edit.clear()
        self._set_results_empty_state()
        self._set_status("Model list cleared.")

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
        if leaderboard:
            self.lb_title.setText(f"Council performance over time · {len(leaderboard)} tracked model(s)")
        else:
            self.lb_title.setText("Council performance over time · no votes recorded yet")

    def _filter_model_rows(self):
        query = self.model_filter_edit.text().strip().lower()
        for model, row in self.model_rows.items():
            badge = self.model_meta_labels.get(model)
            haystack = f"{model} {badge.text() if badge else ''}".lower()
            row.setVisible((not query) or (query in haystack))
        self._refresh_selection_summary()

    # ----- UI helpers -----
    def _prepare_tabs(self, selected_models: List[str]):
        self._current_answers = {}
        self._current_tally = {}
        self._current_details = {}
        self._current_winner = ""
        self._current_question = ""
        self.selected_model_list.clear()
        for model_id in selected_models:
            self.selected_model_list.addItem(model_id)
        if self.selected_model_list.count():
            self.selected_model_list.setCurrentRow(0)
        self.results_overview_view.setHtml(
            self._placeholder_html(
                "Run In Progress",
                "Weighted voting, ballot notes, answer latencies, and the winning answer will appear here when the council finishes.",
            )
        )
        self.results_winner_view.setHtml(
            self._placeholder_html("Winner", "The highest-ranked answer will appear here when the run finishes.")
        )
        self.results_ballots_view.setHtml(
            self._placeholder_html("Ballots", "Voting notes and per-model scoring will appear here when the run finishes.")
        )
        self.results_discussion_view.setHtml(
            self._placeholder_html("Discussion", "Discussion mode is inactive for this run.")
        )
        self._refresh_selected_model_detail()
        self.results_hint_label.setText("Council run in progress.")
        self.tabs.setCurrentWidget(self.results_overview_view)

    def _append_chat(self, text: str):
        if not text:
            return

        is_dark = self._is_dark_palette()
        colors = self._surface_tokens()
        text_color = colors["text_primary"]
        muted_color = colors["text_secondary"]
        border_color = colors["border"]
        base_color = colors["panel_subtle"]
        alt_color = colors["panel_alt"]
        user_bg = "#1f4f82" if is_dark else "#e8f2ff"
        user_fg = "#f7fbff" if is_dark else text_color
        council_bg = colors["panel"]
        note_bg = "transparent"
        error_bg = colors["danger_bg"]
        style = "margin: 6px 0; padding: 10px 12px; border-radius: 12px;"
        header = ""

        if text.startswith("You:"):
            style += f"background-color: {user_bg}; color: {user_fg}; border: 1px solid {border_color};"
            header = "You"
            body = text[len("You:"):].strip()
        elif text.startswith("[Error]"):
            style += f"background-color: {error_bg}; color: {text_color}; border: 1px solid {colors['danger']};"
            header = "Error"
            body = text[len("[Error]"):].strip()
        elif text.startswith("[Stopped]"):
            style += f"background-color: {alt_color}; color: {text_color}; border: 1px solid {border_color};"
            header = "Stopped"
            body = text[len("[Stopped]"):].strip()
        elif "→" in text:
            style += f"background-color: {council_bg}; color: {text_color}; border: 1px solid {border_color};"
            parts = text.split(":", 1)
            header = parts[0].strip()
            body = parts[1].strip() if len(parts) == 2 else text
        elif text.startswith("<i>") and text.endswith("</i>"):
            style += f"background-color: {note_bg}; color: {muted_color};"
            body = text[3:-4].strip()
            body_html = f"<p><em>{escape_text(body) if escape_text else body}</em></p>"
        else:
            style += f"background-color: {base_color}; border: 1px solid {border_color}; color: {text_color};"
            body = text.strip()

        if not (text.startswith("<i>") and text.endswith("</i>")):
            body_html = self._safe_markdown_html(body)
            if header:
                safe_header = escape_text(header) if escape_text else header
                body_html = f"<p><strong>{safe_header}:</strong></p>{body_html}"

        full_html = f"<div style='{style}'>{body_html}</div>"
        self.chat_view.append(full_html)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def _append_log(self, text: str):
        if not text:
            return
        if self.debug_enabled and not self.log_dock.isVisible():
            self.log_dock.show()
        self.log_view.appendPlainText(text)
        if hasattr(self, "inline_log_view") and self.inline_log_view:
            self.inline_log_view.appendPlainText(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _log_sink_dispatch(self, label: str, message: str):
        if not message:
            return
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        formatted = f"{stamp} [{label}] {message}"
        self.log_signal.emit(formatted)

    def _set_status(self, text: str):
        if self._animated_status:
            self._animated_status.set_status(text)
            self._status_tone = self._animated_status.badge.property("tone") or "neutral"
            self._set_context_status(text, self._status_tone)
            self._refresh_header_badges()
            return
        self.status_label.setText(text)
        lowered = text.lower()
        if any(word in lowered for word in ("error", "failed")):
            tone = "error"
            badge_text = "Error"
        elif any(word in lowered for word in ("warning", "select", "unavailable", "no ")) and not self.busy.isVisible():
            tone = "warn"
            badge_text = "Attention"
        elif self.busy.isVisible():
            tone = "busy"
            badge_text = "Working"
        elif any(word in lowered for word in ("done", "ready", "complete", "saved", "added", "found", "loaded", "updated", "cleared")):
            tone = "success"
            badge_text = "Ready"
        else:
            tone = "neutral"
            badge_text = "Info"
        self._status_tone = tone
        self.status_badge.setText(badge_text)
        self._set_badge_tone(self.status_badge, tone)
        self._set_context_status(text, tone)
        self._refresh_header_badges()

    def _busy(self, on: bool):
        if self._animated_status:
            self._animated_status.set_busy(on)
        else:
            self.busy.setVisible(on)
            self.busy.setMaximum(0 if on else 1)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.setEnabled(on)
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(not on)
        for attr in (
            "connect_btn",
            "replace_models_btn",
            "refresh_models_btn",
            "add_provider_btn",
            "provider_combo",
            "api_service_combo",
            "base_edit",
            "api_key_inline",
            "mode_combo",
            "single_voter_check",
            "single_voter_combo",
            "upload_btn",
            "remove_file_btn",
            "clear_files_btn",
            "model_filter_edit",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(not on)
        if on:
            self.status_badge.setText("Working")
            self._set_badge_tone(self.status_badge, "busy")
            self._set_context_status("Running council…", "busy")
        else:
            self._refresh_attachment_summary()
            self._set_status(self.status_label.text())

    def _mode_changed(self, index: int):
        """Handle mode selection change."""
        self.mode = "discussion" if index == 1 else "deliberation"
        # Update UI based on mode
        if self.mode == "discussion":
            self.tabs_title.setText("Discussion View")
            # Hide single voter controls in discussion mode
            self.single_voter_check.setVisible(False)
            self.single_voter_combo.setVisible(False)
            self.leaderboard_group.setVisible(False)
            self.mode_help_label.setText(
                "Collaborative discussion runs multiple turns and produces a synthesized report instead of a weighted vote."
            )
            self.prompt_edit.setPlaceholderText(
                "Start a collaborative discussion. Press Enter to send, Shift+Enter for a new line."
            )
        else:
            self.tabs_title.setText("Per-Model Answers")
            self.single_voter_check.setVisible(True)
            self.single_voter_combo.setVisible(True)
            self.leaderboard_group.setVisible(True)
            self.mode_help_label.setText(
                "Weighted deliberation compares model answers, scores them against the rubric, and picks a winner."
            )
            self.prompt_edit.setPlaceholderText(
                "Ask the council a question. Press Enter to send, Shift+Enter for a new line."
            )
        self._refresh_header_badges()
    
    def _current_has_visual_support(self) -> bool:
        selected_models = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if selected_models:
            has_visual = any(
                self.model_capabilities.get(model, {}).get("visual", False)
                for model in selected_models
            )
            return has_visual or any("vl" in model.lower() for model in selected_models)
        has_visual = any(caps.get("visual", False) for caps in self.model_capabilities.values())
        return has_visual or any("vl" in model.lower() for model in self.models)
    
    def _update_file_upload_capabilities(self, has_visual: bool):
        """Update file upload button and tooltip based on visual support."""
        if has_visual:
            self.upload_btn.setToolTip("Upload documents or images (PDF, TXT, DOCX, JPG, PNG, etc.)")
        else:
            self.upload_btn.setToolTip("Upload documents only (PDF, TXT, DOCX) - No visual models selected")
        self.remove_file_btn.setEnabled(bool(self.uploaded_files))

    def _file_item_text(self, path: Path) -> str:
        kind = "image" if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"} else "document"
        try:
            size = human_file_size(path.stat().st_size)
        except OSError:
            size = "size unavailable"
        return f"{path.name} · {kind} · {size}"

    def _add_uploaded_path(self, path: Path):
        if path in self.uploaded_files:
            return False
        self.uploaded_files.append(path)
        item = QtWidgets.QListWidgetItem(self._file_item_text(path))
        item.setData(QtCore.Qt.UserRole, str(path))
        item.setToolTip(str(path))
        self.files_list.addItem(item)
        return True

    def _handle_candidate_paths(self, candidate_paths: Iterable[Path]):
        if FileParser is None:
            QtWidgets.QMessageBox.warning(
                self,
                "File Parsing Unavailable",
                "File parsing libraries are not available. Install the document dependencies listed in requirements.txt.",
            )
            return

        has_visual = self._current_has_visual_support()
        document_suffixes = {".txt", ".pdf", ".docx", ".doc"}
        image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        added = 0
        rejected: List[str] = []
        for path in candidate_paths:
            suffix = path.suffix.lower()
            if suffix in image_suffixes and not has_visual:
                rejected.append(f"{path.name} (image requires a visual-capable model)")
                continue
            if suffix not in document_suffixes and suffix not in image_suffixes:
                rejected.append(f"{path.name} (unsupported file type)")
                continue
            if self._add_uploaded_path(path):
                added += 1

        self._refresh_attachment_summary()
        self._update_file_upload_capabilities(has_visual)
        if added:
            self._set_status(f"Added {added} attachment{'s' if added != 1 else ''}.")
        if rejected:
            QtWidgets.QMessageBox.information(
                self,
                "Some Files Were Skipped",
                "The following files were not attached:\n\n" + "\n".join(rejected),
            )

    def _upload_file(self):
        """Handle file upload."""
        has_visual = self._current_has_visual_support()

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

        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Files", "", file_filter
        )

        if file_paths:
            self._handle_candidate_paths(Path(p) for p in file_paths)
    
    def _on_model_selection_changed(self):
        """Update UI when model selection changes."""
        # Update capability indicators based on selected models
        self._update_capability_ui()
        self._refresh_selection_summary()
    
    def _clear_files(self):
        """Clear uploaded files."""
        self.uploaded_files.clear()
        self.files_list.clear()
        self._refresh_attachment_summary()
        self._set_status("Files cleared")

    def _remove_selected_files(self):
        items = self.files_list.selectedItems()
        if not items:
            self._set_status("Select one or more attachments to remove.")
            return
        for item in items:
            self._remove_file_item(item)

    def _remove_file_item(self, item: QtWidgets.QListWidgetItem):
        path_value = item.data(QtCore.Qt.UserRole)
        path = Path(path_value) if path_value else None
        if path and path in self.uploaded_files:
            self.uploaded_files.remove(path)
        row = self.files_list.row(item)
        self.files_list.takeItem(row)
        self._refresh_attachment_summary()
        self._set_status("Attachment removed.")

    def _handle_dropped_files(self, paths: List[str]):
        self._handle_candidate_paths(Path(p) for p in paths)
    
    def _web_search_toggled(self, checked: bool):
        """Handle web search toggle."""
        self.web_search_enabled = checked
        self._refresh_header_badges()
    
    def _stop_process(self):
        """Handle stop button click."""
        self._stop_requested = True
        self._set_status("Stopping...")
        self._busy(False)
        self._append_chat("[Stopped] Process cancelled by user.")
        self._set_status("Stopped.")
    
    def _send(self):
        selected = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            self._set_status("Select at least one model.")
            return
        question = self.prompt_edit.toPlainText().strip()
        if not question:
            self._set_status("Ask something first.")
            return
        self.last_submitted_question = question

        # Parse uploaded files for context and images
        context_block = ""
        image_urls: List[str] = []
        
        if self.uploaded_files:
            context_parts = []
            max_file_size = 50000  # Limit each file to 50k chars before parsing
            for file_path in self.uploaded_files:
                suffix = file_path.suffix.lower()
                # Check for images
                if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    try:
                        if ModelCapabilityDetector:
                            encoded_image = ModelCapabilityDetector.encode_image(file_path)
                            if encoded_image:
                                image_urls.append(encoded_image)
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
            self._send_discussion_mode(selected, question, context_block, image_urls)
        else:
            self._send_deliberation_mode(selected, question, context_block, image_urls)
    
    def _send_deliberation_mode(
        self,
        selected: List[str],
        question: str,
        context_block: str,
        images: Optional[List[str]] = None,
    ):
        """Send in Deliberation Mode (existing functionality)."""
        images = images or []
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

        # Web search status
        web_search = self.web_search_check.isChecked()
        model_entries = []
        for model_id in selected:
            model_entries.append({
                "id": model_id,
                "model": self.model_actual_ids.get(model_id, model_id),
                "provider": self.model_provider_map.get(model_id, self._current_provider_config()),
            })

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
                        model_entries, question, roles_map, self.status_signal.emit,
                        max_concurrency=conc, voter_override=voters,
                        images=images, web_search=web_search,
                        rubric_weights=self.rubric_weights,
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
    
    def _send_discussion_mode(
        self,
        selected: List[str],
        question: str,
        context_block: str,
        images: Optional[List[str]] = None,
    ):
        """Send in Collaborative Discussion Mode."""
        if DiscussionManager is None:
            QtWidgets.QMessageBox.warning(self, "Discussion Mode Unavailable", 
                "Discussion mode requires core modules. Please check installation.")
            return
        images = images or []
        
        # Build agent configurations
        agents = []
        for model in selected:
            persona_name = self.persona_assignments.get(model, "None")
            persona_config = self._build_persona_config(model, persona_name)
            provider = self.model_provider_map.get(model, self._current_provider_config())
            
            agent = {
                "name": f"Agent {model[:20]}",
                "model": self.model_actual_ids.get(model, model),
                "is_active": True,
                "persona_config": persona_config,
                "persona_name": persona_name if persona_name != "None" else None,
                "provider_type": provider.provider_type,
                "base_url": provider.base_url,
                "api_key": provider.api_key,
                "model_path": provider.model_path,
            }
            agents.append(agent)
        
        # Prepare UI for discussion mode
        self._prepare_discussion_tabs(selected)
        self._append_chat(f"You: {question}")
        if images:
            self._append_chat(f"<i>[Attached {len(images)} image(s)]</i>")
        self.prompt_edit.clear()
        self._set_status("Starting collaborative discussion…")

        conc = int(self.conc_spin.value())
        web_search = self.web_search_check.isChecked()

        def worker():
            try:
                self._stop_requested = False
                base_provider = self.model_provider_map.get(selected[0], self._current_provider_config())
                manager = DiscussionManager(
                    provider_type=base_provider.provider_type,
                    base_url=base_provider.base_url,
                    api_key=base_provider.api_key,
                    model_path=base_provider.model_path,
                    agents=agents,
                    user_prompt=question,
                    context_block=context_block,
                    images=images,
                    web_search_enabled=web_search,
                    status_callback=self.status_signal.emit,
                    update_callback=lambda entry: self.discussion_update_signal.emit(entry),
                    max_turns=10,
                    max_concurrency=conc,
                    is_cancelled=lambda: self._stop_requested
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
        self.selected_model_list.clear()
        for model_id in selected:
            self.selected_model_list.addItem(model_id)
        if self.selected_model_list.count():
            self.selected_model_list.setCurrentRow(0)
        self.results_overview_view.setHtml(
            self._placeholder_html(
                "Discussion Starting",
                "Collaborative discussion agents are being prepared. The overview will update when the synthesis is ready.",
            )
        )
        self.results_winner_view.setHtml(
            self._placeholder_html("Winner", "Discussion mode does not produce a weighted-vote winner.")
        )
        self.results_ballots_view.setHtml(
            self._placeholder_html("Ballots", "Discussion mode does not use the ballot tab.")
        )
        self.results_discussion_view.setHtml(
            self._placeholder_html(
                "Discussion Standby",
                "Agents will post their turns here as the collaborative discussion progresses.",
            )
        )
        self.discussion_transcript = []
        self._refresh_selected_model_detail()
        self.results_hint_label.setText("Discussion is preparing.")
        self.tabs.setCurrentWidget(self.results_discussion_view)
    
    def _update_discussion_view(self, entry: Dict):
        """Update the discussion view in real-time as each agent responds."""
        if not hasattr(self, 'results_discussion_view') or not self.results_discussion_view:
            return
        
        # Add entry to transcript
        self.discussion_transcript.append(entry)
        self.results_discussion_view.setHtml(
            result_presenter.build_discussion_transcript_html(
                transcript=self.discussion_transcript,
                colors=self._surface_tokens(),
                safe_markdown_html=self._safe_markdown_html,
                placeholder_html=self._placeholder_html,
                escape_text=escape_text,
            )
        )
        self.results_discussion_view.verticalScrollBar().setValue(
            self.results_discussion_view.verticalScrollBar().maximum()
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
                exported_question = self.last_submitted_question if self.last_submitted_question else "N/A"
                f.write(f"**Question:** {exported_question}\n\n")
                
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
                
                synthesis = (self.last_session_record or {}).get("synthesis")
                if synthesis:
                    f.write("## Final Synthesis\n\n")
                    f.write(f"{synthesis}\n")
            
            self._set_status(f"Discussion exported to {Path(file_path).name}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export discussion: {str(e)}")

    def _handle_error(self, msg: str):
        self._busy(False)
        self.results_hint_label.setText("The last run failed. Review the status message and debug log for details.")
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
        self._current_question = question
        self._current_answers = dict(answers)
        self._current_tally = dict(tally)
        self._current_details = dict(details)
        self._current_winner = winner

        # Refresh leaderboard list
        self._refresh_leaderboard()

        timings_ms = details.get("timings_ms", {})
        self.results_overview_view.setHtml(
            result_presenter.build_deliberation_summary_html(
                question=question,
                answers=answers,
                winner=winner,
                details=details,
                tally=tally,
                colors=self._surface_tokens(),
                short_id=short_id,
                safe_markdown_html=self._safe_markdown_html,
                escape_text=escape_text,
                normalize_rubric_weights=normalize_rubric_weights,
            )
        )
        self.results_winner_view.setHtml(
            result_presenter.build_winner_html(
                winner=winner,
                answer=answers.get(winner, ""),
                tally=tally,
                details=details,
                colors=self._surface_tokens(),
                short_id=short_id,
                safe_markdown_html=self._safe_markdown_html,
                escape_text=escape_text,
            )
        )
        self.results_ballots_view.setHtml(
            result_presenter.build_ballots_html(
                answers=answers,
                details=details,
                tally=tally,
                colors=self._surface_tokens(),
                short_id=short_id,
                escape_text=escape_text,
                normalize_rubric_weights=normalize_rubric_weights,
            )
        )
        self.results_discussion_view.setHtml(
            self._placeholder_html("Discussion", "Discussion mode is not active for this result.")
        )
        self.selected_model_list.clear()
        for model_id in answers.keys():
            self.selected_model_list.addItem(model_id)
        preferred_model = winner if winner in answers else (next(iter(answers.keys()), ""))
        if preferred_model:
            matches = self.selected_model_list.findItems(preferred_model, QtCore.Qt.MatchExactly)
            if matches:
                self.selected_model_list.setCurrentItem(matches[0])
            else:
                self.selected_model_list.setCurrentRow(0)
        self._refresh_selected_model_detail()
        self.last_session_record = {
            "mode": "deliberation",
            "question": question,
            "answers": answers,
            "winner": winner,
            "details": details,
            "tally": tally,
            "timings_ms": timings_ms,
        }
        for model, label in self.model_meta_labels.items():
            label.setText(self._model_badge_text(model))
        self.results_hint_label.setText("Latest council result loaded.")
        self.tabs.setCurrentWidget(self.results_overview_view)
        if save_session:
            try:
                session_path = save_session(self.last_session_record)
                self._set_status(f"Done. Session saved to {session_path.name}.")
            except Exception:
                self._set_status("Done.")
        else:
            self._set_status("Done.")

        # Show winning answer in chat
        ans = answers.get(winner, "")
        self._append_chat(f"Council → {short_id(winner)}:\n{ans}")
    
    def _handle_discussion_result(self, payload: object):
        """Handle result from Collaborative Discussion Mode."""
        mode, question, transcript, synthesis = payload
        self.results_overview_view.setHtml(
            result_presenter.build_discussion_report_html(
                question=question,
                transcript=transcript,
                synthesis=synthesis,
                colors=self._surface_tokens(),
                safe_markdown_html=self._safe_markdown_html,
                placeholder_html=self._placeholder_html,
                escape_text=escape_text,
            )
        )
        self.results_winner_view.setHtml(
            self._placeholder_html("Winner", "Discussion mode does not produce a single vote winner.")
        )
        self.results_ballots_view.setHtml(
            self._placeholder_html("Ballots", "Discussion mode does not produce ballots.")
        )
        self.results_discussion_view.setHtml(
            result_presenter.build_discussion_transcript_html(
                transcript=transcript,
                colors=self._surface_tokens(),
                safe_markdown_html=self._safe_markdown_html,
                placeholder_html=self._placeholder_html,
                escape_text=escape_text,
            )
        )
        self.selected_model_list.clear()
        self._refresh_selected_model_detail()
        
        # Show summary in chat
        self._append_chat(f"Discussion complete. {len(transcript)} turns recorded.")
        if synthesis and synthesis.strip():
            # Show full synthesis in chat (no truncation)
            self._append_chat(f"Final Synthesis:\n{synthesis}")
        elif synthesis is None:
            self._append_chat("Note: Synthesis generation failed. Check console for details.")
        self.last_session_record = {
            "mode": "discussion",
            "question": question,
            "transcript": transcript,
            "synthesis": synthesis,
            "timings_ms": {},
        }
        self.results_hint_label.setText("Latest discussion result loaded.")
        self.tabs.setCurrentWidget(self.results_overview_view)
        if save_session:
            try:
                session_path = save_session(self.last_session_record)
                self._set_status(f"Discussion complete. Session saved to {session_path.name}.")
            except Exception:
                self._set_status("Discussion complete.")
        else:
            self._set_status("Discussion complete.")

    def _replay_last_session(self):
        if not load_session:
            self._set_status("Session history is unavailable in this build.")
            return
        try:
            record = load_session()
        except Exception as exc:
            self._set_status(f"Failed to load session history: {exc}")
            return
        if not record:
            self._set_status("No saved session history found.")
            return
        self.last_session_record = record
        if record.get("mode") == "discussion":
            self.mode_combo.setCurrentIndex(1)
            self._prepare_discussion_tabs(self.models)
            self._handle_discussion_result(("discussion", record.get("question", ""), record.get("transcript", []), record.get("synthesis")))
        else:
            self.mode_combo.setCurrentIndex(0)
            answers = record.get("answers", {}) or {}
            selected_models = list(answers.keys())
            if selected_models:
                self._prepare_tabs(selected_models)
            self._handle_deliberation_result((
                record.get("question", ""),
                answers,
                record.get("winner", ""),
                record.get("details", {}) or {},
                record.get("tally", {}) or {},
            ))

    def _export_session_json(self):
        if not self.last_session_record:
            self._set_status("No session data available to export.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Session JSON",
            "polycouncil-session.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        try:
            Path(file_path).write_text(
                json.dumps(self.last_session_record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._set_status(f"Session exported to {Path(file_path).name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export session JSON: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            provider = self._current_provider_config()
            save_settings({
                "provider_type": provider.provider_type,
                "base_url": provider.base_url,
                "api_key": provider.api_key,
                "model_path": provider.model_path,
                "api_service": provider.api_service,
                "provider_profiles": self.provider_profiles,
                "single_voter_enabled": self.single_voter_check.isChecked(),
                "single_voter_model": self.single_voter_combo.currentText().strip(),
                "max_concurrency": int(self.conc_spin.value()),
                "roles_enabled": bool(self.use_roles),
                "rubric_weights": self.rubric_weights,
            })
        except Exception:
            pass
        super().closeEvent(event)

# -----------------------
# Entry point
# -----------------------
def main():
    import sys
    import platform
    sys.excepthook = log_unhandled_exception
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
