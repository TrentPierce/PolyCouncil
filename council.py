
# council_gui_qt.py
# GUI: PySide6 (Qt widgets)
# Deps: PySide6, aiohttp
# Optional: qdarktheme (auto dark/light), lmstudio (local-only "loaded models" detection)

import asyncio
import aiohttp
import sqlite3
import datetime
import json
import logging
import threading
import re
import random
import uuid
import traceback
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Any, Iterable, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from constants import (
    ACTIONS,
    FONT_BASE,
    FONT_LG,
    FONT_SM,
    FONT_XL,
    HEADING_FONT_FAMILY,
    INTERNAL_GAP,
    MIN_H,
    MIN_SIDEBAR,
    MIN_W,
    PADDING_LG,
    PADDING_MD,
    PADDING_SM,
    SECTION_GAP,
    STRINGS,
    THEME,
    dp,
    make_font,
    refresh_dpi_scale,
)
from ui.factories import make_text_browser

# Import new core modules
try:
    from core.tool_manager import FileParser, ModelCapabilityDetector
    from core.api_client import ModelFetchError, call_model, fetch_models
    from core.discussion_manager import DiscussionManager
    from core import leaderboard as leaderboard_store
    from core import result_presenter
    from core.personas import (
        DEFAULT_PERSONAS_PATH,
        USER_PERSONAS_PATH,
        add_user_persona,
        assignment_count,
        build_persona_config,
        cleanup_persona_assignments,
        clear_persona_assignment,
        merge_persona_library,
        persona_by_name,
        persona_names,
        persona_prompt,
        rename_persona_assignments,
        sort_personas_inplace,
        update_user_persona,
        delete_user_persona,
    )
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
    leaderboard_store = None
    result_presenter = None
    DEFAULT_PERSONAS_PATH = None
    USER_PERSONAS_PATH = None
    add_user_persona = None
    assignment_count = None
    build_persona_config = None
    cleanup_persona_assignments = None
    clear_persona_assignment = None
    merge_persona_library = None
    persona_by_name = None
    persona_names = None
    persona_prompt = None
    rename_persona_assignments = None
    sort_personas_inplace = None
    update_user_persona = None
    delete_user_persona = None
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

try:
    import qasync  # type: ignore
except Exception:
    qasync = None

# Import new UI package
try:
    from ui.theme import ThemeEngine
    from ui.animations import FadeIn, set_reduce_motion
    from ui.components import (
        CollapsibleGroupBox,
        ModelCard,
        StatefulRunButton,
        ToastNotification,
        EnhancedPromptEditor,
        AnimatedStatusBar,
        OnboardingOverlay,
        KeyboardShortcutOverlay,
        WorkflowStepCard,
    )
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False
    set_reduce_motion = None

try:
    from gui import (
        apply_persona_visibility,
        build_model_badge_text,
        build_persona_preview_html,
        build_provider_profile_row,
        build_workspace_panel,
        clear_layout,
        DebugTimelineWidget,
        filter_model_rows,
        make_unique_display_model_name,
        populate_model_rows,
        populate_persona_library_list,
    )
except ImportError:
    apply_persona_visibility = None
    build_model_badge_text = None
    build_persona_preview_html = None
    build_provider_profile_row = None
    build_workspace_panel = None
    clear_layout = None
    DebugTimelineWidget = None
    filter_model_rows = None
    make_unique_display_model_name = None
    populate_model_rows = None
    populate_persona_library_list = None

# -----------------------
# Debug logging switches
# -----------------------
DEBUG_VOTING = True                 # set False to silence
LOG_TRUNCATE: Optional[int] = None  # e.g., 8000 to cap output, or None for full
LOG_SINK: Optional[Callable[[str, str], None]] = None
LOGGER = logging.getLogger(__name__)

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
    LOGGER.debug("===== %s =====\n%s", label, s)


def set_log_sink(callback: Optional[Callable[[str, str], None]]):
    global LOG_SINK
    LOG_SINK = callback


class RightPanelState(str, Enum):
    PRE_RUN = "pre_run"
    IN_RUN = "in_run"
    POST_RUN = "post_run"


class RunState(str, Enum):
    LOCKED = "locked"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"

def create_app_icon(size: int = 256) -> QtGui.QIcon:
    size = max(64, int(size))
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    bg_color = QtGui.QColor(THEME["accent"])
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
    painter.fillPath(bubble_path, QtGui.QColor(THEME["accent_fg"]))

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
DATA_DIR = app_data_dir() if app_data_dir else APP_DIR
LOG_DIR = app_log_dir() if app_log_dir else APP_DIR
DB_PATH = DATA_DIR / "council_stats.db"
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
    timeout_seconds: int = 120,
    is_cancelled: Optional[Callable[[], bool]] = None,
    answer_stream_cb: Optional[Callable[[str, str, bool], None]] = None,
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
                async def on_stream_chunk(chunk: str) -> None:
                    if answer_stream_cb:
                        answer_stream_cb(model_id, chunk, False)

                ans = await call_model(
                    session, provider, raw_model, user_prompt,
                    debug_hook=_dbg,
                    temperature=temperature, sys_prompt=sys_p,
                    images=images, web_search=web_search,
                    timeout_sec=timeout_seconds,
                    stream=bool(answer_stream_cb),
                    stream_callback=on_stream_chunk if answer_stream_cb else None,
                )
                if isinstance(ans, dict):
                    ans = json.dumps(ans, ensure_ascii=False)
                elapsed_ms = int((datetime.datetime.now() - started_at).total_seconds() * 1000)
                if answer_stream_cb:
                    answer_stream_cb(model_id, "", True)
                return model_id, ans, elapsed_ms
            except Exception as e1:
                elapsed_ms = int((datetime.datetime.now() - started_at).total_seconds() * 1000)
                if answer_stream_cb:
                    answer_stream_cb(model_id, "", True)
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
        cancelled = {m: a for m, a, _ in pairs if isinstance(a, str) and a.startswith("[Cancelled]")}

        if errors:
            status_cb(f"⚠ {len(errors)}/{len(model_entries)} models failed to answer: {', '.join(short_id(m) for m in errors)}")

        status_cb("Models are voting…")

        model_entry_by_id = {str(m["id"]): m for m in model_entries}
        successful_model_ids = [
            model_id
            for model_id, answer_text, _elapsed in pairs
            if not (
                isinstance(answer_text, str)
                and (answer_text.startswith("[ERROR") or answer_text.startswith("[Cancelled]"))
            )
        ]
        candidate_model_ids = [model_id for model_id in successful_model_ids if model_id in model_entry_by_id]
        candidate_answers = {model_id: answers[model_id] for model_id in candidate_model_ids}
        candidate_entries = [model_entry_by_id[model_id] for model_id in candidate_model_ids]

        if not candidate_model_ids:
            status_cb("⚠ No successful model answers were returned, so no winner was selected.")
            details = {
                "question": question,
                "answers": answers,
                "valid_votes": {},
                "invalid_votes": {},
                "vote_messages": {},
                "tally": {},
                "errors": errors,
                "cancelled": cancelled,
                "timings_ms": timings_ms,
                "rubric_weights": weights,
                "winner": "",
                "participation_rate": 0.0,
                "voters_used": [],
                "candidate_models": [],
            }
            return answers, "", details, {}

        voters_to_use = (
            [model_id for model_id in voter_override if model_id in candidate_model_ids]
            if voter_override
            else list(candidate_model_ids)
        )
        # Note: Voting phase does not use images or web search, typically.
        valid_votes: Dict[str, dict] = {}
        invalid_votes: Dict[str, str] = {}
        vote_messages: Dict[str, str] = {}

        if len(candidate_model_ids) > 1 and voters_to_use:
            vote_results = await run_limited(
                max_concurrency,
                [
                    lambda m=m: vote_one(session, model_entry_by_id[m], question, candidate_answers, candidate_entries)
                    for m in voters_to_use if m in model_entry_by_id
                ]
            )

            for voter_id, ballot, msg in vote_results:
                vote_messages[voter_id] = msg
                if ballot is not None:
                    valid_votes[voter_id] = ballot
                else:
                    invalid_votes[voter_id] = msg
        else:
            if len(candidate_model_ids) == 1:
                status_cb(f"Only one model returned a usable answer: {short_id(candidate_model_ids[0])}")
            elif not voters_to_use:
                status_cb("⚠ No eligible voters produced a usable answer.")

        if invalid_votes:
            status_cb(f"⚠ {len(invalid_votes)}/{len(voters_to_use)} invalid ballots: {', '.join(short_id(m) for m in invalid_votes)}")

        totals: Dict[str, int] = {mid: 0 for mid in candidate_model_ids}
        for ballot in valid_votes.values():
            for candidate_mid, score_dict in ballot["scores"].items():
                weighted = sum(score_dict[k] * weights[k] for k in weights.keys())
                totals[candidate_mid] = totals.get(candidate_mid, 0) + int(weighted)

        if not valid_votes:
            status_cb("⚠ No valid ballots were produced, selecting the first successful answer by default.")
            winner = candidate_model_ids[0]
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
            "cancelled": cancelled,
            "timings_ms": timings_ms,
            "rubric_weights": weights,
            "winner": winner,
            "participation_rate": len(valid_votes) / max(1, len(voters_to_use)),
            "voters_used": voters_to_use,
            "candidate_models": candidate_model_ids,
        }

        status_cb(f"Vote complete. Valid: {len(valid_votes)}/{len(voters_to_use)}")
        return answers, winner, details, tally

# -----------------------
# Qt GUI
# -----------------------
class ModelFetchWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, provider: ProviderConfig, timeout_seconds: int = 20):
        super().__init__()
        self.provider = provider
        self.timeout_seconds = max(15, int(timeout_seconds))

    @QtCore.Slot()
    def run(self):
        try:
            models = asyncio.run(
                fetch_models(
                    self.provider,
                    provider_label=provider_label,
                    timeout_sec=self.timeout_seconds,
                )
            )
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
        self.setWindowTitle(STRINGS["app_title"])
        self.resize(max(dp(1320), MIN_W), max(dp(900), MIN_H))
        self.setMinimumSize(dp(MIN_W), dp(MIN_H))
        self.app_icon = create_app_icon()
        self.setWindowIcon(self.app_icon)

        self.use_roles = False
        self.debug_enabled = False
        self.timeout_seconds = 120
        self.reduce_motion = False
        self.right_panel_state = RightPanelState.PRE_RUN
        self.run_state = RunState.LOCKED
        self._results_stale = False
        self._stop_requested = False
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

        if leaderboard_store:
            leaderboard_store.ensure_db()
        else:
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
        self._winner_stream_timer: Optional[QtCore.QTimer] = None

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
        self._configure_tab_order()
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
        self.timeout_seconds = int(s.get("timeout_seconds", 120) or 120)
        self.reduce_motion = bool(s.get("reduce_motion", False))
        self.theme_mode = str(s.get("theme_mode", "system") or "system")
        self.provider_type = normalize_provider_type(s.get("provider_type", PROVIDER_LM_STUDIO))
        self.api_key = s.get("api_key", "") or ""
        self.model_path = s.get("model_path", "") or ""
        self.api_service = normalize_api_service(s.get("api_service", API_SERVICE_CUSTOM))
        self.rubric_weights = normalize_rubric_weights(s.get("rubric_weights"))
        if set_reduce_motion:
            set_reduce_motion(self.reduce_motion)
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

        self.personas = merge_persona_library(s.get("personas", [])) if merge_persona_library else []
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
        self._restore_window_state(s)
        if self._theme_engine and self.theme_mode in {"light", "dark"}:
            if self._theme_engine.is_dark != (self.theme_mode == "dark"):
                self._toggle_theme()
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
        layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_MD))
        layout.setSpacing(dp(INTERNAL_GAP))
        self.setCentralWidget(central)

        hero = QtWidgets.QFrame()
        hero.setObjectName("HeroCard")
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(dp(SECTION_GAP), dp(PADDING_LG), dp(SECTION_GAP), dp(PADDING_LG))
        hero_layout.setSpacing(dp(INTERNAL_GAP))

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(dp(PADDING_LG))
        nav.setContentsMargins(0, 0, 0, 0)
        title_block = QtWidgets.QVBoxLayout()
        title_block.setSpacing(2)
        title = QtWidgets.QLabel("PolyCouncil")
        title.setObjectName("HeroTitle")
        title.setFont(make_font(FONT_XL, bold=True, family=HEADING_FONT_FAMILY))
        subtitle = QtWidgets.QLabel(
            "Connect providers, select models, compose once, and review a readable council run without losing the thread."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        mode_stack = QtWidgets.QVBoxLayout()
        mode_stack.setContentsMargins(0, 0, 0, 0)
        mode_stack.setSpacing(4)
        mode_label = QtWidgets.QLabel("RUN MODE")
        mode_label.setObjectName("SectionEyebrow")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Deliberation Mode", "Collaborative Discussion Mode"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setMinimumWidth(dp(220))
        self.settings_btn = QtWidgets.QPushButton("&Settings")
        self.settings_btn.setObjectName("SecondaryButton")
        self.settings_btn.setMinimumWidth(dp(128))
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip("Open or close the workspace settings panel.")
        mode_stack.addWidget(mode_label)
        mode_stack.addWidget(self.mode_combo)

        nav.addLayout(title_block, stretch=1)
        nav.addLayout(mode_stack, stretch=0)
        nav.addWidget(self.settings_btn)
        hero_layout.addLayout(nav)

        badge_row = QtWidgets.QHBoxLayout()
        badge_row.setSpacing(dp(PADDING_MD))
        self.mode_badge = QtWidgets.QLabel("mode: deliberation")
        self.mode_badge.setObjectName("HeaderStatus")
        self.connection_btn = QtWidgets.QPushButton("● No models loaded")
        self.connection_btn.setObjectName("HeaderStatusButton")
        self.connection_btn.setToolTip("Open provider actions and connection status.")
        self.selection_badge = QtWidgets.QLabel("sel: 0")
        self.selection_badge.setObjectName("HeaderStatus")
        self.attachment_badge = QtWidgets.QLabel("files: 0")
        self.attachment_badge.setObjectName("HeaderStatus")
        self.tool_badge = QtWidgets.QLabel("web: off")
        self.tool_badge.setObjectName("HeaderStatus")
        badge_row.addWidget(self.mode_badge)
        badge_row.addWidget(self.connection_btn)
        self.selection_badge.hide()
        self.attachment_badge.hide()
        self.tool_badge.hide()
        badge_row.addStretch(1)
        hero_layout.addLayout(badge_row)
        layout.addWidget(hero)

        shell = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        shell.setChildrenCollapsible(False)
        shell.setObjectName("MainSplitter")

        sidebar_col = QtWidgets.QWidget()
        sidebar_col.setObjectName("SidebarCol")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_col)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(dp(INTERNAL_GAP))

        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setObjectName("PanelScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        sidebar_scroll.setWidget(sidebar_col)
        sidebar_scroll.setMinimumWidth(max(dp(MIN_SIDEBAR), dp(280)))
        sidebar_scroll.setMaximumWidth(dp(320))
        sidebar_scroll.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        workspace_col = QtWidgets.QWidget()
        workspace_col.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        workspace_layout = QtWidgets.QVBoxLayout(workspace_col)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(dp(INTERNAL_GAP))

        results_col = QtWidgets.QWidget()
        results_col.setObjectName("ResultsCol")
        results_layout = QtWidgets.QVBoxLayout(results_col)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(dp(INTERNAL_GAP))

        results_scroll = QtWidgets.QScrollArea()
        results_scroll.setObjectName("PanelScroll")
        results_scroll.setWidgetResizable(True)
        results_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        results_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        results_scroll.setWidget(results_col)
        results_scroll.setMinimumWidth(dp(360))
        results_scroll.setMaximumWidth(dp(420))
        results_scroll.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        self.step_connect = WorkflowStepCard(
            1,
            "Connect",
            "Choose a provider and load models.",
            collapsible=True,
            start_collapsed=False,
        )
        provider_layout = QtWidgets.QGridLayout()
        provider_layout.setContentsMargins(0, 0, 0, 0)
        provider_layout.setHorizontalSpacing(dp(PADDING_MD))
        provider_layout.setVerticalSpacing(dp(INTERNAL_GAP))

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems([
            provider_label(PROVIDER_LM_STUDIO),
            provider_label(PROVIDER_OPENAI_COMPAT),
            provider_label(PROVIDER_OLLAMA),
        ])

        self.api_service_combo = QtWidgets.QComboBox()
        self.api_service_combo.addItems([
            api_service_label(API_SERVICE_CUSTOM),
            api_service_label(API_SERVICE_OPENAI),
            api_service_label(API_SERVICE_OPENROUTER),
            api_service_label(API_SERVICE_GEMINI),
        ])
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
        self.replace_models_btn.setObjectName("SecondaryButton")
        self.replace_models_btn.setToolTip("Clear the current model list and replace it with the selected provider's models.")
        self.connect_btn.setToolTip("Load models from the provider currently shown here and append them to the model list.")
        self.add_provider_btn = QtWidgets.QPushButton("&Save profile")
        self.add_provider_btn.setObjectName("SecondaryButton")

        provider_layout.addWidget(
            self._create_form_field("Provider", self.provider_combo),
            0, 0, 1, 2
        )
        self.api_service_field = self._create_form_field(
            "API Service",
            self.api_service_combo,
            helper_text="Choose a hosted service preset or keep Custom for your own compatible endpoint.",
        )
        provider_layout.addWidget(self.api_service_field, 1, 0, 1, 2)
        provider_layout.addWidget(
            self._create_form_field(
                "Base URL",
                self.base_edit,
                helper_text="Use the local or hosted API base URL for the current provider.",
            ),
            2, 0, 1, 2
        )
        provider_layout.addWidget(
            self._create_form_field(
                "API Key",
                self.api_key_inline,
                helper_text="Keys are stored securely when a Windows keychain backend is available.",
                trailing=self.show_key_check,
            ),
            3, 0, 1, 2
        )
        provider_actions = QtWidgets.QVBoxLayout()
        provider_actions.setContentsMargins(0, 0, 0, 0)
        provider_actions.setSpacing(dp(PADDING_MD))
        provider_actions.addWidget(self.connect_btn)
        secondary_provider_actions = QtWidgets.QHBoxLayout()
        secondary_provider_actions.setContentsMargins(0, 0, 0, 0)
        secondary_provider_actions.setSpacing(dp(PADDING_MD))
        secondary_provider_actions.addWidget(self.replace_models_btn)
        secondary_provider_actions.addWidget(self.add_provider_btn)
        provider_actions.addLayout(secondary_provider_actions)
        provider_layout.addLayout(provider_actions, 4, 0, 1, 2)
        provider_layout.setColumnStretch(0, 1)
        provider_layout.setColumnStretch(1, 1)
        provider_container = QtWidgets.QWidget()
        provider_container.setLayout(provider_layout)
        self.step_connect.add_widget(provider_container)

        profiles_label = QtWidgets.QLabel("Saved provider profiles")
        profiles_label.setObjectName("FieldLabel")
        self.step_connect.add_widget(profiles_label)
        self.providers_scroll = QtWidgets.QScrollArea()
        self.providers_scroll.setObjectName("PanelScroll")
        self.providers_scroll.setWidgetResizable(True)
        self.providers_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.providers_inner = QtWidgets.QWidget()
        self.providers_layout = QtWidgets.QVBoxLayout(self.providers_inner)
        self.providers_layout.setContentsMargins(0, 0, 0, 0)
        self.providers_layout.setSpacing(dp(PADDING_SM))
        self.providers_layout.addStretch(1)
        self.providers_scroll.setWidget(self.providers_inner)
        self.providers_scroll.setMaximumHeight(dp(180))
        self.step_connect.add_widget(self.providers_scroll)
        sidebar_layout.addWidget(self.step_connect)

        self.step_models = WorkflowStepCard(
            2,
            "Select models",
            "Load models to unlock selection.",
            collapsible=True,
            start_collapsed=False,
        )
        models_container = QtWidgets.QWidget()
        models_layout = QtWidgets.QVBoxLayout()
        models_layout.setContentsMargins(0, 0, 0, 0)
        models_layout.setSpacing(dp(INTERNAL_GAP))
        self.model_filter_edit = QtWidgets.QLineEdit()
        self.model_filter_edit.setPlaceholderText("Filter models by provider, capability, or name")
        self.model_selection_label = QtWidgets.QLabel("0 selected of 0 loaded")
        self.model_selection_label.setObjectName("HintLabel")
        model_actions = QtWidgets.QGridLayout()
        model_actions.setContentsMargins(0, 0, 0, 0)
        model_actions.setHorizontalSpacing(dp(PADDING_MD))
        model_actions.setVerticalSpacing(dp(PADDING_MD))
        self.refresh_models_btn = QtWidgets.QPushButton("&Reload provider")
        self.refresh_models_btn.setObjectName("SecondaryButton")
        self.select_all_btn = QtWidgets.QPushButton("Select &All")
        self.clear_btn = QtWidgets.QPushButton("Select &None")
        self.clear_model_list_btn = QtWidgets.QPushButton("Clear Models")
        self.refresh_models_btn.setToolTip("Reload models from the provider currently shown in the provider card.")
        self.clear_model_list_btn.setToolTip("Remove all loaded models from all providers.")
        model_actions.addWidget(self.refresh_models_btn, 0, 0)
        model_actions.addWidget(self.clear_model_list_btn, 0, 1)
        model_actions.addWidget(self.select_all_btn, 1, 0)
        model_actions.addWidget(self.clear_btn, 1, 1)
        self.models_area = QtWidgets.QScrollArea()
        self.models_area.setObjectName("PanelScroll")
        self.models_area.setWidgetResizable(True)
        self.models_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.models_inner = QtWidgets.QWidget()
        self.models_layout = QtWidgets.QVBoxLayout(self.models_inner)
        self.models_layout.setContentsMargins(2, 2, 2, 2)
        self.models_layout.setSpacing(dp(PADDING_SM))
        self.models_layout.addStretch(1)
        self.models_area.setWidget(self.models_inner)
        models_layout.addWidget(self.model_filter_edit)
        models_layout.addWidget(self.model_selection_label)
        models_layout.addLayout(model_actions)
        models_layout.addWidget(self.models_area, stretch=1)
        models_container.setLayout(models_layout)
        self.step_models.add_widget(models_container)
        sidebar_layout.addWidget(self.step_models, 1)

        self.step_history = WorkflowStepCard(
            3,
            "History",
            "Past winners and tracked performance.",
            collapsible=True,
            start_collapsed=True,
        )
        self.leaderboard_group = self.step_history
        leaderboard_container = QtWidgets.QWidget()
        leaderboard_layout = QtWidgets.QVBoxLayout(leaderboard_container)
        leaderboard_layout.setContentsMargins(0, 0, 0, 0)
        leaderboard_layout.setSpacing(dp(INTERNAL_GAP))
        lb_header = QtWidgets.QHBoxLayout()
        self.lb_title = QtWidgets.QLabel("Council performance over time")
        self.lb_title.setObjectName("HintLabel")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setObjectName("SecondaryButton")
        lb_header.addWidget(self.lb_title)
        lb_header.addStretch(1)
        lb_header.addWidget(self.reset_btn)
        self.leader_list = QtWidgets.QListWidget()
        self.leader_list.setObjectName("LeaderboardList")
        self.leader_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.leader_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.leader_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.leader_list.setSpacing(dp(PADDING_SM))
        leaderboard_layout.addLayout(lb_header)
        leaderboard_layout.addWidget(self.leader_list, stretch=1)
        self.step_history.add_widget(leaderboard_container)
        sidebar_layout.addWidget(self.step_history)

        attachment_group, attachment_layout = self._create_panel_card(
            "Context",
            "Attachments",
            "Drop or upload files before you run. They are added to the next council request.",
        )
        attachment_layout.setContentsMargins(0, 0, 0, 0)
        attachment_layout.setSpacing(dp(PADDING_MD))
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
        self.files_list = AttachmentListWidget()
        self.files_list.setMaximumHeight(dp(112))
        self.files_list.setToolTip("Uploaded files are parsed and added to the council context. Double-click an item to remove it.")
        self.attachment_help_label = QtWidgets.QLabel("No attachments loaded yet.")
        self.attachment_help_label.setObjectName("HintLabel")
        self.attachment_help_label.setWordWrap(True)
        attachment_layout.addLayout(file_btn_row)
        attachment_layout.addWidget(self.attachment_help_label)
        attachment_layout.addWidget(self.files_list)
        attachment_layout.addWidget(self.visual_status)
        self.visual_status.hide()

        self.run_banner = QtWidgets.QFrame()
        self.run_banner.setObjectName("ComposerCard")
        run_banner_layout = QtWidgets.QHBoxLayout(self.run_banner)
        run_banner_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_MD), dp(PADDING_LG), dp(PADDING_MD))
        run_banner_layout.setSpacing(dp(INTERNAL_GAP))
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

        feed_card, feed_body = self._create_panel_card(
            "Workspace",
            "Council feed",
            "Active run updates, council notes, and status messages appear here in order.",
        )
        self.chat_view = self._create_output_view()
        feed_body.addWidget(self.chat_view, stretch=1)
        workspace_layout.addWidget(feed_card, stretch=1)

        composer = QtWidgets.QFrame()
        composer.setObjectName("ComposerCard")
        composer_layout = QtWidgets.QVBoxLayout(composer)
        composer_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
        composer_layout.setSpacing(dp(PADDING_MD))
        composer_eyebrow = QtWidgets.QLabel("COMPOSE")
        composer_eyebrow.setObjectName("SectionEyebrow")
        composer_title = QtWidgets.QLabel("Prompt")
        composer_title.setObjectName("PanelTitle")
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
        composer_actions.setSpacing(dp(PADDING_MD))
        composer_actions.addWidget(self.upload_btn)
        composer_actions.addWidget(self.remove_file_btn)
        composer_actions.addWidget(self.clear_files_btn)
        composer_actions.addStretch(1)
        self.run_btn = StatefulRunButton() if _UI_AVAILABLE else QtWidgets.QPushButton("Run council")
        if not _UI_AVAILABLE:
            self.run_btn.setObjectName("PrimaryButton")
            self.run_btn.setEnabled(False)
        self.send_btn = self.run_btn
        composer_actions.addWidget(self.run_btn)
        composer_layout.addWidget(composer_eyebrow)
        composer_layout.addWidget(composer_title)
        composer_layout.addWidget(self.attachment_help_label)
        composer_layout.addWidget(self.files_list)
        composer_layout.addWidget(self.prompt_edit)
        composer_layout.addWidget(self.composer_hint_label)
        composer_layout.addLayout(composer_actions)
        workspace_layout.addWidget(composer)

        run_config_card, run_config_body = self._create_panel_card(
            "Configure",
            "Run configuration",
            "",
        )
        self.single_voter_check = QtWidgets.QCheckBox("Use single voter")
        self.single_voter_combo = QtWidgets.QComboBox()
        self.conc_label = QtWidgets.QLabel("Max concurrent jobs")
        self.conc_spin = QtWidgets.QSpinBox()
        self.conc_spin.setRange(1, 8)
        self.conc_spin.setValue(1)
        self.run_timeout_spin = QtWidgets.QSpinBox()
        self.run_timeout_spin.setRange(15, 600)
        self.run_timeout_spin.setSuffix(" s")
        self.run_timeout_spin.setValue(int(getattr(self, "timeout_seconds", 90)))
        self.web_search_check = QtWidgets.QCheckBox("Enable web search")
        self.web_search_check.setEnabled(False)
        self.web_search_check.setToolTip("Enable only when a selected model exposes web tools.")
        self.mode_help_label = QtWidgets.QLabel(
            "Weighted deliberation compares answers and ballots. Discussion mode runs a collaborative round-table instead."
        )
        self.mode_help_label.setObjectName("HintLabel")
        self.mode_help_label.setWordWrap(True)
        config_grid = QtWidgets.QGridLayout()
        config_grid.setContentsMargins(0, 0, 0, 0)
        config_grid.setHorizontalSpacing(10)
        config_grid.setVerticalSpacing(8)
        self.single_voter_field = self._create_form_field(
            "Voting mode",
            self.single_voter_combo,
            helper_text="Turn this on to let one selected model judge all candidates.",
            trailing=self.single_voter_check,
        )
        config_grid.addWidget(self.mode_help_label, 0, 0, 1, 2)
        config_grid.addWidget(self.single_voter_field, 1, 0, 1, 2)
        config_grid.addWidget(
            self._create_form_field(
                "Concurrency",
                self.conc_spin,
                helper_text="Keep this low on local hardware.",
            ),
            2,
            0,
        )
        config_grid.addWidget(
            self._create_form_field(
                "Timeout",
                self.run_timeout_spin,
                helper_text="Per-request timeout for model calls.",
            ),
            2,
            1,
        )
        config_grid.addWidget(
            self._create_form_field(
                "Tools",
                self.web_search_check,
                helper_text="Use only when selected models support it.",
            ),
            3,
            0,
            1,
            2,
        )
        run_config_body.addLayout(config_grid)

        self.right_panel_stack = QtWidgets.QStackedWidget()

        pre_run_page = QtWidgets.QWidget()
        pre_run_layout = QtWidgets.QVBoxLayout(pre_run_page)
        pre_run_layout.setContentsMargins(0, 0, 0, 0)
        pre_run_layout.setSpacing(12)
        self.pre_run_warning_label = QtWidgets.QLabel("")
        self.pre_run_warning_label.setObjectName("HintLabel")
        self.pre_run_warning_label.setWordWrap(True)
        self.pre_run_warning_label.hide()
        self.pre_run_placeholder = self._create_output_view()
        self.pre_run_placeholder.setHtml(
            self._placeholder_html(
                "Results will appear here",
                "Finish setup, run the council, and this panel will switch from configuration to result review automatically.",
            )
        )
        pre_run_layout.addWidget(run_config_card)
        pre_run_layout.addWidget(self.pre_run_warning_label)
        pre_run_layout.addWidget(self.pre_run_placeholder, 1)

        in_run_page = QtWidgets.QWidget()
        in_run_layout = QtWidgets.QVBoxLayout(in_run_page)
        in_run_layout.setContentsMargins(0, 0, 0, 0)
        in_run_layout.setSpacing(12)
        in_run_card, in_run_body = self._create_panel_card(
            "Running",
            "Council in progress",
            "The active run state, progress, and streamed updates live here until the round completes.",
        )
        self.in_run_status_label = QtWidgets.QLabel("Preparing run…")
        self.in_run_status_label.setObjectName("PanelTitle")
        self.in_run_progress_label = QtWidgets.QLabel("Waiting for model responses.")
        self.in_run_progress_label.setObjectName("HintLabel")
        self.in_run_progress_label.setWordWrap(True)
        self.in_run_live_view = self._create_output_view()
        in_run_body.addWidget(self.in_run_status_label)
        in_run_body.addWidget(self.in_run_progress_label)
        in_run_body.addWidget(self.in_run_live_view, 1)
        in_run_layout.addWidget(in_run_card, 1)

        post_run_page = QtWidgets.QWidget()
        post_run_layout = QtWidgets.QVBoxLayout(post_run_page)
        post_run_layout.setContentsMargins(0, 0, 0, 0)
        post_run_layout.setSpacing(12)
        results_group, results_group_layout = self._create_panel_card(
            "Review",
            "Results",
            "Winner, ballots, selected model detail, and logs stay in one stable review surface.",
        )
        self.tabs_title = QtWidgets.QLabel("Council results")
        self.tabs_title.setObjectName("PanelTitle")
        self.results_hint_label = QtWidgets.QLabel("Run a council round to populate model outputs, scoring, and exports.")
        self.results_hint_label.setObjectName("HintLabel")
        self.results_hint_label.setWordWrap(True)
        self.results_stage_label = QtWidgets.QLabel("Idle")
        self.results_stage_label.setObjectName("StatusBadge")
        results_header = QtWidgets.QHBoxLayout()
        results_header.setContentsMargins(0, 0, 0, 0)
        results_header.setSpacing(8)
        results_header.addWidget(self.tabs_title)
        results_header.addStretch(1)
        results_header.addWidget(self.results_stage_label)
        self.replay_last_btn = QtWidgets.QPushButton("&Replay last")
        self.replay_last_btn.setObjectName("SecondaryButton")
        self.export_json_btn = QtWidgets.QPushButton("Export &JSON")
        self.export_json_btn.setObjectName("SecondaryButton")
        results_actions = QtWidgets.QHBoxLayout()
        results_actions.setContentsMargins(0, 0, 0, 0)
        results_actions.setSpacing(8)
        results_actions.addWidget(self.replay_last_btn)
        results_actions.addWidget(self.export_json_btn)
        results_actions.addStretch(1)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        results_group_layout.addLayout(results_header)
        results_group_layout.addWidget(self.results_hint_label)
        results_group_layout.addLayout(results_actions)
        results_group_layout.addWidget(self.tabs, stretch=1)
        self._init_results_tabs()

        self.run_config_used_collapsible = CollapsibleGroupBox("Run config used", start_collapsed=True) if _UI_AVAILABLE else None
        self.run_config_used_view = self._create_output_view()
        if self.run_config_used_collapsible:
            self.run_config_used_collapsible.add_widget(self.run_config_used_view)
            post_run_layout.addWidget(results_group, 1)
            post_run_layout.addWidget(self.run_config_used_collapsible)
        else:
            post_run_layout.addWidget(results_group, 1)
            post_run_layout.addWidget(self.run_config_used_view)

        self.right_panel_stack.addWidget(pre_run_page)
        self.right_panel_stack.addWidget(in_run_page)
        self.right_panel_stack.addWidget(post_run_page)
        results_layout.addWidget(self.right_panel_stack, 1)

        shell.addWidget(sidebar_scroll)
        shell.addWidget(workspace_col)
        shell.addWidget(results_scroll)
        shell.setStretchFactor(0, 0)
        shell.setStretchFactor(1, 1)
        shell.setStretchFactor(2, 0)
        layout.addWidget(shell, 1)

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
        view = make_text_browser("OutputView")
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
        layout.setSpacing(dp(PADDING_MD))
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("FieldLabel")
        label.setBuddy(control)
        layout.addWidget(label)
        if trailing is None:
            layout.addWidget(control)
        else:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(dp(PADDING_MD))
            row.addWidget(control, 1)
            row.addWidget(trailing, 0)
            layout.addLayout(row)
        if helper_text:
            helper = QtWidgets.QLabel(helper_text)
            helper.setObjectName("FieldHelper")
            helper.setWordWrap(True)
            layout.addWidget(helper)
        return wrapper

    def _create_panel_card(
        self,
        eyebrow: str,
        title: str,
        description: str = "",
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        panel = QtWidgets.QFrame()
        panel.setObjectName("PanelCard")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
        layout.setSpacing(dp(INTERNAL_GAP))

        if eyebrow:
            eyebrow_label = QtWidgets.QLabel(eyebrow.upper())
            eyebrow_label.setObjectName("SectionEyebrow")
            layout.addWidget(eyebrow_label)

        if title:
            title_label = QtWidgets.QLabel(title)
            title_label.setObjectName("PanelTitle")
            layout.addWidget(title_label)

        if description:
            description_label = QtWidgets.QLabel(description)
            description_label.setObjectName("HintLabel")
            description_label.setWordWrap(True)
            layout.addWidget(description_label)

        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(dp(INTERNAL_GAP))
        layout.addLayout(body, 1)
        return panel, body

    def _init_results_tabs(self):
        self.tabs.clear()
        self.model_tabs.clear()
        self.model_texts.clear()

        self.results_overview_view = self._create_output_view()
        self.results_winner_view = self._create_output_view()
        self.results_ballots_view = self._create_output_view()
        self.results_discussion_view = self._create_output_view()

        self.selected_model_list = QtWidgets.QListWidget()
        self.selected_model_list.setMinimumWidth(dp(160))
        self.selected_model_list.setMaximumWidth(dp(200))
        self.selected_model_list.currentTextChanged.connect(self._refresh_selected_model_detail)
        self.selected_model_detail_view = self._create_output_view()
        selected_model_page = QtWidgets.QWidget()
        selected_model_layout = QtWidgets.QHBoxLayout(selected_model_page)
        selected_model_layout.setContentsMargins(0, 0, 0, 0)
        selected_model_layout.setSpacing(10)
        selected_model_layout.addWidget(self.selected_model_list, 0)
        selected_model_layout.addWidget(self.selected_model_detail_view, 1)

        self.inline_log_view = QtWidgets.QPlainTextEdit()
        self.inline_log_view.setObjectName("OutputLog")
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

    def _set_badge_tone(self, badge: QtWidgets.QWidget, tone: str):
        badge.setProperty("tone", tone)
        style = badge.style()
        if style:
            style.unpolish(badge)
            style.polish(badge)
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
            "accent": THEME["accent"],
            "accent_muted": self._palette_hex(QtGui.QPalette.AlternateBase),
            "danger": THEME["danger"],
            "danger_bg": THEME["bg_tertiary"],
            "success": THEME["success"],
            "success_bg": THEME["bg_tertiary"],
        }

    def _output_document_css(self) -> str:
        c = self._surface_tokens()
        return f"""
        body {{
            background: transparent;
            color: {c["text_primary"]};
            font-family: "Segoe UI";
            font-size: {FONT_BASE}pt;
            line-height: 1.4;
        }}
        p, ul, ol {{
            margin-top: 0;
            margin-bottom: {PADDING_MD}px;
            color: {c["text_primary"]};
        }}
        li {{
            margin-bottom: {PADDING_SM}px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {c["text_primary"]};
            margin-top: 0;
            margin-bottom: {PADDING_MD}px;
        }}
        a {{
            color: {c["accent"]};
        }}
        code {{
            color: {c["text_primary"]};
            background: {c["panel_subtle"]};
            border-radius: {PADDING_SM}px;
            padding: 1px {PADDING_SM}px;
        }}
        pre {{
            white-space: pre-wrap;
            color: {c["text_primary"]};
            background: {c["panel_subtle"]};
            border: 1px solid {c["border"]};
            border-radius: {PADDING_MD}px;
            padding: {PADDING_MD}px {PADDING_LG}px;
            margin: 0 0 {PADDING_LG}px 0;
        }}
        blockquote {{
            margin: 0 0 {PADDING_LG}px 0;
            padding: 0 0 0 {PADDING_LG}px;
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
        <div style="border:1px solid {colors['border']}; border-radius:{PADDING_LG}px; padding:{PADDING_LG}px; background:{colors['panel_subtle']};">
            <div style="font-size:{FONT_LG}pt; font-weight:700; color:{colors['text_primary']}; margin-bottom:{PADDING_SM}px;">{title}</div>
            <div style="color:{colors['text_secondary']}; line-height:1.5;">{message}</div>
        </div>
        """

    def _refresh_header_badges(self):
        selected_count = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        file_count = len(self.uploaded_files)
        loaded_count = len(self.models)
        ready = loaded_count > 0 and selected_count > 0
        self.mode_badge.setText("mode: discussion" if self.mode == "discussion" else "mode: deliberation")
        self._set_badge_tone(self.mode_badge, "neutral")
        if ready:
            self.connection_btn.setText("✓ Ready")
            self._set_badge_tone(self.connection_btn, "success")
        else:
            self.connection_btn.setText("⚠ Setup required")
            self._set_badge_tone(self.connection_btn, "warn")
        self.selection_badge.setText(f"sel: {selected_count}")
        self.attachment_badge.setText(f"files: {file_count}")
        self.tool_badge.setText("web: on" if self.web_search_check.isChecked() else "web: off")
        if hasattr(self, "step_connect"):
            self._refresh_workflow_steps()
        self._sync_run_ready_state()

    def _set_run_button_state(self, state: RunState):
        self.run_state = state
        if isinstance(self.run_btn, StatefulRunButton):
            self.run_btn.set_run_state(state.value)
        else:
            self.run_btn.setEnabled(state == RunState.READY)
            self.run_btn.setText("Run council" if state in (RunState.LOCKED, RunState.READY) else "Running...")

    def _sync_run_ready_state(self):
        selected_count = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        if self.run_state in (RunState.RUNNING, RunState.STOPPING):
            return
        self._set_run_button_state(RunState.READY if selected_count > 0 else RunState.LOCKED)

    def _set_right_panel_state(self, state: RightPanelState):
        self.right_panel_state = state
        index_map = {
            RightPanelState.PRE_RUN: 0,
            RightPanelState.IN_RUN: 1,
            RightPanelState.POST_RUN: 2,
        }
        if hasattr(self, "right_panel_stack"):
            self.right_panel_stack.setCurrentIndex(index_map[state])
        tabs_enabled = state == RightPanelState.POST_RUN
        if hasattr(self, "tabs"):
            for idx in range(self.tabs.count()):
                self.tabs.setTabEnabled(idx, tabs_enabled)
            if tabs_enabled:
                self.tabs.setCurrentIndex(0)

    def _run_config_summary_html(self) -> str:
        single_voter = self.single_voter_combo.currentText().strip() if self.single_voter_check.isChecked() else "All selected models vote"
        return (
            f"<div><strong>Mode:</strong> {escape_text(self.mode) if escape_text else self.mode}</div>"
            f"<div><strong>Voting:</strong> {escape_text(single_voter) if escape_text else single_voter}</div>"
            f"<div><strong>Concurrency:</strong> {int(self.conc_spin.value())}</div>"
            f"<div><strong>Timeout:</strong> {int(self.timeout_seconds)} s</div>"
            f"<div><strong>Web search:</strong> {'On' if self.web_search_check.isChecked() else 'Off'}</div>"
        )

    def _update_run_config_used_view(self):
        if hasattr(self, "run_config_used_view"):
            self.run_config_used_view.setHtml(self._run_config_summary_html())

    def _mark_results_stale(self, message: str = "New run will clear the currently displayed results."):
        if not self.last_session_record:
            return
        self._results_stale = True
        if self.right_panel_state == RightPanelState.POST_RUN:
            self.pre_run_warning_label.setText(message)
            self.pre_run_warning_label.show()
            self._set_right_panel_state(RightPanelState.PRE_RUN)

    def _refresh_workflow_steps(self):
        if not hasattr(self, "step_connect"):
            return
        loaded_count = len(self.models)
        selected_count = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        provider = self._current_provider_config()
        provider_summary = f"{provider_label(provider.provider_type)} · {provider.base_url}"
        self.step_connect.set_summary(provider_summary if loaded_count else "Choose a provider and load models.")
        self.step_connect.set_state(WorkflowStepCard.COMPLETE if loaded_count else WorkflowStepCard.ACTIVE)
        self.step_connect.set_collapsed(bool(loaded_count), animate=False)

        if not loaded_count:
            self.step_models.set_summary("Load models to unlock selection.")
            self.step_models.set_state(WorkflowStepCard.LOCKED)
            self.step_models.set_collapsed(False, animate=False)
        else:
            self.step_models.set_summary(
                f"{selected_count} model{'s' if selected_count != 1 else ''} selected"
                if selected_count else "Choose one or more models."
            )
            self.step_models.set_state(WorkflowStepCard.COMPLETE if selected_count else WorkflowStepCard.ACTIVE)
            self.step_models.set_collapsed(bool(selected_count), animate=False)

        self.step_history.set_state(WorkflowStepCard.IDLE)
        self.step_history.set_collapsed(True, animate=False)

    def _handle_workflow_step_clicked(self, step_number: int):
        mapping = {
            1: getattr(self, "step_connect", None),
            2: getattr(self, "step_models", None),
            3: getattr(self, "step_history", None),
        }
        target = mapping.get(step_number)
        if not target:
            return
        for num, widget in mapping.items():
            if widget is None or widget is target:
                continue
            if num != 3:
                widget.set_collapsed(widget._state == WorkflowStepCard.COMPLETE, animate=False)

    def _handle_workflow_step_completed(self, _step_number: int, _completed: bool):
        self._sync_run_ready_state()

    def _show_connection_menu(self):
        menu = QtWidgets.QMenu(self)
        provider = self._current_provider_config()
        selected_count = sum(1 for cb in self.model_checks.values() if cb.isChecked())
        loaded_count = len(self.models)
        file_count = len(self.uploaded_files)
        ready = loaded_count > 0 and selected_count > 0
        menu.addSection("Ready summary" if ready else "Setup checklist")
        if ready:
            for line in (
                f"Provider: {provider_label(provider.provider_type)}",
                f"Endpoint: {provider.base_url}",
                f"Models loaded: {loaded_count}",
                f"Models selected: {selected_count}",
                f"Attachments: {file_count}",
                f"Web search: {'On' if self.web_search_check.isChecked() else 'Off'}",
            ):
                action = menu.addAction(line)
                action.setEnabled(False)
        else:
            checklist = [
                ("Load models", loaded_count > 0),
                ("Select at least one model", selected_count > 0),
            ]
            for label, done in checklist:
                action = menu.addAction(f"{'✓' if done else '○'} {label}")
                action.setEnabled(False)
        menu.addSeparator()
        load_action = menu.addAction("Load models")
        replace_action = menu.addAction("Replace model list")
        settings_action = menu.addAction("Open settings panel")
        selected = menu.exec(self.connection_btn.mapToGlobal(self.connection_btn.rect().bottomLeft()))
        if selected == load_action:
            self._connect_base()
        elif selected == replace_action:
            self._replace_models_clicked()
        elif selected == settings_action:
            self._open_settings_dialog()

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
                "Attach supporting files here before you run."
            )
        else:
            self.attachment_help_label.setText(
                f"{count} attachment{'s' if count != 1 else ''} ready to include in the next run."
            )
        self.files_list.setVisible(count > 0)
        self.remove_file_btn.setEnabled(count > 0)
        self.clear_files_btn.setEnabled(count > 0)
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
        <div style="font-size:{FONT_XL}pt; font-weight:700; margin-bottom:{PADDING_SM}px;">{esc(short_id(model_id))}</div>
        <div style="margin-bottom:{PADDING_MD}px; color:{colors['text_secondary']};">{esc(provider_text)}</div>
        <div style="margin-bottom:{PADDING_LG}px;"><strong>Score:</strong> {esc(str(score))}<br><strong>Persona:</strong> {esc(persona)}</div>
        {self._safe_markdown_html(answer or "_No response returned._")}
        """
        self.selected_model_detail_view.setHtml(detail_html)

    def _stream_winner_result(self, winner: str, answer: str, tally: Dict[str, Any], details: Dict[str, Any]):
        if self._winner_stream_timer:
            self._winner_stream_timer.stop()
            self._winner_stream_timer.deleteLater()
            self._winner_stream_timer = None

        words = (answer or "").split()
        if not words:
            self.results_winner_view.setHtml(
                result_presenter.build_winner_html(
                    winner=winner,
                    answer=answer,
                    tally=tally,
                    details=details,
                    colors=self._surface_tokens(),
                    short_id=short_id,
                    safe_markdown_html=self._safe_markdown_html,
                    escape_text=escape_text,
                )
            )
            return

        state = {"index": 0}
        timer = QtCore.QTimer(self)
        timer.setInterval(24 if not self.reduce_motion else 1)

        def tick():
            state["index"] = min(len(words), state["index"] + 12)
            partial_answer = " ".join(words[: state["index"]])
            if state["index"] < len(words):
                partial_answer = f"{partial_answer} ▋"
            self.results_winner_view.setHtml(
                result_presenter.build_winner_html(
                    winner=winner,
                    answer=partial_answer,
                    tally=tally,
                    details=details,
                    colors=self._surface_tokens(),
                    short_id=short_id,
                    safe_markdown_html=self._safe_markdown_html,
                    escape_text=escape_text,
                )
            )
            if state["index"] >= len(words):
                timer.stop()
                timer.deleteLater()
                self._winner_stream_timer = None

        timer.timeout.connect(tick)
        self._winner_stream_timer = timer
        tick()
        timer.start()

    def _set_results_empty_state(self):
        if self._winner_stream_timer:
            self._winner_stream_timer.stop()
            self._winner_stream_timer.deleteLater()
            self._winner_stream_timer = None
        self.last_session_record = None
        self._results_stale = False
        if hasattr(self, "pre_run_warning_label"):
            self.pre_run_warning_label.hide()
        self.results_hint_label.setText("Run a council round to populate model outputs, scoring, and exports.")
        self._init_results_tabs()
        for idx in range(self.tabs.count()):
            self.tabs.setTabEnabled(idx, False)
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
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.reset_timeline()
        self.selected_model_list.clear()
        self._current_answers.clear()
        self._current_tally.clear()
        self._current_details.clear()
        self._current_winner = ""
        self._current_question = ""
        self._refresh_selected_model_detail()
        if hasattr(self, "in_run_live_view"):
            self.in_run_live_view.clear()
        self._update_run_config_used_view()
        self._set_right_panel_state(RightPanelState.PRE_RUN)
        self._sync_run_ready_state()

    def _setup_log_dock(self):
        self.debug_timeline_view = DebugTimelineWidget() if DebugTimelineWidget else None
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self.log_history_limit)
        if self.debug_timeline_view:
            log_tabs = QtWidgets.QTabWidget()
            log_tabs.setDocumentMode(True)
            log_tabs.addTab(self.debug_timeline_view, "Timeline")
            log_tabs.addTab(self.log_view, "Raw Log")
            log_widget = log_tabs
        else:
            log_widget = self.log_view
        self.log_dock = QtWidgets.QDockWidget("Debug Log", self)
        self.log_dock.setWidget(log_widget)
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
            self.settings_timeout_spin = panel.settings_timeout_spin
            self.settings_reduce_motion_check = panel.settings_reduce_motion_check
            self.rubric_weight_spins = panel.rubric_weight_spins
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
            timeout_row = QtWidgets.QWidget()
            timeout_layout = QtWidgets.QHBoxLayout(timeout_row)
            timeout_layout.setContentsMargins(0, 0, 0, 0)
            timeout_layout.setSpacing(8)
            timeout_layout.addWidget(QtWidgets.QLabel("Request timeout"))
            timeout_layout.addStretch(1)
            self.settings_timeout_spin = QtWidgets.QSpinBox()
            self.settings_timeout_spin.setRange(15, 600)
            self.settings_timeout_spin.setSuffix(" s")
            timeout_layout.addWidget(self.settings_timeout_spin)
            self.settings_reduce_motion_check = QtWidgets.QCheckBox("Reduce motion and skip non-essential animations")
            self.rubric_weight_spins = {}
            rubric_group = QtWidgets.QGroupBox("Scoring Rubric")
            rubric_layout = QtWidgets.QGridLayout(rubric_group)
            rubric_layout.setContentsMargins(10, 12, 10, 10)
            rubric_layout.setHorizontalSpacing(10)
            rubric_layout.setVerticalSpacing(8)
            for row, (label_text, key) in enumerate(
                [
                    ("Correctness", "correctness"),
                    ("Relevance", "relevance"),
                    ("Specificity", "specificity"),
                    ("Safety", "safety"),
                    ("Conciseness", "conciseness"),
                ]
            ):
                rubric_layout.addWidget(QtWidgets.QLabel(label_text), row, 0)
                spin = QtWidgets.QSpinBox()
                spin.setRange(0, 10)
                rubric_layout.addWidget(spin, row, 1)
                self.rubric_weight_spins[key] = spin
            self.settings_shortcuts_btn = QtWidgets.QPushButton("Keyboard Shortcuts")
            self.settings_shortcuts_btn.setObjectName("SecondaryButton")
            self.settings_issue_btn = QtWidgets.QPushButton("Report an Issue")
            self.settings_issue_btn.setObjectName("SecondaryButton")

            settings_layout.addWidget(self.settings_debug_check)
            settings_layout.addWidget(self.settings_personas_check)
            settings_layout.addWidget(self.settings_storage_label)
            settings_layout.addWidget(timeout_row)
            settings_layout.addWidget(self.settings_reduce_motion_check)
            settings_layout.addWidget(rubric_group)
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
        self.settings_timeout_spin.valueChanged.connect(self._timeout_changed)
        if hasattr(self, "run_timeout_spin"):
            self.run_timeout_spin.valueChanged.connect(self._timeout_changed)
        self.settings_reduce_motion_check.toggled.connect(self._reduce_motion_toggled)
        for key, spin in getattr(self, "rubric_weight_spins", {}).items():
            spin.valueChanged.connect(lambda value, rubric_key=key: self._rubric_weight_changed(rubric_key, value))
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
        if isinstance(self.run_btn, StatefulRunButton):
            self.run_btn.runRequested.connect(self._send)
            self.run_btn.stopRequested.connect(self._stop_process)
        else:
            self.send_btn.clicked.connect(self._send)
        self.prompt_edit.submitRequested.connect(self._send)
        self.conc_spin.valueChanged.connect(self._concurrency_changed)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        self.connection_btn.clicked.connect(self._show_connection_menu)
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
        if hasattr(self, "step_connect"):
            self.step_connect.headerClicked.connect(self._handle_workflow_step_clicked)
            self.step_models.headerClicked.connect(self._handle_workflow_step_clicked)
            self.step_history.headerClicked.connect(self._handle_workflow_step_clicked)
            self.step_connect.stepCompleted.connect(self._handle_workflow_step_completed)
            self.step_models.stepCompleted.connect(self._handle_workflow_step_completed)

        self.status_signal.connect(self._set_status)
        self.result_signal.connect(self._handle_result)
        self.error_signal.connect(self._handle_error)
        self.log_signal.connect(self._append_log)
        self.discussion_update_signal.connect(self._update_discussion_view)
        self.capability_update_signal.connect(self._update_capability_ui)
        
        # Keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _configure_tab_order(self):
        tab_sequence = [
            self.provider_combo,
            self.api_service_combo,
            self.base_edit,
            self.api_key_inline,
            self.show_key_check,
            self.connect_btn,
            self.replace_models_btn,
            self.add_provider_btn,
            self.model_filter_edit,
            self.refresh_models_btn,
            self.select_all_btn,
            self.clear_btn,
            self.upload_btn,
            self.remove_file_btn,
            self.clear_files_btn,
            self.prompt_edit,
            self.run_btn,
            self.settings_btn,
        ]
        for current, nxt in zip(tab_sequence, tab_sequence[1:]):
            QtWidgets.QWidget.setTabOrder(current, nxt)
    
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

        theme_shortcut = QtGui.QShortcut(QtGui.QKeySequence(ACTIONS["toggle_theme"]["shortcut"]), self)
        theme_shortcut.activated.connect(self._toggle_theme)

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

    def _toggle_theme(self):
        if self._theme_engine is None:
            return
        from ui.theme import toggle_theme

        toggle_theme(self, self._theme_engine)
        save_settings({"theme_mode": "dark" if self._theme_engine.is_dark else "light"})
        self._refresh_output_document_styles()
        self._set_status(f"Theme set to {'dark' if self._theme_engine.is_dark else 'light'} mode.")

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

    def _timeout_changed(self, value: int):
        self.timeout_seconds = int(value)
        if hasattr(self, "settings_timeout_spin") and self.settings_timeout_spin.value() != self.timeout_seconds:
            self.settings_timeout_spin.blockSignals(True)
            self.settings_timeout_spin.setValue(self.timeout_seconds)
            self.settings_timeout_spin.blockSignals(False)
        if hasattr(self, "run_timeout_spin") and self.run_timeout_spin.value() != self.timeout_seconds:
            self.run_timeout_spin.blockSignals(True)
            self.run_timeout_spin.setValue(self.timeout_seconds)
            self.run_timeout_spin.blockSignals(False)
        save_settings({"timeout_seconds": self.timeout_seconds})
        self._mark_results_stale()
        self._update_run_config_used_view()
        self._set_status(f"Request timeout set to {self.timeout_seconds} seconds.")

    def _reduce_motion_toggled(self, checked: bool):
        self.reduce_motion = bool(checked)
        if set_reduce_motion:
            set_reduce_motion(self.reduce_motion)
        save_settings({"reduce_motion": self.reduce_motion})
        self._set_status("Reduced motion enabled." if self.reduce_motion else "Reduced motion disabled.")

    def _rubric_weight_changed(self, key: str, value: int):
        self.rubric_weights[key] = max(0, int(value))
        self.rubric_weights = normalize_rubric_weights(self.rubric_weights)
        save_settings({"rubric_weights": self.rubric_weights})
        self._set_status(f"Updated rubric weight for {key}.")

    def _refresh_settings_panel(self):
        if not hasattr(self, "settings_debug_check"):
            return
        self.settings_debug_check.blockSignals(True)
        self.settings_personas_check.blockSignals(True)
        self.settings_timeout_spin.blockSignals(True)
        self.settings_reduce_motion_check.blockSignals(True)
        self.settings_debug_check.setChecked(self.debug_enabled)
        self.settings_personas_check.setChecked(self.use_roles)
        self.settings_timeout_spin.setValue(int(self.timeout_seconds))
        if hasattr(self, "run_timeout_spin"):
            self.run_timeout_spin.blockSignals(True)
            self.run_timeout_spin.setValue(int(self.timeout_seconds))
            self.run_timeout_spin.blockSignals(False)
        self.settings_reduce_motion_check.setChecked(bool(self.reduce_motion))
        for key, spin in getattr(self, "rubric_weight_spins", {}).items():
            spin.blockSignals(True)
            spin.setValue(int(self.rubric_weights.get(key, 0)))
            spin.blockSignals(False)
        self.settings_debug_check.blockSignals(False)
        self.settings_personas_check.blockSignals(False)
        self.settings_timeout_spin.blockSignals(False)
        self.settings_reduce_motion_check.blockSignals(False)
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
        if populate_persona_library_list:
            populate_persona_library_list(
                self.persona_library_list,
                self.personas,
                query=query,
                current_name=current_name,
            )
        else:
            self.persona_library_list.clear()
            for persona in self.personas:
                name = persona["name"]
                prompt = persona.get("prompt") or ""
                if query and query not in name.lower() and query not in prompt.lower():
                    continue
                item = QtWidgets.QListWidgetItem(name)
                if persona.get("builtin", False):
                    item.setForeground(self.palette().color(QtGui.QPalette.Mid))
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
            f"<div style='font-size:{FONT_LG}pt; font-weight:700; margin-bottom:{PADDING_SM}px;'>{escape_text(persona['name']) if escape_text else persona['name']}</div>"
            f"<div style='margin-bottom:{PADDING_MD}px; color:{colors['text_secondary']};'>"
            f"{'Built-in' if persona.get('builtin', False) else 'Custom'} persona · Assigned to {assignment_count} model(s)</div>"
            f"{self._safe_markdown_html(prompt)}"
        )

    def _prompt_for_persona_text(self, title: str, current_prompt: str = "") -> Optional[str]:
        prompt_dialog = QtWidgets.QDialog(self)
        prompt_dialog.setWindowTitle(title)
        prompt_dialog.resize(dp(560), dp(360))
        prompt_dialog.setMinimumSize(dp(480), dp(320))
        prompt_dialog.setModal(True)
        prompt_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        layout = QtWidgets.QVBoxLayout(prompt_dialog)
        layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
        layout.setSpacing(dp(INTERNAL_GAP))
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
        self.personas.append({"name": name, "prompt": prompt if prompt else None, "builtin": False})
        try:
            if add_user_persona:
                add_user_persona(name, prompt if prompt else None)
        except Exception:
            LOGGER.exception("Error saving user persona")
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
            if update_user_persona:
                update_user_persona(name, new_name, prompt if prompt else None)
        except Exception:
            LOGGER.exception("Error updating user persona")
        if name != new_name:
            self.persona_assignments = (
                rename_persona_assignments(self.persona_assignments, name, new_name)
                if rename_persona_assignments
                else self.persona_assignments
            )
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
            if delete_user_persona:
                delete_user_persona(name)
        except Exception:
            LOGGER.exception("Error deleting user persona")
        self.persona_assignments = (
            clear_persona_assignment(self.persona_assignments, name)
            if clear_persona_assignment
            else self.persona_assignments
        )
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
        enabled = bool(self.use_roles)
        if apply_persona_visibility:
            apply_persona_visibility(self.model_persona_combos, enabled=enabled)
        else:
            for btn in list(self.model_persona_combos.values()):
                if not btn or not isinstance(btn, QtWidgets.QPushButton):
                    continue
                btn.setVisible(enabled)
                btn.setEnabled(enabled)
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
        self._refresh_persona_combos()
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
        if merge_persona_library:
            return merge_persona_library(stored)
        return []

    def _sort_personas_inplace(self):
        if sort_personas_inplace:
            sort_personas_inplace(self.personas)

    def _persona_names(self) -> List[str]:
        return persona_names(self.personas) if persona_names else [persona["name"] for persona in self.personas]

    def _persona_prompt(self, name: str) -> Optional[str]:
        return persona_prompt(self.personas, name) if persona_prompt else None

    def _persona_by_name(self, name: str) -> Optional[dict]:
        return persona_by_name(self.personas, name) if persona_by_name else None

    def _cleanup_persona_assignments(self):
        if cleanup_persona_assignments:
            self.persona_assignments, dirty = cleanup_persona_assignments(self.personas, self.persona_assignments)
        else:
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
            if hasattr(btn, "setPersonaText"):
                btn.setPersonaText(assigned)
            else:
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
        colors = self._surface_tokens()
        if build_persona_preview_html and escape_text and assignment_count:
            html = build_persona_preview_html(
                persona,
                assignment_count=assignment_count(self.persona_assignments, persona["name"]),
                colors=colors,
                render_markdown=self._safe_markdown_html,
                escape_text=escape_text,
                placeholder_html=self._placeholder_html,
            )
        else:
            prompt = persona.get("prompt") or "No prompt configured."
            assigned_count = sum(1 for assigned in self.persona_assignments.values() if assigned == persona["name"])
            html = (
                f"<div style='font-size:{FONT_LG}pt; font-weight:700; margin-bottom:{PADDING_SM}px;'>{escape_text(persona['name']) if escape_text else persona['name']}</div>"
                f"<div style='margin-bottom:{PADDING_MD}px; color:{colors['text_secondary']};'>"
                f"{'Built-in' if persona.get('builtin', False) else 'Custom'} persona &middot; Assigned to {assigned_count} model(s)</div>"
                f"{self._safe_markdown_html(prompt)}"
            )
        self.persona_preview.setHtml(html)

    def _concurrency_changed(self, value: int):
        save_settings({"max_concurrency": int(value)})
        self._mark_results_stale()
        self._update_run_config_used_view()
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
        self._mark_results_stale()
        self._update_run_config_used_view()
        self._set_status("Single-voter mode: ON" if checked else "Single-voter mode: OFF")

    def _single_voter_changed(self, text: str):
        save_settings({"single_voter_model": text})
        self._mark_results_stale()
        self._update_run_config_used_view()

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

            self._model_worker = ModelFetchWorker(provider, timeout_seconds=self.timeout_seconds)
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
        if make_unique_display_model_name:
            return make_unique_display_model_name(raw_model, self._provider_tag(provider), self.model_actual_ids)
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
            self.upload_btn.setToolTip("Attach documents or images. Visual-capable models are available.")
        else:
            self.upload_btn.setToolTip("Attach documents. Image attachments require a visual-capable model.")
        
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
        session_timings = (self.last_session_record or {}).get("timings_ms", {}) or {}
        latency_ms = session_timings.get(model)
        if build_model_badge_text:
            return build_model_badge_text(
                provider_badge,
                capabilities=self.model_capabilities.get(model, {}),
                latency_ms=latency_ms,
            )
        caps = self.model_capabilities.get(model, {})
        cap_parts = []
        if caps.get("visual"):
            cap_parts.append("vision")
        if caps.get("web_search"):
            cap_parts.append("web")
        details = [provider_badge]
        if cap_parts:
            details.append(", ".join(cap_parts))
        if isinstance(latency_ms, (int, float)):
            details.append(f"{int(latency_ms)} ms")
        return " | ".join(details)

    def _populate_models(self):
        if not hasattr(self, 'personas') or not self.personas:
            s = load_settings()
            self.personas = self._merge_persona_library(s.get("personas", []))
            if not hasattr(self, 'persona_assignments'):
                self.persona_assignments = dict(s.get("persona_assignments", {}) or {})

        persona_name_list = self._persona_names()
        if not persona_name_list:
            self.personas = [{"name": "None", "prompt": None, "builtin": True}]
            persona_name_list = ["None"]

        if populate_model_rows:
            widgets = populate_model_rows(
                layout=self.models_layout,
                models=self.models,
                persona_assignments=self.persona_assignments,
                persona_names=persona_name_list,
                personas_enabled=self.use_roles,
                provider_name_for_model=lambda model: (
                    provider_label(self.model_provider_map.get(model).provider_type)
                    if self.model_provider_map.get(model)
                    else "Unknown"
                ),
                meta_text_for_model=self._model_badge_text,
                on_persona_click=self._show_persona_menu,
                on_selection_changed=self._on_model_selection_changed,
                ui_available=_UI_AVAILABLE,
                model_card_class=ModelCard if _UI_AVAILABLE else None,
            )
            self.model_checks = widgets.checks
            self.model_persona_combos = widgets.persona_buttons
            self.model_meta_labels = widgets.meta_labels
            self.model_rows = widgets.rows
        else:
            while self.models_layout.count():
                item = self.models_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                del item
            self.model_checks.clear()
            self.model_persona_combos.clear()
            self.model_meta_labels.clear()
            self.model_rows.clear()
            self.models_layout.addStretch(1)

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
            if leaderboard_store:
                leaderboard_store.reset_leaderboard()
            else:
                if DB_PATH.exists():
                    DB_PATH.unlink()
                ensure_db()
            self._refresh_leaderboard()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reset Leaderboard", str(e))

    def _refresh_leaderboard(self):
        self.leader_list.clear()
        leaderboard = leaderboard_store.load_leaderboard() if leaderboard_store else load_leaderboard()
        total_wins = sum(count for _, count in leaderboard)
        for mid, count in leaderboard:
            pct = (count / total_wins * 100) if total_wins > 0 else 0
            item = QtWidgets.QListWidgetItem(self.leader_list)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            row = QtWidgets.QWidget()
            row.setObjectName("LeaderboardItem")
            row_layout = QtWidgets.QVBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(6)

            top = QtWidgets.QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(8)
            name_label = QtWidgets.QLabel(short_id(mid))
            name_label.setObjectName("FieldLabel")
            wins_label = QtWidgets.QLabel(f"{count} win{'s' if count != 1 else ''}")
            wins_label.setObjectName("HintLabel")
            top.addWidget(name_label)
            top.addStretch(1)
            top.addWidget(wins_label)

            bar = QtWidgets.QProgressBar()
            bar.setObjectName("WinRateBar")
            bar.setRange(0, 100)
            bar.setValue(int(round(pct)))
            bar.setTextVisible(False)
            bar.setFixedHeight(8)

            bottom = QtWidgets.QHBoxLayout()
            bottom.setContentsMargins(0, 0, 0, 0)
            bottom.setSpacing(8)
            share_label = QtWidgets.QLabel("share of recorded wins")
            share_label.setObjectName("HintLabel")
            pct_label = QtWidgets.QLabel(f"{pct:.0f}%")
            pct_label.setObjectName("FieldLabel")
            bottom.addWidget(share_label)
            bottom.addStretch(1)
            bottom.addWidget(pct_label)

            row_layout.addLayout(top)
            row_layout.addWidget(bar)
            row_layout.addLayout(bottom)
            item.setSizeHint(row.sizeHint())
            self.leader_list.setItemWidget(item, row)
        if leaderboard:
            self.lb_title.setText(f"Council performance over time · {len(leaderboard)} tracked model(s)")
        else:
            self.lb_title.setText("Council performance over time · no votes recorded yet")
            item = QtWidgets.QListWidgetItem(self.leader_list)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            placeholder = QtWidgets.QLabel("No wins recorded yet. Run a deliberation round to start tracking model performance.")
            placeholder.setObjectName("HintLabel")
            placeholder.setWordWrap(True)
            placeholder.setContentsMargins(10, 8, 10, 8)
            item.setSizeHint(placeholder.sizeHint())
            self.leader_list.setItemWidget(item, placeholder)

    def _filter_model_rows(self):
        query = self.model_filter_edit.text().strip().lower()
        if filter_model_rows:
            filter_model_rows(self.model_rows, self.model_meta_labels, query)
        else:
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
        user_bg = THEME["accent"] if is_dark else THEME["bg_tertiary"]
        user_fg = THEME["accent_fg"] if is_dark else text_color
        council_bg = colors["panel"]
        note_bg = "transparent"
        error_bg = colors["danger_bg"]
        style = f"margin: {PADDING_SM}px 0; padding: {PADDING_MD}px {PADDING_LG}px; border-radius: {PADDING_MD}px;"
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
        if hasattr(self, "in_run_live_view"):
            self.in_run_live_view.append(full_html)
            self.in_run_live_view.verticalScrollBar().setValue(self.in_run_live_view.verticalScrollBar().maximum())

    def _append_log(self, text: str):
        if not text:
            return
        if self.debug_enabled and not self.log_dock.isVisible():
            self.log_dock.show()
        self.log_view.appendPlainText(text)
        if hasattr(self, "inline_log_view") and self.inline_log_view:
            self.inline_log_view.appendPlainText(text)
        if hasattr(self, "in_run_progress_label"):
            self.in_run_progress_label.setText(text)
        if hasattr(self, "in_run_live_view"):
            self.in_run_live_view.append(f"<div>{escape_text(text) if escape_text else text}</div>")
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.add_status(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _log_sink_dispatch(self, label: str, message: str):
        if not message:
            return
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        formatted = f"{stamp} [{label}] {message}"
        self.log_signal.emit(formatted)

    def _set_status(self, text: str):
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.add_status(text)
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
            self._set_right_panel_state(RightPanelState.IN_RUN)
        else:
            self._refresh_attachment_summary()
            self._set_status(self.status_label.text())
            if self.right_panel_state == RightPanelState.IN_RUN:
                self._set_right_panel_state(RightPanelState.PRE_RUN if not self.last_session_record else RightPanelState.POST_RUN)

    def _mode_changed(self, index: int):
        """Handle mode selection change."""
        self.mode = "discussion" if index == 1 else "deliberation"
        self._mark_results_stale()
        # Update UI based on mode
        if self.mode == "discussion":
            self.tabs_title.setText("Discussion report")
            # Hide single voter controls in discussion mode
            self.single_voter_check.setVisible(False)
            self.single_voter_combo.setVisible(False)
            self.mode_help_label.setText(
                "Discussion mode runs multiple turns and produces a synthesized report instead of weighted ballots."
            )
            self.prompt_edit.setPlaceholderText(
                "Start a collaborative discussion. Press Enter to send, Shift+Enter for a new line."
            )
        else:
            self.tabs_title.setText("Council results")
            self.single_voter_check.setVisible(True)
            self.single_voter_combo.setVisible(True)
            self.mode_help_label.setText(
                "Weighted deliberation compares answers, scores them against the rubric, and picks a winner."
            )
            self.prompt_edit.setPlaceholderText(
                "Ask the council a question. Press Enter to send, Shift+Enter for a new line."
            )
        self._update_run_config_used_view()
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
        self._mark_results_stale()
        self._update_run_config_used_view()
        self._refresh_header_badges()
    
    def _stop_process(self):
        """Handle stop button click."""
        self._stop_requested = True
        self._set_run_button_state(RunState.STOPPING)
        self.in_run_progress_label.setText("Stopping the current run…")
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
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.reset_timeline()
        self._append_chat(f"You: {question}")
        if images:
            self._append_chat(f"<i>[Attached {len(images)} image(s)]</i>")
        self.prompt_edit.clear()
        self._results_stale = False
        self.pre_run_warning_label.hide()
        self._set_run_button_state(RunState.RUNNING)
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
                        timeout_seconds=self.timeout_seconds,
                        is_cancelled=lambda: self._stop_requested
                    )
                )
                if not self._stop_requested:
                    # record leaderboard
                    try:
                        if leaderboard_store:
                            leaderboard_store.record_vote(question, winner, details, debug_hook=_dbg)
                        else:
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
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.reset_timeline()
        self._append_chat(f"You: {question}")
        if images:
            self._append_chat(f"<i>[Attached {len(images)} image(s)]</i>")
        self.prompt_edit.clear()
        self._results_stale = False
        self.pre_run_warning_label.hide()
        self._set_run_button_state(RunState.RUNNING)
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
                    timeout_seconds=self.timeout_seconds,
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
        if build_persona_config:
            return build_persona_config(persona_name, self.personas)
        persona_prompt_text = self._persona_prompt(persona_name)
        if persona_prompt_text:
            return {"source": "one_time", "id": None, "one_time_prompt": persona_prompt_text}
        return {"source": "default", "id": None, "one_time_prompt": ""}
    
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
        self._set_run_button_state(RunState.READY if any(cb.isChecked() for cb in self.model_checks.values()) else RunState.LOCKED)
        self.results_hint_label.setText("The last run failed. Review the status message and debug log for details.")
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.add_error(msg)
        self._set_status(f"Error: {msg}")
        self._append_chat(f"[Error] {msg}")

    def _handle_result(self, payload: object):
        self._busy(False)
        self._set_run_button_state(RunState.READY if any(cb.isChecked() for cb in self.model_checks.values()) else RunState.LOCKED)
        
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
        self._stream_winner_result(winner, answers.get(winner, ""), tally, details)
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
        self._update_run_config_used_view()
        self._set_right_panel_state(RightPanelState.POST_RUN)
        self.tabs.setCurrentWidget(self.results_overview_view)
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.show_deliberation_result(details=details, tally=tally, short_id=short_id)
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
        if getattr(self, "debug_timeline_view", None):
            self.debug_timeline_view.show_discussion_result(transcript=transcript, synthesis=synthesis)
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
        self._update_run_config_used_view()
        self._set_right_panel_state(RightPanelState.POST_RUN)
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

    def _restore_window_state(self, settings: Dict[str, Any]) -> None:
        geometry = settings.get("window_geometry", "") or ""
        state = settings.get("window_state", "") or ""
        if geometry:
            try:
                self.restoreGeometry(QtCore.QByteArray.fromBase64(geometry.encode("ascii")))
            except Exception:
                LOGGER.exception("Failed to restore window geometry")
        if state:
            try:
                self.restoreState(QtCore.QByteArray.fromBase64(state.encode("ascii")))
            except Exception:
                LOGGER.exception("Failed to restore window state")
        available = self.screen().availableGeometry() if self.screen() else QtCore.QRect(0, 0, dp(MIN_W), dp(MIN_H))
        current = self.frameGeometry()
        if not available.intersects(current):
            self.resize(max(self.width(), dp(MIN_W)), max(self.height(), dp(MIN_H)))
            self.move(available.topLeft() + QtCore.QPoint(dp(PADDING_LG), dp(PADDING_LG)))

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
                "timeout_seconds": int(self.timeout_seconds),
                "reduce_motion": bool(self.reduce_motion),
                "theme_mode": "dark" if getattr(self, "_theme_engine", None) and self._theme_engine.is_dark else "light",
                "window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                "window_state": bytes(self.saveState().toBase64()).decode("ascii"),
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QtWidgets.QApplication(sys.argv)
    refresh_dpi_scale(app)
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
    
    if qasync:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        app.aboutToQuit.connect(loop.stop)
        with loop:
            w = CouncilWindow()
            w.setWindowIcon(icon)
            w.show()
            loop.run_forever()
    else:
        w = CouncilWindow()
        w.setWindowIcon(icon)
        w.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()
