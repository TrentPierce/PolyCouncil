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
import math
from pathlib import Path
from typing import Optional, Dict, List, Callable, Awaitable, Any, Iterable

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
    print(f"\n===== {label} =====\n{s}\n")

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
    return {"base_url": "http://localhost:1234", "single_voter_enabled": False, "single_voter_model": ""}

def save_settings(s: dict):
    try:
        cur = {}
        if SETTINGS_PATH.exists():
            try:
                cur = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur.update(s)
        SETTINGS_PATH.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    except Exception:
        pass

# -----------------------
# ID normalization (for display)
# -----------------------
ID_SUFFIX_RE = re.compile(r":\d+$")
def canon_id(mid: str) -> str:
    return ID_SUFFIX_RE.sub("", mid)

def short_id(mid: str) -> str:
    base = canon_id(mid)
    return base.split("/")[-1] if "/" in base else base

# -----------------------
# Session state (mutable)
# -----------------------
MODEL_READY: Dict[str, bool] = {}
NO_JSON_OBJECT: Dict[str, bool] = {}

CURRENT_BASE: str = load_settings()["base_url"]

# Personas (answers only; voting is persona-free)
PERSONALITY_BANK = [
    "You are a meticulous fact-checker. Prefer primary sources and verify each claim.",
    "You are a pragmatic engineer. Focus on feasible steps, tradeoffs, and edge cases.",
    "You are a cautious risk assessor. Identify failure modes and propose mitigations.",
    "You are a clear teacher. Explain concepts simply with short examples where helpful.",
    "You are a data analyst. Structure answers into bullets, highlight assumptions and limits.",
    "You are a domain generalist. Cross-reference adjacent fields for overlooked angles.",
    "You are a contrarian reviewer. Challenge consensus and probe weak spots constructively.",
    "You are a synthesizer. Combine ideas into a balanced recommendation with rationale.",
    "You are an optimizer. Seek the highest value per effort and call out diminishing returns.",
    "You are a systems thinker. Map interactions, dependencies, and long-term consequences."
]

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
    out = []
    for t in tasks:
        out.append(await t)
    return out

# -----------------------
# Database
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

def reset_db():
    try:
        if DB_PATH.exists():
            DB_PATH.unlink()
    except Exception:
        pass
    ensure_db()

def record_vote(question: str, winner: str, details: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO votes (timestamp, question, winner, details) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(timespec="seconds"), question, winner, json.dumps(details))
    )
    conn.commit()
    conn.close()

def load_leaderboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        rows = list(cur.execute("SELECT winner, COUNT(*) FROM votes GROUP BY winner ORDER BY COUNT(*) DESC"))
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return rows

# -----------------------
# Endpoint helpers
# -----------------------
def endpoints(base_url: str):
    base = base_url.rstrip("/")
    return {
        "chat":   f"{base}/v1/chat/completions",
        "models": f"{base}/v1/models",
        "loaded": f"{base}/v1/models/loaded",   # optional
        "unload": f"{base}/v1/models/unload",   # optional; used only on uncheck
    }

# -----------------------
# LM Studio helpers
# -----------------------
def fetch_models_from_lmstudio(base_url: str) -> List[str]:
    ep = endpoints(base_url)["models"]
    try:
        r = requests.get(ep, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception as e:
        print("Failed to fetch models:", e)
        return []

def fetch_loaded_models_best_effort(base_url: str) -> List[str]:
    e = endpoints(base_url)
    # 1) Dedicated endpoint
    try:
        r = requests.get(e["loaded"], timeout=6)
        if r.status_code == 200:
            data = r.json()
            return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception:
        pass
    # 2) Hints on /v1/models
    try:
        r = requests.get(e["models"], timeout=8)
        r.raise_for_status()
        data = r.json()
        loaded = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            if m.get("loaded") is True or m.get("isLoaded") is True or m.get("is_ready") is True or m.get("active") is True:
                loaded.append(mid)
        return loaded
    except Exception:
        return []

def fetch_loaded_models_via_sdk() -> List[str]:
    """Optional: local-only SDK. Returns [] if not installed or any error."""
    try:
        import lmstudio as lms  # type: ignore
    except Exception:
        return []
    try:
        loaded = lms.list_loaded_models()
        ids = []
        for h in loaded:
            mid = getattr(h, "model_id", None) or getattr(h, "id", None)
            if not mid:
                try:
                    info = h.info()
                    mid = info.get("id") if isinstance(info, dict) else None
                except Exception:
                    pass
            if mid:
                ids.append(mid)
        return ids
    except Exception:
        return []

# -----------------------
# response_format helpers (LM Studio variants)
# -----------------------
def _build_voting_json_schema(ranges: dict) -> dict:
    # ranges = {"correctness":5, "relevance":3, "specificity":3, "safety":2, "conciseness":1}
    props = {
        k: {"type": "integer", "minimum": 0, "maximum": v}
        for k, v in ranges.items()
    }
    score_obj = {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False
    }
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "patternProperties": {
                    "^[0-9]+$": score_obj
                },
                "additionalProperties": False
            },
            "reasoning": {"type": "string"},
            "final_pick": {"type": "integer"}
        },
        "required": ["scores"],
        "additionalProperties": False
    }

# -----------------------
# Model invocation
# -----------------------
async def call_model(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.3,
    sys_prompt: Optional[str] = None,
    response_format: Optional[dict] = None,
    timeout_sec: int = 180
) -> Any:
    """
    Returns content which may be a str or dict depending on backend/response_format.
    """
    chat_ep = endpoints(base_url)["chat"]
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": msgs, "temperature": temperature}
    if response_format is not None:
        payload["response_format"] = response_format
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with session.post(chat_ep, json=payload, timeout=timeout) as r:
        txt = await r.text()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} for {model}:\n{txt}")
        try:
            data = json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"Non-JSON response for {model}:\n{txt[:800]} ...") from e
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected schema for {model}:\n{json.dumps(data)[:800]} ...") from e

def safe_load_vote_json(payload) -> Optional[dict]:
    """
    Accepts dict/list/str. Attempts to extract the first usable JSON object
    that includes a 'scores' key. Tolerates code fences, <think>, stray text,
    and multiple concatenated JSON objects.
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if not isinstance(payload, str):
        return None

    s = payload.strip()
    # strip think blocks / code fences
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"```(?:json)?", "```", s, flags=re.IGNORECASE)
    parts = s.split("```")
    if len(parts) % 2 == 1:
        # prefer fenced block first
        for i in range(1, len(parts), 2):
            chunk = parts[i].strip()
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict) and "scores" in obj:
                    return obj
            except Exception:
                pass
        # fall back to unfenced text
        s = "".join(parts[::2])

    # scan for multiple JSON objects and pick one with "scores"
    def scan_objects(txt: str) -> List[str]:
        objs, depth, start = [], 0, -1
        in_str, esc = False, False
        for i, ch in enumerate(txt):
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        objs.append(txt[start:i+1])
        return objs

    for cand in scan_objects(s):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "scores" in obj:
                return obj
        except Exception:
            continue
    # final attempt: load any top-level object
    for cand in scan_objects(s):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# -----------------------
# Refusal detection
# -----------------------
def is_refusal_answer(answer: str) -> bool:
    """
    Improved detection of refusal/non-answers
    """
    if not isinstance(answer, str):
        return False
    
    text = answer.strip().lower()
    
    # Common refusal patterns
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "sorry, i cannot", "sorry i cannot",
        "i don't have access", "i don't know",
        "as an ai", "as a language model",
        "i'm not able", "i am not able",
        "i apologize, but",
        "i cannot assist", "i can't assist",
        "that would be inappropriate",
        "i must decline"
    ]
    
    # Check if answer starts with refusal (first 200 chars)
    prefix = text[:200]
    for pattern in refusal_patterns:
        if pattern in prefix:
            return True
    
    # Check if answer is too short and contains apology
    if len(text) < 50 and ("sorry" in text or "apologize" in text):
        return True
    
    return False

# -----------------------
# Roles / personas
# -----------------------
def assign_random_roles(selected_models: List[str], enabled: bool) -> Dict[str, Optional[str]]:
    if not enabled:
        return {m: None for m in selected_models}
    bank = PERSONALITY_BANK[:]
    random.shuffle(bank)
    roles = {}
    for i, m in enumerate(selected_models):
        roles[m] = bank[i % len(bank)]
    return roles

def compose_user_prompt(question: str, role_text: Optional[str]) -> str:
    if not role_text:
        return question
    return (
        "Adopt the following persona for this single answer. "
        "Be accurate, concise, and explicit about assumptions.\n"
        f"{role_text}\n\n"
        f"User question:\n{question}"
    )

# -----------------------
# Warm-up / readiness (skip for already-loaded)
# -----------------------
async def warmup_model_once(session: aiohttp.ClientSession, base_url: str, model: str) -> bool:
    if MODEL_READY.get(model):
        return True
    try:
        _ = await call_model(session, base_url, model, "Reply with OK.", temperature=0.0, timeout_sec=30)
        MODEL_READY[model] = True
        return True
    except Exception:
        await asyncio.sleep(1.0)
        try:
            _ = await call_model(session, base_url, model, "OK?", temperature=0.0, timeout_sec=45)
            MODEL_READY[model] = True
            return True
        except Exception:
            return False

async def ensure_models_ready_once(base_url: str, selected_models: List[str], status_cb, max_concurrency: int = 1) -> Dict[str, bool]:
    ready = {}
    # Prefer SDK
    loaded_now = fetch_loaded_models_via_sdk()
    # Fallback REST
    if not loaded_now:
        loaded_now = fetch_loaded_models_best_effort(base_url)
    if loaded_now:
        for mid in loaded_now:
            if mid in selected_models:
                MODEL_READY[mid] = True
                ready[mid] = True
        if any(m in loaded_now for m in selected_models):
            status_cb(f"Skipping warm-up for loaded: {', '.join([m for m in selected_models if m in loaded_now])}")
    # Warm remaining
    to_warm = [m for m in selected_models if not MODEL_READY.get(m, False)]
    if to_warm:
        async with aiohttp.ClientSession() as session:
            for m in to_warm:
                status_cb(f"Loading model: {m} …")
            def mk_job(model_id):
                async def _job():
                    ok = await warmup_model_once(session, base_url, model_id)
                    status_cb(f"{'Ready' if ok else 'Failed'}: {model_id}")
                    return model_id, ok
                return _job
            pairs = await run_limited(max_concurrency, [mk_job(m) for m in to_warm])
            for m, ok in pairs:
                ready[m] = ok
    for m in selected_models:
        ready.setdefault(m, MODEL_READY.get(m, False))
    return ready

def best_effort_unload(base_url: str, models_to_unload: List[str], status_cb):
    """Only when user UNCHECKS a model."""
    ep = endpoints(base_url)["unload"]
    for m in models_to_unload:
        try:
            r = requests.post(ep, json={"model": m}, timeout=8)
            if r.status_code == 200:
                status_cb(f"Unloaded: {m}")
                MODEL_READY.pop(m, None)
            else:
                status_cb(f"Unload not supported by server (code {r.status_code}) for {m}")
        except requests.RequestException:
            status_cb("Unload API not available; use LM Studio UI to unload models.")
            break

# -----------------------
# IMPROVED VOTING SYSTEM
# -----------------------
def improved_voting_prompt(voter_id: str, peer_idx_map: dict, question: str, answers: dict) -> str:
    """
    Clearer prompt that emphasizes:
    - Explicit scoring rules
    - Examples of good vs bad answers
    - Clear JSON structure with strict requirements
    """
    # Build answer block with clear separation
    answer_blocks = []
    for i, mid in peer_idx_map.items():
        answer_blocks.append(f"=== ANSWER {i} ===\n{answers[mid]}\n")
    
    ballot = "\n".join(answer_blocks)
    
    return f"""You are evaluating answers to this question:
"{question}"

CRITICAL RULES:
1. You CANNOT see or vote for your own answer (you are '{voter_id}')
2. Score ONLY the numbered answers shown below
3. Use EXACT integer scores based on these criteria

SCORING RUBRIC (use full scale):
- correctness (0-5): Is the answer factually accurate and complete?
  * 5 = Perfect, fully correct
  * 3 = Mostly correct with minor issues
  * 0 = Wrong, refuses to answer, or says "sorry I can't help"
  
- relevance (0-3): Does it directly address the question?
  * 3 = Perfectly on-topic
  * 0 = Off-topic or refuses to answer
  
- specificity (0-3): Does it provide concrete details/examples?
  * 3 = Highly specific with examples
  * 1 = General information only
  * 0 = Vague or no useful detail
  
- safety (0-2): Is it safe and ethical?
  * 2 = Completely safe
  * 0 = Contains harmful content
  
- conciseness (0-1): Is it appropriately brief?
  * 1 = Concise and clear
  * 0 = Overly verbose or too brief

EXAMPLES:
- If answer says "Sorry, I cannot help": correctness=0, relevance=0, specificity=0, safety=2, conciseness=1

Return ONLY this JSON structure (no other text):
{{
  "scores": {{
    "1": {{"correctness": <int>, "relevance": <int>, "specificity": <int>, "safety": <int>, "conciseness": <int>}},
    "2": {{"correctness": <int>, "relevance": <int>, "specificity": <int>, "safety": <int>, "conciseness": <int>}},
    ... for each answer number below
  }},
  "reasoning": "Brief explanation of your evaluation",
  "final_pick": <int>
}}

ANSWERS TO EVALUATE:
{ballot}

Remember: Use the FULL scoring range (0 to max for each category). Be decisive, not neutral."""


def validate_ballot_strict(
    idx_map_peer: dict,
    parsed: dict,
    voter_model: str
) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message)
    Rejects ballots with:
    - Missing candidates
    - All-zero scores (unless answer was truly bad)
    - Self-voting attempts
    """
    if not isinstance(parsed, dict) or "scores" not in parsed:
        return False, "Missing 'scores' key"
    
    scores = parsed["scores"]
    if not isinstance(scores, dict):
        return False, "Scores must be a dictionary"
    
    # Check all candidates are scored
    expected_keys = {str(i) for i in idx_map_peer.keys()}
    actual_keys = set(scores.keys())
    
    missing = expected_keys - actual_keys
    if missing:
        return False, f"Missing scores for candidates: {missing}"
    
    extra = actual_keys - expected_keys
    if extra:
        return False, f"Unexpected candidate keys: {extra}"
    
    # Check for self-voting (voter's own model shouldn't appear)
    for idx, mid in idx_map_peer.items():
        if mid == voter_model:
            return False, f"Voter {voter_model} attempted to score itself at position {idx}"
    
    # Check all scores are valid dictionaries
    required_metrics = {"correctness", "relevance", "specificity", "safety", "conciseness"}
    for key, score_dict in scores.items():
        if not isinstance(score_dict, dict):
            return False, f"Score for candidate {key} is not a dictionary"
        
        actual_metrics = set(score_dict.keys())
        if actual_metrics != required_metrics:
            return False, f"Candidate {key} has wrong metrics: {actual_metrics}"
        
        # Validate ranges
        ranges = {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}
        for metric, value in score_dict.items():
            if not isinstance(value, int):
                return False, f"Candidate {key}, {metric}: must be integer, got {type(value)}"
            if not (0 <= value <= ranges[metric]):
                return False, f"Candidate {key}, {metric}={value} out of range [0, {ranges[metric]}]"
    
    # Warn about all-zero ballots (but don't reject - might be legitimate)
    all_zeros = all(
        sum(sc.values()) == 0 
        for sc in scores.values()
    )
    if all_zeros:
        return True, "WARNING: All scores are zero"
    
    return True, "Valid"


async def vote_one_improved(
    session: aiohttp.ClientSession,
    base_url: str,
    voter_model_id: str,
    question: str,
    answers: dict,
    selected_models: list
) -> tuple[str, Optional[dict], str]:
    """
    Simplified voting: Try json_schema once, then text mode once.
    Return early on first valid ballot.
    Returns: (voter_id, parsed_ballot, status_message)
    """
    peer_models = [m for m in selected_models if m != voter_model_id]
    idx_map_peer = {i + 1: m for i, m in enumerate(peer_models)}
    
    _dbg(f"VOTE index map (voter={voter_model_id})", {i: mid for i, mid in idx_map_peer.items()})
    
    prompt = improved_voting_prompt(voter_model_id, idx_map_peer, question, answers)
    _dbg(f"VOTE prompt (voter={voter_model_id})", prompt)
    
    # Try 1: JSON Schema (most reliable)
    schema = _build_voting_json_schema({
        "correctness": 5,
        "relevance": 3,
        "specificity": 3,
        "safety": 2,
        "conciseness": 1
    })
    
    try:
        content = await call_model(
            session, base_url, voter_model_id, prompt,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "vote_ballot", "schema": schema}
            }
        )
        
        _dbg(f"VOTE raw OUTPUT json_schema (voter={voter_model_id})", content)
        
        parsed = safe_load_vote_json(content)
        if parsed:
            is_valid, msg = validate_ballot_strict(idx_map_peer, parsed, voter_model_id)
            if is_valid:
                # Convert string keys to model IDs for consistency
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", "")
                }
                _dbg(f"VOTE ACCEPTED json_schema (voter={voter_model_id})", normalized)
                return voter_model_id, normalized, f"Valid ballot via json_schema ({msg})"
            else:
                _dbg(f"VOTE REJECTED json_schema (voter={voter_model_id})", msg)
    
    except Exception as e:
        _dbg(f"VOTE ERROR json_schema (voter={voter_model_id})", str(e))
    
    # Try 2: Plain text (fallback)
    try:
        content = await call_model(
            session, base_url, voter_model_id, prompt,
            temperature=0.0,
            response_format=None
        )
        
        _dbg(f"VOTE raw OUTPUT text (voter={voter_model_id})", content)
        
        parsed = safe_load_vote_json(content)
        if parsed:
            is_valid, msg = validate_ballot_strict(idx_map_peer, parsed, voter_model_id)
            if is_valid:
                normalized = {
                    "scores": {idx_map_peer[int(k)]: v for k, v in parsed["scores"].items()},
                    "final_pick": parsed.get("final_pick"),
                    "reasoning": parsed.get("reasoning", "")
                }
                _dbg(f"VOTE ACCEPTED text (voter={voter_model_id})", normalized)
                return voter_model_id, normalized, f"Valid ballot via text ({msg})"
            else:
                _dbg(f"VOTE REJECTED text (voter={voter_model_id})", msg)
    
    except Exception as e:
        _dbg(f"VOTE ERROR text (voter={voter_model_id})", str(e))
    
    # Failed - return empty ballot with clear error
    _dbg(f"VOTE FAILED ALL ATTEMPTS (voter={voter_model_id})", "No valid ballot produced")
    return voter_model_id, None, f"Failed to obtain valid ballot from {short_id(voter_model_id)}"


# -----------------------
# Council orchestration (IMPROVED)
# -----------------------
async def council_round(
    base_url: str,
    selected_models: List[str],
    question: str,
    roles: Dict[str, Optional[str]],
    status_cb,
    max_concurrency: int = 1,
    voter_override: Optional[List[str]] = None
):
    status_cb("Collecting answers…")
    async with aiohttp.ClientSession() as session:

        async def answer_one(model_id: str) -> tuple[str, str]:
            user_prompt = compose_user_prompt(question, roles.get(model_id))
            try:
                ans = await call_model(session, base_url, model_id, user_prompt, temperature=0.5, sys_prompt=None)
                if isinstance(ans, dict):
                    ans = json.dumps(ans, ensure_ascii=False)
                return model_id, ans
            except Exception as e1:
                try:
                    ans = await call_model(session, base_url, model_id, question, temperature=0.5, sys_prompt=None)
                    if isinstance(ans, dict):
                        ans = json.dumps(ans, ensure_ascii=False)
                    return model_id, ans
                except Exception as e2:
                    return model_id, f"[ERROR fetching answer]\nFirst attempt failed: {e1}\nSecond attempt failed: {e2}"

        pairs = await run_limited(max_concurrency, [lambda m=m: answer_one(m) for m in selected_models])
        answers = {m: a for m, a in pairs}
        errors = {m: a for m, a in pairs if isinstance(a, str) and a.startswith("[ERROR")}

        # ---------- IMPROVED VOTING ----------
        status_cb("Models are voting…")
        
        vote_results = await run_limited(
            max_concurrency,
            [lambda m=m: vote_one_improved(session, base_url, m, question, answers, selected_models) 
             for m in (voter_override if voter_override else selected_models)]
        )
        
        # Separate valid and invalid ballots
        valid_votes = {}
        invalid_votes = {}
        vote_messages = {}
        
        for voter_id, ballot, msg in vote_results:
            vote_messages[voter_id] = msg
            if ballot is not None:
                valid_votes[voter_id] = ballot
            else:
                invalid_votes[voter_id] = msg
        
        # Report invalid ballots
        if invalid_votes:
            invalid_list = [short_id(m) for m in invalid_votes.keys()]
            status_cb(f"⚠ {len(invalid_votes)}/{len(selected_models)} invalid ballots: {', '.join(invalid_list)}")
        
        _dbg("VALID VOTES", valid_votes)
        _dbg("INVALID VOTES", invalid_votes)
        
        # Aggregate only valid votes
        RUBRIC = {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}
        totals = {mid: 0 for mid in selected_models}
        
        for voter, ballot in valid_votes.items():
            for candidate_mid, score_dict in ballot["scores"].items():
                weighted = sum(score_dict[k] * RUBRIC[k] for k in RUBRIC.keys())
                totals[candidate_mid] = totals.get(candidate_mid, 0) + weighted
        
        # Add small bonus for explicit final_pick (tie-breaker)
        for voter, ballot in valid_votes.items():
            fp = ballot.get("final_pick")
            if isinstance(fp, int):
                peer_models = [m for m in selected_models if m != voter]
                if 1 <= fp <= len(peer_models):
                    picked_mid = peer_models[fp - 1]
                    totals[picked_mid] = totals.get(picked_mid, 0) + 1
        
        # Winner selection with clear tie-breaking
        if not valid_votes:
            status_cb("⚠ NO VALID VOTES - selecting first model by default")
            winner = selected_models[0]
            tally = totals
        else:
            max_score = max(totals.values())
            contenders = [m for m, score in totals.items() if score == max_score]
            
            if len(contenders) > 1:
                # Tie-break by position in selected_models (deterministic)
                winner = min(contenders, key=lambda m: selected_models.index(m))
                status_cb(f"Tie between {len(contenders)} models, selected by order: {short_id(winner)}")
            else:
                winner = contenders[0]
            
            tally = totals
        
        _dbg("VOTE AGGREGATION — totals (points per candidate)", tally)
        _dbg("VOTE AGGREGATION — winner", winner)
        
        details = {
            "question": question,
            "answers": answers,
            "valid_votes": valid_votes,
            "invalid_votes": invalid_votes,
            "vote_messages": vote_messages,
            "tally": tally,
            "errors": errors,
            "winner": winner,
            "participation_rate": len(valid_votes) / len(selected_models) if selected_models else 0.0
        }
        
        status_cb(f"Vote complete. Valid: {len(valid_votes)}/{len(selected_models)}")
        return answers, winner, details, tally

# -----------------------
# Qt GUI
# -----------------------
class CouncilWindow(QtWidgets.QMainWindow):
    status_signal = QtCore.Signal(str)
    result_signal = QtCore.Signal(object)   # (question, answers, winner, details, tally)
    error_signal  = QtCore.Signal(str)
    done_signal   = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolyCouncil")
        self.resize(1320, 900)

        ensure_db()

        self.models: List[str] = []
        self.model_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.model_tabs: Dict[str, QtWidgets.QWidget] = {}
        self.model_texts: Dict[str, QtWidgets.QPlainTextEdit] = {}
        self.use_roles = False

        self._build_ui()
        self._connect_signals()

        # restore base URL
        self.base_edit.setText(CURRENT_BASE)
        # restore single-voter
        try:
            s = load_settings()
            self.single_voter_check.setChecked(bool(s.get("single_voter_enabled", False)))
            sel = s.get("single_voter_model", "") or ""
            if sel:
                self.single_voter_combo.addItem(sel)
                self.single_voter_combo.setCurrentText(sel)
        except Exception:
            pass
        self._connect_base()

        self._refresh_leaderboard()

    # ----- UI -----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10,10,10,8)
        layout.setSpacing(8)
        self.setCentralWidget(central)

        # Top Bar
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)

        self.base_edit = QtWidgets.QLineEdit()
        self.base_edit.setPlaceholderText("http://localhost:1234")
        self.connect_btn = QtWidgets.QPushButton("Connect")

        self.roles_check = QtWidgets.QCheckBox("Enable personas (answers only)")
        self.single_voter_check = QtWidgets.QCheckBox("Single-voter")
        self.single_voter_combo = QtWidgets.QComboBox()
        self.single_voter_combo.setMinimumWidth(220)

        # Concurrency controls + warning on the RIGHT
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
        top.addWidget(self.roles_check)
        top.addWidget(self.single_voter_check)
        top.addWidget(self.single_voter_combo)
        top.addStretch(1)
        top.addLayout(conc_box)

        layout.addLayout(top)

        # Main Grid
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(0, 1)
        layout.addLayout(grid, stretch=1)

        # Left pane
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
        self.models_inner = QtWidgets.QWidget()
        self.models_layout = QtWidgets.QVBoxLayout(self.models_inner)
        self.models_layout.setContentsMargins(6,6,6,6)
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
        left_widget.setMinimumWidth(320)
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

    def _connect_signals(self):
        self.connect_btn.clicked.connect(self._connect_base)
        self.refresh_models_btn.clicked.connect(self._refresh_models_clicked)
        self.select_all_btn.clicked.connect(self._select_all_models)
        self.clear_btn.clicked.connect(self._clear_models)
        self.reset_btn.clicked.connect(self._reset_leaderboard_clicked)
        self.send_btn.clicked.connect(self._send)
        self.prompt_edit.returnPressed.connect(self._send)
        self.roles_check.stateChanged.connect(self._roles_toggled)
        self.single_voter_check.stateChanged.connect(self._single_voter_toggled)
        self.single_voter_combo.currentTextChanged.connect(self._single_voter_changed)

        self.status_signal.connect(self._set_status)
        self.result_signal.connect(self._handle_result)
        self.error_signal.connect(self._handle_error)
        self.done_signal.connect(self._done)

    # ----- actions -----
    def _single_voter_toggled(self, state):
        enabled = (state == QtCore.Qt.CheckState.Checked)
        save_settings({"single_voter_enabled": enabled, "single_voter_model": self.single_voter_combo.currentText()})

    def _single_voter_changed(self, text: str):
        save_settings({"single_voter_model": text})

    def _roles_toggled(self, state):
        self.use_roles = (state == QtCore.Qt.CheckState.Checked)

    def _connect_base(self):
        global CURRENT_BASE, MODEL_READY
        base = self.base_edit.text().strip()
        if not base:
            QtWidgets.QMessageBox.information(self, "Base URL", "Please enter a base URL such as http://localhost:1234")
            return
        CURRENT_BASE = base
        save_settings({"base_url": CURRENT_BASE})
        MODEL_READY = {}
        self._set_status(f"Connecting to {CURRENT_BASE} …")
        self._busy(True)
        QtCore.QTimer.singleShot(50, self._refresh_models)

    def _refresh_models_clicked(self):
        self._set_status("Refreshing models …")
        self._busy(True)
        QtCore.QTimer.singleShot(50, self._refresh_models)

    def _refresh_models(self):
        self.models = fetch_models_from_lmstudio(CURRENT_BASE)
        loaded_now = fetch_loaded_models_via_sdk()
        if not loaded_now:
            loaded_now = fetch_loaded_models_best_effort(CURRENT_BASE) if self.models else []
        self._populate_models(prechecked=loaded_now)
        self._busy(False)
        if not self.models:
            self._set_status("No models found. Check base URL or LM Studio.")
        else:
            suffix = f" (loaded: {len(loaded_now)})" if loaded_now else ""
            self._set_status(f"Found {len(self.models)} models{suffix}.")

    def _populate_models(self, prechecked: List[str] = None):
        prev_checked = {m for m, cb in self.model_checks.items() if cb.isChecked()}
        for i in reversed(range(self.models_layout.count())):
            item = self.models_layout.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.model_checks.clear()

        pre = set(prechecked or [])
        pre |= prev_checked
        pre |= {m for m, ok in MODEL_READY.items() if ok}

        for m in self.models:
            cb = QtWidgets.QCheckBox(m)
            cb.setChecked(m in pre)
            cb.stateChanged.connect(self._checkbox_toggled)
            self.models_layout.insertWidget(self.models_layout.count()-1, cb)
            self.model_checks[m] = cb
            # keep single-voter dropdown synced with available models
            try:
                self.single_voter_combo.blockSignals(True)
                self.single_voter_combo.clear()
                for _mid in self.models:
                    self.single_voter_combo.addItem(_mid)
                last = load_settings().get("single_voter_model", "")
                if last:
                    ix = self.single_voter_combo.findText(last)
                    if ix >= 0:
                        self.single_voter_combo.setCurrentIndex(ix)
            finally:
                self.single_voter_combo.blockSignals(False)


        for m in pre:
            MODEL_READY[m] = True

    def _select_all_models(self):
        for cb in self.model_checks.values():
            cb.setChecked(True)

    def _clear_models(self):
        for cb in self.model_checks.values():
            cb.setChecked(False)
        self._checkbox_toggled()

    def _checkbox_toggled(self):
        to_unload = [m for m, cb in self.model_checks.items() if (not cb.isChecked()) and MODEL_READY.get(m)]
        if to_unload:
            def worker():
                best_effort_unload(CURRENT_BASE, to_unload, self._queue_status)
            threading.Thread(target=worker, daemon=True).start()

    def _send(self):
        question = self.prompt_edit.text().strip()
        if not question:
            return
        selected = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QtWidgets.QMessageBox.information(self, "Select Models", "Please select at least one model.")
            return

        self._prune_inactive_tabs(selected)
        self._prepare_tabs(selected)

        self._append_chat(f"You: {question}\n")
        self._set_status("Checking models …")
        self._busy(True)
        self.prompt_edit.setEnabled(False)
        self.send_btn.setEnabled(False)

        maxc = self.conc_spin.value()
        roles = assign_random_roles(selected, self.use_roles)

        def worker():
            try:
                ready = asyncio.run(ensure_models_ready_once(CURRENT_BASE, selected, self._queue_status, max_concurrency=maxc))
                not_ready = [m for m, ok in ready.items() if not ok]
                if not_ready:
                    self.error_signal.emit(f"Failed to load: {', '.join(not_ready)}")
                    self.done_signal.emit()
                    return

                self._queue_status("Starting council …")
                                # Sync single-voter dropdown with the current selected models
                try:
                    self.single_voter_combo.blockSignals(True)
                    self.single_voter_combo.clear()
                    for _mid in selected:
                        self.single_voter_combo.addItem(_mid)
                    last = load_settings().get("single_voter_model", "")
                    if last:
                        ix = self.single_voter_combo.findText(last)
                        if ix >= 0:
                            self.single_voter_combo.setCurrentIndex(ix)
                finally:
                    self.single_voter_combo.blockSignals(False)
                answers, winner, details, tally = asyncio.run(
                    council_round(CURRENT_BASE, selected, question, roles, self._queue_status, max_concurrency=maxc, voter_override=( [self.single_voter_combo.currentText().strip()] if self.single_voter_check.isChecked() else None ))
                )
                record_vote(question, winner, details)
                self.result_signal.emit((question, answers, winner, details, tally))
            except Exception as e:
                self.error_signal.emit(str(e))
            finally:
                self.done_signal.emit()

        threading.Thread(target=worker, daemon=True).start()

    # ----- UI helpers -----
    def _prepare_tabs(self, selected_models: List[str]):
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

    def _prune_inactive_tabs(self, active_models: List[str]):
        inactive = [m for m in list(self.model_tabs.keys()) if m not in active_models]
        for m in inactive:
            idx = self.tabs.indexOf(self.model_tabs[m])
            if idx >= 0:
                self.tabs.removeTab(idx)
            self.model_tabs.pop(m, None)
            self.model_texts.pop(m, None)

    def _append_chat(self, text: str):
        self.chat_view.appendPlainText(text)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _busy(self, on: bool):
        self.busy.setVisible(on)

    def _queue_status(self, msg: str):
        self.status_signal.emit(msg)

    def _handle_result(self, payload):
        question, answers, winner, details, tally = payload

        # Per-model tabs: show each model's answer and the SCORES it received from each voter
        valid_votes = details.get("valid_votes", {})
        invalid_votes = details.get("invalid_votes", {})
        
        for m, a in answers.items():
            if m in self.model_texts:
                received_lines = []
                # show per-voter metrics
                for voter, ballot in valid_votes.items():
                    sc = ballot.get("scores", {}).get(m)
                    if sc:
                        # show weighted total
                        weighted = sc["correctness"]*5 + sc["relevance"]*3 + sc["specificity"]*3 + sc["safety"]*2 + sc["conciseness"]*1
                        received_lines.append(f"{short_id(voter)}: {sc} → total {weighted}")
                
                scores_text = "\n".join(received_lines) if received_lines else "No valid scores received for this answer."
                
                # Add reasoning if available
                reasoning_lines = []
                for voter, ballot in valid_votes.items():
                    fp = ballot.get("final_pick")
                    reasoning = ballot.get("reasoning", "")
                    if reasoning:
                        reasoning_lines.append(f"{short_id(voter)}: {reasoning}")
                
                reasoning_text = "\n".join(reasoning_lines) if reasoning_lines else ""
                
                display = f"{a}\n\n---\nVotes received:\n{scores_text}"
                if reasoning_text:
                    display += f"\n\nVoter reasoning:\n{reasoning_text}"
                
                self.model_texts[m].setPlainText(display)

        participation_rate = details.get("participation_rate", 0.0)
        num_voters = len(valid_votes)
        
        votes_line = ", ".join(
            f"{short_id(k)}:{v}"
            for k, v in sorted(tally.items(), key=lambda x: -x[1])
        )
        avg_line = ", ".join(
            f"{short_id(k)}:{(v/num_voters):.1f}"
            for k, v in sorted(tally.items(), key=lambda x: -x[1])
        ) if num_voters else "n/a"

        winner_display = short_id(winner)
        win_text = answers.get(winner, "")
        
        result_text = (
            f"Winner: {winner_display}\n"
            f"Valid votes: {len(valid_votes)}/{len(answers)} ({participation_rate:.0%})\n"
            f"Total points → {votes_line or 'n/a'}\n"
            f"Avg points (per voter) → {avg_line}\n"
        )
        
        if invalid_votes:
            invalid_list = [short_id(m) for m in invalid_votes.keys()]
            result_text += f"⚠ Invalid ballots from: {', '.join(invalid_list)}\n"
        
        result_text += f"\n{win_text}\n"
        
        self._append_chat(result_text)
        self._refresh_leaderboard()
        self._set_status("Ready.")

    def _handle_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._set_status("Error.")

    def _done(self):
        self._busy(False)
        self.prompt_edit.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.prompt_edit.clear()
        self.prompt_edit.setFocus()

    def _reset_leaderboard_clicked(self):
        if QtWidgets.QMessageBox.question(self, "Reset Leaderboard", "This will erase all vote history. Continue?") == QtWidgets.QMessageBox.StandardButton.Yes:
            reset_db()
            self._refresh_leaderboard()
            self._set_status("Leaderboard reset.")

    def _refresh_leaderboard(self):
        self.leader_list.clear()
        rows = load_leaderboard()
        if not rows:
            self.leader_list.addItem("No votes yet.")
            return
        total = sum(c for _, c in rows)
        for model, cnt in rows:
            pct = (cnt / total) * 100 if total else 0.0
            short = short_id(model)
            self.leader_list.addItem(f"{short} — {cnt} wins ({pct:.1f}%)")

# -----------------------
# Main
# -----------------------
def apply_system_theme(app: QtWidgets.QApplication):
    if qdarktheme:
        try:
            qdarktheme.setup_theme("auto")
            return
        except Exception:
            pass

if __name__ == "__main__":
    ensure_db()
    qtapp = QtWidgets.QApplication([])
    apply_system_theme(qtapp)
    win = CouncilWindow()
    win.show()
    qtapp.exec()