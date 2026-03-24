from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from PySide6.QtCore import QObject, Property, QUrl, Signal, Slot

from core import leaderboard as leaderboard_store
from core.api_client import fetch_models
from core.app_state import app_data_dir, ensure_app_dirs, migrate_legacy_state
from core.discussion_manager import DiscussionManager
from core.personas import (
    add_user_persona,
    build_persona_config,
    cleanup_persona_assignments,
    clear_persona_assignment,
    delete_user_persona,
    merge_persona_library,
    persona_names,
    rename_persona_assignments,
    sort_personas_inplace,
    update_user_persona,
)
from core.provider_config import (
    API_SERVICE_CUSTOM,
    API_SERVICE_LABELS,
    API_SERVICE_OPENAI,
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
from core.session_history import list_sessions, load_session, save_session
from core.settings_store import load_settings, save_settings
from core.tool_manager import FileParser, ModelCapabilityDetector
from council import council_round, normalize_rubric_weights, short_id

try:
    from qasync import asyncSlot
except Exception:  # pragma: no cover
    def asyncSlot(*_args: Any, **_kwargs: Any):
        def decorator(fn):
            return fn

        return decorator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


class QmlBridge(QObject):
    modelsChanged = Signal()
    resultsReady = Signal()
    statusUpdated = Signal(str)
    discussionUpdated = Signal(str)
    stateChanged = Signal()
    feedChanged = Signal()
    leaderboardChanged = Signal()
    attachmentsChanged = Signal()
    sessionsChanged = Signal()
    providerProfilesChanged = Signal()
    personasChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        migrate_legacy_state()
        ensure_app_dirs()
        leaderboard_store.ensure_db()

        self._status_message = "Initializing..."
        self._is_busy = False
        self._run_state = "locked"
        self._mode = "deliberation"

        self._provider_type = PROVIDER_LM_STUDIO
        self._base_url = provider_defaults(PROVIDER_LM_STUDIO)[0]
        self._api_key = ""
        self._model_path = provider_defaults(PROVIDER_LM_STUDIO)[1]
        self._api_service = API_SERVICE_CUSTOM

        self._timeout_seconds = 120
        self._max_concurrency = 1
        self._use_roles = False
        self._web_search_enabled = False
        self._rubric_weights = normalize_rubric_weights(None)

        self._models: List[Dict[str, Any]] = []
        self._personas: List[dict] = []
        self._persona_assignments: Dict[str, str] = {}
        self._leaderboard: List[Dict[str, Any]] = []
        self._feed_entries: List[Dict[str, Any]] = []
        self._attachments: List[Dict[str, str]] = []
        self._provider_profiles: List[Dict[str, str]] = []
        self._recent_sessions: List[Dict[str, str]] = []
        self._active_stream_entries: Dict[str, int] = {}

        self._last_session_record: Optional[Dict[str, Any]] = None
        self._last_export_path = ""
        self._results_summary = ""
        self._winner_name = ""
        self._winner_text = ""
        self._answers_json = ""
        self._ballots_json = ""
        self._session_json = ""
        self._discussion_text = ""
        self._last_question = ""

        self._provider_options = [
            {"value": key, "label": label}
            for key, label in PROVIDER_LABELS.items()
        ]
        self._api_service_options = [
            {"value": key, "label": label}
            for key, label in API_SERVICE_LABELS.items()
        ]
        self._mode_options = [
            {"value": "deliberation", "label": "Weighted Vote"},
            {"value": "discussion", "label": "Collaborative Discussion"},
        ]

        self._restore_state()
        self._refresh_leaderboard()
        self._refresh_recent_sessions()
        self._load_latest_session()
        self._recompute_run_state()
        self._set_status("Ready. Choose a provider, load models, and start a run.")

    def _restore_state(self) -> None:
        settings = load_settings()
        self._provider_type = normalize_provider_type(settings.get("provider_type", PROVIDER_LM_STUDIO))
        self._api_service = normalize_api_service(settings.get("api_service", API_SERVICE_CUSTOM))
        self._api_key = str(settings.get("api_key", "") or "")
        self._timeout_seconds = max(15, int(settings.get("timeout_seconds", 120) or 120))
        self._max_concurrency = max(1, min(8, int(settings.get("max_concurrency", 1) or 1)))
        self._use_roles = bool(settings.get("roles_enabled", False))
        self._rubric_weights = normalize_rubric_weights(settings.get("rubric_weights"))

        default_base, default_model_path = provider_defaults(self._provider_type)
        raw_base = str(settings.get("base_url", default_base) or default_base)
        self._base_url = canonicalize_base_url(self._provider_type, raw_base) or default_base
        self._model_path = str(settings.get("model_path", default_model_path) or default_model_path)

        loaded_profiles = settings.get("provider_profiles", [])
        if isinstance(loaded_profiles, list):
            self._provider_profiles = [
                {
                    "id": str(profile.get("id", "")),
                    "provider_type": normalize_provider_type(profile.get("provider_type", PROVIDER_LM_STUDIO)),
                    "api_service": normalize_api_service(profile.get("api_service", API_SERVICE_CUSTOM)),
                    "base_url": str(profile.get("base_url", "")),
                    "api_key": str(profile.get("api_key", "")),
                    "model_path": str(profile.get("model_path", "")),
                }
                for profile in loaded_profiles
                if isinstance(profile, dict)
            ]
        if not self._provider_profiles:
            self._provider_profiles = [self._current_profile_payload()]

        self._personas = merge_persona_library(settings.get("personas", []))
        self._persona_assignments = dict(settings.get("persona_assignments", {}) or {})
        self._persona_assignments, _ = cleanup_persona_assignments(self._personas, self._persona_assignments)

    def _persist_provider_settings(self) -> None:
        save_settings(
            {
                "provider_type": self._provider_type,
                "base_url": self._base_url,
                "api_key": self._api_key,
                "model_path": self._model_path,
                "api_service": self._api_service,
            }
        )

    def _persist_runtime_settings(self) -> None:
        save_settings(
            {
                "roles_enabled": self._use_roles,
                "max_concurrency": self._max_concurrency,
                "timeout_seconds": self._timeout_seconds,
                "personas": self._personas,
                "persona_assignments": self._persona_assignments,
                "provider_profiles": self._provider_profiles,
                "rubric_weights": self._rubric_weights,
            }
        )

    def _current_provider_config(self) -> ProviderConfig:
        return make_provider_config(
            self._provider_type,
            self._base_url,
            self._api_key,
            self._model_path,
            self._api_service,
        )

    def _provider_tag(self, provider: ProviderConfig) -> str:
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and provider.api_service != API_SERVICE_CUSTOM:
            return api_service_label(provider.api_service)
        return provider_label(provider.provider_type)

    def _current_profile_payload(self) -> Dict[str, str]:
        provider = self._current_provider_config()
        return {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
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
        key_status = "key set" if profile.get("api_key", "") else "no key"
        return f"{provider_text} - {base} ({key_status})"

    def _set_status(self, message: str) -> None:
        self._status_message = message
        self.statusUpdated.emit(message)
        self.stateChanged.emit()

    def _recompute_run_state(self) -> None:
        if self._is_busy:
            self._run_state = "running"
        elif self.selectedCount:
            self._run_state = "ready"
        else:
            self._run_state = "locked"
        self.stateChanged.emit()

    def _set_busy(self, busy: bool) -> None:
        self._is_busy = busy
        self._recompute_run_state()

    def _model_row(self, model: str) -> Optional[Dict[str, Any]]:
        for row in self._models:
            if row["displayName"] == model:
                return row
        return None

    def _has_provider_model(self, provider: ProviderConfig, raw_model: str) -> bool:
        for row in self._models:
            if (
                row.get("rawModel") == raw_model
                and row.get("providerType") == provider.provider_type
                and row.get("baseUrl") == provider.base_url
                and row.get("modelPath") == provider.model_path
                and row.get("apiService") == provider.api_service
            ):
                return True
        return False

    def _make_display_model_name(self, provider: ProviderConfig, raw_model: str) -> str:
        existing = {row["displayName"] for row in self._models}
        if raw_model not in existing:
            return raw_model
        provider_name = f"{raw_model} [{self._provider_tag(provider)}]"
        if provider_name not in existing:
            return provider_name
        index = 2
        while True:
            candidate = f"{provider_name} ({index})"
            if candidate not in existing:
                return candidate
            index += 1

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()

    def _refresh_model_search_text(self, row: Dict[str, Any]) -> None:
        raw_model = str(row.get("rawModel", ""))
        display_name = str(row.get("displayName", raw_model))
        provider = str(row.get("providerLabel", ""))
        capability = str(row.get("capabilitySummary", ""))
        service = api_service_label(str(row.get("apiService", API_SERVICE_CUSTOM)))
        variants = [
            raw_model,
            display_name,
            provider,
            capability,
            service,
            raw_model.replace("/", " "),
            raw_model.replace("-", " "),
            raw_model.replace(":", " "),
            display_name.replace("[", " ").replace("]", " "),
        ]
        row["searchText"] = self._normalize_search_text(" ".join(part for part in variants if part))

    async def _detect_model_capabilities(self) -> None:
        if not self._models:
            return
        async with aiohttp.ClientSession() as session:
            model_data_cache: Dict[str, dict] = {}
            for row in self._models:
                provider = make_provider_config(
                    row["providerType"],
                    row["baseUrl"],
                    row.get("apiKey", ""),
                    row.get("modelPath", ""),
                    row.get("apiService", API_SERVICE_CUSTOM),
                )
                raw_model = row["rawModel"]
                has_vl_name = "vl" in raw_model.lower()
                try:
                    if provider.provider_type == PROVIDER_OLLAMA:
                        row["hasWebSearch"] = False
                        row["hasVision"] = any(
                            token in raw_model.lower()
                            for token in ("vision", "vl", "llava", "multimodal")
                        )
                    else:
                        cache_key = "|".join(
                            (
                                provider.provider_type,
                                provider.base_url,
                                provider.model_path,
                                provider.api_service,
                                "1" if provider.api_key else "0",
                            )
                        )
                        if cache_key not in model_data_cache:
                            model_data_cache[cache_key] = await ModelCapabilityDetector.fetch_models_data(
                                session,
                                provider.base_url,
                                provider.api_key,
                                provider.model_path,
                            )
                        model_data = model_data_cache[cache_key]
                        row["hasWebSearch"] = ModelCapabilityDetector.detect_web_search_from_data(model_data, raw_model)
                        row["hasVision"] = ModelCapabilityDetector.detect_visual_from_data(model_data, raw_model) or has_vl_name
                except Exception:
                    row["hasWebSearch"] = False
                    row["hasVision"] = has_vl_name
                row["capabilitySummary"] = self._capability_summary(row)
                self._refresh_model_search_text(row)
        self.modelsChanged.emit()

    def _capability_summary(self, row: Dict[str, Any]) -> str:
        parts = [row["providerLabel"]]
        caps = []
        if row.get("hasVision"):
            caps.append("vision")
        if row.get("hasWebSearch"):
            caps.append("web")
        if caps:
            parts.append(", ".join(caps))
        latency_ms = row.get("latencyMs")
        if isinstance(latency_ms, int) and latency_ms > 0:
            parts.append(f"{latency_ms} ms")
        return " | ".join(parts)

    def _refresh_leaderboard(self) -> None:
        raw = leaderboard_store.load_leaderboard()
        total = sum(count for _, count in raw)
        self._leaderboard = [
            {
                "model": model_id,
                "label": short_id(model_id),
                "wins": count,
                "share": round((count / total) * 100, 1) if total else 0.0,
            }
            for model_id, count in raw
        ]
        self.leaderboardChanged.emit()
        self.stateChanged.emit()

    def _refresh_recent_sessions(self) -> None:
        sessions: List[Dict[str, str]] = []
        for path in list_sessions(limit=8):
            try:
                record = load_session(path)
            except Exception:
                record = None
            saved_at = ""
            mode = ""
            question = ""
            if record:
                saved_at = str(record.get("saved_at", ""))
                mode = str(record.get("mode", ""))
                question = str(record.get("question", ""))
            sessions.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "savedAt": saved_at,
                    "mode": mode,
                    "question": question,
                }
            )
        self._recent_sessions = sessions
        self.sessionsChanged.emit()

    def _append_feed_entry(self, kind: str, title: str, body: str, meta: str = "") -> None:
        self._feed_entries.append(
            {
                "kind": kind,
                "title": title,
                "body": body,
                "meta": meta,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        self._discussion_text = "\n\n".join(
            f"[{entry['timestamp']}] {entry['title']}\n{entry['body']}"
            for entry in self._feed_entries
        )
        self.feedChanged.emit()
        self.discussionUpdated.emit(self._discussion_text)
        self.stateChanged.emit()

    def _rebuild_discussion_text(self) -> None:
        self._discussion_text = "\n\n".join(
            f"[{entry['timestamp']}] {entry['title']}\n{entry['body']}"
            for entry in self._feed_entries
        )

    def _upsert_stream_feed_entry(
        self,
        stream_key: str,
        kind: str,
        title: str,
        body: str,
        meta: str = "",
    ) -> None:
        index = self._active_stream_entries.get(stream_key)
        if index is None or index >= len(self._feed_entries):
            self._feed_entries.append(
                {
                    "kind": kind,
                    "title": title,
                    "body": body,
                    "meta": meta,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )
            self._active_stream_entries[stream_key] = len(self._feed_entries) - 1
        else:
            entry = self._feed_entries[index]
            entry["kind"] = kind
            entry["title"] = title
            entry["body"] = body
            entry["meta"] = meta
        self._rebuild_discussion_text()
        self.feedChanged.emit()
        self.discussionUpdated.emit(self._discussion_text)
        self.stateChanged.emit()

    def _finish_stream_feed_entry(self, stream_key: str) -> None:
        self._active_stream_entries.pop(stream_key, None)

    def _clear_feed(self) -> None:
        self._feed_entries = []
        self._active_stream_entries = {}
        self._discussion_text = ""
        self.feedChanged.emit()
        self.discussionUpdated.emit(self._discussion_text)
        self.stateChanged.emit()

    def _apply_session_record(self, record: Dict[str, Any], *, append_feed: bool) -> None:
        self._last_session_record = record
        self._last_question = str(record.get("question", ""))
        self._session_json = json.dumps(record, indent=2, ensure_ascii=False)

        if record.get("mode") == "discussion":
            transcript = list(record.get("transcript", []) or [])
            synthesis = record.get("synthesis")
            self._winner_name = ""
            self._winner_text = synthesis or "No synthesis available."
            self._answers_json = json.dumps(transcript, indent=2, ensure_ascii=False)
            self._ballots_json = "Discussion mode does not produce ballots."
            self._results_summary = (
                f"Discussion completed with {len(transcript)} turns."
                if transcript
                else "No discussion transcript available."
            )
            if append_feed:
                self._clear_feed()
                self._append_feed_entry("user", "User Prompt", self._last_question)
                for entry in transcript:
                    agent_title = entry.get("agent", "Agent")
                    persona = entry.get("persona")
                    meta = f"Turn {entry.get('turn', '?')}"
                    if persona:
                        meta = f"{meta} · {persona}"
                    self._append_feed_entry("agent", agent_title, str(entry.get("message", "")), meta)
                if synthesis:
                    self._append_feed_entry("result", "Final Synthesis", str(synthesis))
        else:
            answers = dict(record.get("answers", {}) or {})
            winner = str(record.get("winner", ""))
            details = dict(record.get("details", {}) or {})
            tally = dict(record.get("tally", {}) or {})
            self._winner_name = winner
            self._winner_text = str(answers.get(winner, ""))
            self._answers_json = json.dumps(answers, indent=2, ensure_ascii=False)
            self._ballots_json = json.dumps(
                {
                    "valid_votes": details.get("valid_votes", {}),
                    "invalid_votes": details.get("invalid_votes", {}),
                    "tally": tally,
                    "vote_messages": details.get("vote_messages", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
            valid_votes = len(details.get("valid_votes", {}) or {})
            voters_used = len(details.get("voters_used", []) or [])
            self._results_summary = (
                f"Winner: {short_id(winner)} · valid ballots {valid_votes}/{max(1, voters_used)}"
                if winner
                else "No winner recorded."
            )
            if append_feed:
                self._clear_feed()
                self._append_feed_entry("user", "User Prompt", self._last_question)
                if winner and self._winner_text:
                    self._append_feed_entry("result", f"Winning Answer · {short_id(winner)}", self._winner_text)

        self.resultsReady.emit()
        self.stateChanged.emit()

    def _load_latest_session(self) -> None:
        try:
            record = load_session()
        except Exception:
            record = None
        if record:
            self._apply_session_record(record, append_feed=False)

    def _normalize_file_path(self, value: str) -> Optional[Path]:
        if not value:
            return None
        url = QUrl(value)
        if url.isValid() and url.scheme():
            local = url.toLocalFile()
            if local:
                path = Path(local)
            else:
                return None
        else:
            path = Path(value)
        return path if path.exists() else None

    def _collect_context_and_images(self) -> tuple[str, List[str]]:
        context_parts: List[str] = []
        image_urls: List[str] = []
        max_file_size = 50000

        for attachment in self._attachments:
            file_path = Path(attachment["path"])
            suffix = file_path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                encoded = ModelCapabilityDetector.encode_image(file_path)
                if encoded:
                    image_urls.append(encoded)
                continue

            parsed = FileParser.parse_file(file_path)
            if not parsed:
                continue
            if len(parsed) > max_file_size:
                truncated = parsed[:max_file_size]
                last_period = truncated.rfind(".")
                last_newline = truncated.rfind("\n")
                cutoff = max(last_period, last_newline)
                if cutoff > int(max_file_size * 0.8):
                    parsed = parsed[: cutoff + 1] + "\n\n[... file content truncated for processing ...]"
                else:
                    parsed = truncated + "\n\n[... file content truncated for processing ...]"
            context_parts.append(FileParser.format_context_block(parsed, file_path.name))

        context_block = "\n".join(context_parts)
        if len(context_block) > 100000:
            context_block = context_block[:20000] + "\n\n[... additional content truncated ...]"
        return context_block, image_urls

    def _selected_models(self) -> List[str]:
        return [row["displayName"] for row in self._models if row.get("selected")]

    def _persona_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for persona_name in self._persona_assignments.values():
            counts[persona_name] = counts.get(persona_name, 0) + 1
        return counts

    def _persona_entry(self, name: str) -> Optional[Dict[str, Any]]:
        for persona in self._personas:
            if str(persona.get("name", "")) == name:
                return persona
        return None

    def _sync_model_persona_names(self) -> None:
        valid_names = set(persona_names(self._personas))
        for row in self._models:
            display_name = row["displayName"]
            persona_name = self._persona_assignments.get(display_name, "None")
            if persona_name not in valid_names:
                persona_name = "None"
                self._persona_assignments[display_name] = persona_name
            row["personaName"] = persona_name

    def _selected_model_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for row in self._models:
            if not row.get("selected"):
                continue
            provider = make_provider_config(
                row["providerType"],
                row["baseUrl"],
                row.get("apiKey", ""),
                row.get("modelPath", ""),
                row.get("apiService", API_SERVICE_CUSTOM),
            )
            entries.append(
                {
                    "id": row["displayName"],
                    "model": row["rawModel"],
                    "provider": provider,
                }
            )
        return entries

    def _roles_map_for_selected(self, selected: List[str]) -> Dict[str, Optional[str]]:
        if not self._use_roles:
            return {model: None for model in selected}
        names = set(persona_names(self._personas))
        roles: Dict[str, Optional[str]] = {}
        for model in selected:
            persona_name = self._persona_assignments.get(model, "None")
            if persona_name not in names:
                persona_name = "None"
            config = build_persona_config(persona_name, self._personas)
            one_time = config.get("one_time_prompt") or ""
            roles[model] = one_time or None
            self._persona_assignments[model] = persona_name
        self._persist_runtime_settings()
        return roles

    async def _run_deliberation(self, question: str, context_block: str, images: List[str]) -> None:
        selected = self._selected_models()
        roles_map = self._roles_map_for_selected(selected)
        if context_block:
            for model in roles_map:
                roles_map[model] = f"{context_block}\n\n{roles_map[model]}" if roles_map[model] else context_block

        model_entries = self._selected_model_entries()
        self._append_feed_entry("system", "Council", "Collecting answers from the selected models...")

        streamed_answers: Dict[str, str] = {}

        def handle_answer_stream(model_id: str, chunk: str, done: bool) -> None:
            streamed_answers[model_id] = streamed_answers.get(model_id, "") + chunk
            meta = "Streaming response" if not done else "Completed response"
            self._upsert_stream_feed_entry(
                f"answer:{model_id}",
                "agent",
                f"Streaming · {short_id(model_id)}",
                streamed_answers.get(model_id, "") or "Waiting for content...",
                meta,
            )
            if done:
                self._finish_stream_feed_entry(f"answer:{model_id}")

        answers, winner, details, tally = await council_round(
            model_entries,
            question,
            roles_map,
            self._set_status,
            max_concurrency=self._max_concurrency,
            images=images,
            web_search=self._web_search_enabled,
            rubric_weights=self._rubric_weights,
            timeout_seconds=self._timeout_seconds,
            answer_stream_cb=handle_answer_stream,
        )

        if winner:
            leaderboard_store.record_vote(question, winner, details)
        timings_ms = details.get("timings_ms", {}) or {}
        for row in self._models:
            latency = timings_ms.get(row["displayName"])
            row["latencyMs"] = int(latency) if isinstance(latency, (int, float)) else 0
            row["capabilitySummary"] = self._capability_summary(row)
            self._refresh_model_search_text(row)

        record = {
            "mode": "deliberation",
            "question": question,
            "answers": answers,
            "winner": winner,
            "details": details,
            "tally": tally,
            "timings_ms": timings_ms,
        }
        self._apply_session_record(record, append_feed=False)
        if winner:
            self._append_feed_entry("result", f"Winning Answer · {short_id(winner)}", str(answers.get(winner, "")))
        else:
            self._append_feed_entry("system", "No winner selected", "No model returned a usable answer for voting.")
        save_session(record)
        self._refresh_leaderboard()
        self._refresh_recent_sessions()
        self.modelsChanged.emit()
        self._set_status("Deliberation complete." if winner else "Deliberation finished without a winner.")

    async def _run_discussion(self, question: str, context_block: str, images: List[str]) -> None:
        agents = []
        for row in self._models:
            if not row.get("selected"):
                continue
            provider = make_provider_config(
                row["providerType"],
                row["baseUrl"],
                row.get("apiKey", ""),
                row.get("modelPath", ""),
                row.get("apiService", API_SERVICE_CUSTOM),
            )
            persona_name = self._persona_assignments.get(row["displayName"], "None")
            agents.append(
                {
                    "name": row["displayName"],
                    "model": row["rawModel"],
                    "is_active": True,
                    "persona_config": build_persona_config(persona_name, self._personas),
                    "persona_name": persona_name if persona_name != "None" else None,
                    "provider_type": provider.provider_type,
                    "base_url": provider.base_url,
                    "api_key": provider.api_key,
                    "model_path": provider.model_path,
                    "api_service": provider.api_service,
                }
            )

        base_provider = self._selected_model_entries()[0]["provider"]
        manager = DiscussionManager(
            provider_type=base_provider.provider_type,
            base_url=base_provider.base_url,
            api_key=base_provider.api_key,
            model_path=base_provider.model_path,
            agents=agents,
            user_prompt=question,
            context_block=context_block,
            images=images,
            web_search_enabled=self._web_search_enabled,
            status_callback=self._set_status,
            update_callback=self._handle_discussion_update,
            max_turns=10,
            max_concurrency=self._max_concurrency,
            timeout_seconds=self._timeout_seconds,
        )
        transcript, synthesis = await manager.run_discussion()
        record = {
            "mode": "discussion",
            "question": question,
            "transcript": transcript,
            "synthesis": synthesis,
            "timings_ms": {},
        }
        self._apply_session_record(record, append_feed=False)
        if synthesis:
            self._append_feed_entry("result", "Final Synthesis", str(synthesis))
        save_session(record)
        self._refresh_recent_sessions()
        self._set_status("Discussion complete.")

    def _handle_discussion_update(self, entry: Dict[str, Any]) -> None:
        agent_title = entry.get("agent", "Agent")
        meta = f"Turn {entry.get('turn', '?')}"
        persona = entry.get("persona")
        if persona:
            meta = f"{meta} · {persona}"
        stream_key = f"discussion:{entry.get('turn', '?')}:{agent_title}"
        if entry.get("partial"):
            self._upsert_stream_feed_entry(stream_key, "agent", agent_title, str(entry.get("message", "")), meta + " · streaming")
            return
        self._upsert_stream_feed_entry(stream_key, "agent", agent_title, str(entry.get("message", "")), meta)
        self._finish_stream_feed_entry(stream_key)

    @Property(str, notify=statusUpdated)
    def statusMessage(self) -> str:
        return self._status_message

    @Property(bool, notify=stateChanged)
    def isBusy(self) -> bool:
        return self._is_busy

    @Property(str, notify=stateChanged)
    def runState(self) -> str:
        return self._run_state

    @Property(str, notify=discussionUpdated)
    def discussionText(self) -> str:
        return self._discussion_text

    @Property("QVariantList", notify=modelsChanged)
    def models(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._models]

    @Property("QVariantList", notify=leaderboardChanged)
    def leaderboard(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._leaderboard]

    @Property("QVariantList", notify=feedChanged)
    def feedEntries(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._feed_entries]

    @Property("QVariantList", notify=attachmentsChanged)
    def attachments(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self._attachments]

    @Property("QVariantList", notify=providerProfilesChanged)
    def providerProfiles(self) -> List[Dict[str, str]]:
        rows = []
        for profile in self._provider_profiles:
            row = dict(profile)
            row["summary"] = self._profile_summary(profile)
            rows.append(row)
        return rows

    @Property("QVariantList", notify=sessionsChanged)
    def recentSessions(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self._recent_sessions]

    @Property("QVariantList", notify=personasChanged)
    def personaLibrary(self) -> List[Dict[str, Any]]:
        counts = self._persona_counts()
        rows = []
        for persona in self._personas:
            name = str(persona.get("name", ""))
            rows.append(
                {
                    "name": name,
                    "prompt": str(persona.get("prompt") or ""),
                    "builtin": bool(persona.get("builtin", False)),
                    "assignmentCount": counts.get(name, 0),
                }
            )
        return rows

    @Property("QVariantList", constant=True)
    def providerOptions(self) -> List[Dict[str, str]]:
        return self._provider_options

    @Property("QVariantList", constant=True)
    def apiServiceOptions(self) -> List[Dict[str, str]]:
        return self._api_service_options

    @Property("QVariantList", constant=True)
    def modeOptions(self) -> List[Dict[str, str]]:
        return self._mode_options

    @Property(str, notify=stateChanged)
    def providerType(self) -> str:
        return self._provider_type

    @providerType.setter
    def providerType(self, value: str) -> None:
        new_provider = normalize_provider_type(value)
        if new_provider == self._provider_type:
            return
        self._provider_type = new_provider
        default_base, default_model_path = provider_defaults(new_provider)
        if new_provider != PROVIDER_OPENAI_COMPAT:
            self._api_service = API_SERVICE_CUSTOM
            self._base_url = default_base
            self._model_path = default_model_path
        else:
            if self._api_service == API_SERVICE_CUSTOM:
                self._api_service = API_SERVICE_OPENAI
            preset = service_preset(self._api_service)
            self._base_url = preset["base_url"] or default_base
            self._model_path = preset["model_path"] or default_model_path
        self._persist_provider_settings()
        self._set_status(f"Provider set to {provider_label(self._provider_type)}.")

    @Property(str, notify=stateChanged)
    def apiService(self) -> str:
        return self._api_service

    @apiService.setter
    def apiService(self, value: str) -> None:
        new_service = normalize_api_service(value)
        if new_service == self._api_service:
            return
        self._api_service = new_service
        if self._provider_type != PROVIDER_OPENAI_COMPAT and new_service != API_SERVICE_CUSTOM:
            self._provider_type = PROVIDER_OPENAI_COMPAT
        if self._provider_type == PROVIDER_OPENAI_COMPAT and new_service != API_SERVICE_CUSTOM:
            preset = service_preset(new_service)
            self._base_url = preset["base_url"]
            self._model_path = preset["model_path"]
        self._persist_provider_settings()
        self._set_status(f"API service set to {api_service_label(self._api_service)}.")

    @Property(str, notify=stateChanged)
    def baseUrl(self) -> str:
        return self._base_url

    @baseUrl.setter
    def baseUrl(self, value: str) -> None:
        cleaned = canonicalize_base_url(self._provider_type, value.strip())
        if cleaned == self._base_url:
            return
        self._base_url = cleaned
        self._persist_provider_settings()
        self.stateChanged.emit()

    @Property(str, notify=stateChanged)
    def apiKey(self) -> str:
        return self._api_key

    @apiKey.setter
    def apiKey(self, value: str) -> None:
        cleaned = value.strip()
        if cleaned == self._api_key:
            return
        self._api_key = cleaned
        self._persist_provider_settings()
        self.stateChanged.emit()

    @Property(str, notify=stateChanged)
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        new_value = value if value in {"deliberation", "discussion"} else "deliberation"
        if new_value == self._mode:
            return
        self._mode = new_value
        self.stateChanged.emit()

    @Property(bool, notify=stateChanged)
    def webSearchEnabled(self) -> bool:
        return self._web_search_enabled

    @webSearchEnabled.setter
    def webSearchEnabled(self, value: bool) -> None:
        self._web_search_enabled = bool(value)
        self.stateChanged.emit()

    @Property(bool, notify=stateChanged)
    def useRoles(self) -> bool:
        return self._use_roles

    @useRoles.setter
    def useRoles(self, value: bool) -> None:
        checked = bool(value)
        if checked == self._use_roles:
            return
        self._use_roles = checked
        self._persist_runtime_settings()
        self.stateChanged.emit()

    @Property(int, notify=stateChanged)
    def timeoutSeconds(self) -> int:
        return self._timeout_seconds

    @timeoutSeconds.setter
    def timeoutSeconds(self, value: int) -> None:
        clamped = max(15, min(600, int(value)))
        if clamped == self._timeout_seconds:
            return
        self._timeout_seconds = clamped
        self._persist_runtime_settings()
        self.stateChanged.emit()

    @Property(int, notify=stateChanged)
    def maxConcurrency(self) -> int:
        return self._max_concurrency

    @maxConcurrency.setter
    def maxConcurrency(self, value: int) -> None:
        clamped = max(1, min(8, int(value)))
        if clamped == self._max_concurrency:
            return
        self._max_concurrency = clamped
        self._persist_runtime_settings()
        self.stateChanged.emit()

    @Property(int, notify=stateChanged)
    def selectedCount(self) -> int:
        return len(self._selected_models())

    @Property(int, notify=stateChanged)
    def totalModels(self) -> int:
        return len(self._models)

    @Property(bool, notify=stateChanged)
    def canRun(self) -> bool:
        return bool(self._selected_models()) and not self._is_busy

    @Property(bool, notify=resultsReady)
    def hasResults(self) -> bool:
        return self._last_session_record is not None

    @Property(str, notify=stateChanged)
    def resultsPhase(self) -> str:
        if self._is_busy:
            return "in_run"
        if self._last_session_record:
            return "post_run"
        return "pre_run"

    @Property("QVariantMap", notify=stateChanged)
    def runConfig(self) -> Dict[str, Any]:
        provider = self._current_provider_config()
        return {
            "mode": self._mode,
            "providerLabel": self._provider_tag(provider),
            "providerType": provider.provider_type,
            "baseUrl": provider.base_url,
            "apiService": provider.api_service,
            "selectedModels": self._selected_models(),
            "attachments": [item["name"] for item in self._attachments],
            "webSearchEnabled": self._web_search_enabled,
            "useRoles": self._use_roles,
            "timeoutSeconds": self._timeout_seconds,
            "maxConcurrency": self._max_concurrency,
        }

    @Property(str, notify=resultsReady)
    def resultsSummary(self) -> str:
        return self._results_summary

    @Property(str, notify=resultsReady)
    def winnerName(self) -> str:
        return self._winner_name

    @Property(str, notify=resultsReady)
    def winnerText(self) -> str:
        return self._winner_text

    @Property(str, notify=resultsReady)
    def answersJson(self) -> str:
        return self._answers_json

    @Property(str, notify=resultsReady)
    def ballotsJson(self) -> str:
        return self._ballots_json

    @Property(str, notify=resultsReady)
    def sessionJson(self) -> str:
        return self._session_json

    @Property(str, notify=resultsReady)
    def lastQuestion(self) -> str:
        return self._last_question

    @Property(str, notify=resultsReady)
    def lastExportPath(self) -> str:
        return self._last_export_path

    @Property(str, notify=stateChanged)
    def selectedModelsText(self) -> str:
        selected = self._selected_models()
        return ", ".join(short_id(model) for model in selected) if selected else "No models selected"

    @Slot()
    def select_all(self) -> None:
        for row in self._models:
            row["selected"] = True
        self.modelsChanged.emit()
        self._recompute_run_state()

    @Slot()
    def clear_selection(self) -> None:
        for row in self._models:
            row["selected"] = False
        self.modelsChanged.emit()
        self._recompute_run_state()

    @Slot(str)
    def toggle_model(self, model_name: str) -> None:
        row = self._model_row(model_name)
        if not row:
            return
        row["selected"] = not bool(row.get("selected"))
        self.modelsChanged.emit()
        self._recompute_run_state()

    @Slot(str)
    def add_manual_model(self, model_name: str) -> None:
        clean_name = model_name.strip()
        if not clean_name:
            self._set_status("Enter a model name to add.")
            return
        provider = self._current_provider_config()
        if self._has_provider_model(provider, clean_name):
            self._set_status("Model already added for this provider.")
            return
        display_name = self._make_display_model_name(provider, clean_name)
        persona_name = self._persona_assignments.get(display_name, "None")
        self._models.append(
            {
                "displayName": display_name,
                "rawModel": clean_name,
                "providerType": provider.provider_type,
                "providerLabel": self._provider_tag(provider),
                "baseUrl": provider.base_url,
                "apiKey": provider.api_key,
                "modelPath": provider.model_path,
                "apiService": provider.api_service,
                "selected": True,
                "personaName": persona_name,
                "hasVision": False,
                "hasWebSearch": False,
                "latencyMs": 0,
                "capabilitySummary": self._provider_tag(provider),
            }
        )
        self._refresh_model_search_text(self._models[-1])
        self.modelsChanged.emit()
        self._recompute_run_state()
        self._persist_provider_settings()
        self._set_status(f"Added model '{clean_name}'.")

    @asyncSlot()
    async def load_models(self) -> None:
        if self._is_busy:
            self._set_status("Wait for the current operation to finish.")
            return
        provider = self._current_provider_config()
        if provider.provider_type == PROVIDER_OPENAI_COMPAT and not provider.api_key:
            self._set_status("Add an API key before loading hosted provider models.")
            return

        self._set_busy(True)
        self._set_status(f"Loading models from {self._provider_tag(provider)}...")
        try:
            models = await fetch_models(provider, provider_label=provider_label, timeout_sec=self._timeout_seconds)
            added = 0
            for raw_model in models:
                if self._has_provider_model(provider, raw_model):
                    continue
                display_name = self._make_display_model_name(provider, raw_model)
                persona_name = self._persona_assignments.get(display_name, "None")
                row = {
                    "displayName": display_name,
                    "rawModel": raw_model,
                    "providerType": provider.provider_type,
                    "providerLabel": self._provider_tag(provider),
                    "baseUrl": provider.base_url,
                    "apiKey": provider.api_key,
                    "modelPath": provider.model_path,
                    "apiService": provider.api_service,
                    "selected": True,
                    "personaName": persona_name,
                    "hasVision": False,
                    "hasWebSearch": False,
                    "latencyMs": 0,
                    "capabilitySummary": self._provider_tag(provider),
                }
                self._refresh_model_search_text(row)
                self._models.append(row)
                added += 1
            self.modelsChanged.emit()
            await self._detect_model_capabilities()
            self._persist_provider_settings()
            self._set_status(f"Loaded {added} new models. Total available: {len(self._models)}.")
        except Exception as exc:
            self._set_status(f"Model load failed: {exc}")
        finally:
            self._set_busy(False)

    @Slot()
    def save_provider_profile(self) -> None:
        payload = self._current_profile_payload()
        for existing in self._provider_profiles:
            same = (
                existing.get("provider_type") == payload["provider_type"]
                and existing.get("api_service") == payload["api_service"]
                and existing.get("base_url") == payload["base_url"]
                and existing.get("model_path") == payload["model_path"]
                and existing.get("api_key") == payload["api_key"]
            )
            if same:
                self._set_status("Provider already saved.")
                return
        self._provider_profiles.append(payload)
        self._persist_runtime_settings()
        self.providerProfilesChanged.emit()
        self._set_status("Provider profile saved.")

    @Slot(str)
    def use_provider_profile(self, profile_id: str) -> None:
        for profile in self._provider_profiles:
            if profile.get("id") != profile_id:
                continue
            self._provider_type = normalize_provider_type(profile.get("provider_type", PROVIDER_LM_STUDIO))
            self._api_service = normalize_api_service(profile.get("api_service", API_SERVICE_CUSTOM))
            self._base_url = canonicalize_base_url(self._provider_type, profile.get("base_url", "")) or provider_defaults(self._provider_type)[0]
            self._api_key = str(profile.get("api_key", ""))
            self._model_path = str(profile.get("model_path", "") or provider_defaults(self._provider_type)[1])
            self._persist_provider_settings()
            self.stateChanged.emit()
            self._set_status("Provider profile loaded.")
            return

    @Slot(str)
    def remove_provider_profile(self, profile_id: str) -> None:
        self._provider_profiles = [profile for profile in self._provider_profiles if profile.get("id") != profile_id]
        if not self._provider_profiles:
            self._provider_profiles = [self._current_profile_payload()]
        self._persist_runtime_settings()
        self.providerProfilesChanged.emit()
        self._set_status("Provider profile removed.")

    @Slot(str, str)
    def create_persona(self, name: str, prompt: str) -> None:
        clean_name = name.strip()
        clean_prompt = prompt.strip() or None
        if not clean_name:
            self._set_status("Enter a persona name.")
            return
        if clean_name in set(persona_names(self._personas)):
            self._set_status("Persona name already exists.")
            return
        try:
            self._personas.append(add_user_persona(clean_name, clean_prompt))
        except Exception as exc:
            self._set_status(f"Failed to create persona: {exc}")
            return
        sort_personas_inplace(self._personas)
        self._sync_model_persona_names()
        self._persist_runtime_settings()
        self.personasChanged.emit()
        self.modelsChanged.emit()
        self.stateChanged.emit()
        self._set_status(f"Created persona '{clean_name}'.")

    @Slot(str, str, str)
    def update_persona(self, existing_name: str, new_name: str, prompt: str) -> None:
        clean_existing = existing_name.strip()
        clean_name = new_name.strip()
        clean_prompt = prompt.strip() or None
        persona = self._persona_entry(clean_existing)
        if not persona:
            self._set_status("Persona not found.")
            return
        if persona.get("builtin", False):
            self._set_status("Built-in personas cannot be edited.")
            return
        if not clean_name:
            self._set_status("Enter a persona name.")
            return
        if clean_name != clean_existing and clean_name in set(persona_names(self._personas)):
            self._set_status("Persona name already exists.")
            return
        try:
            updated = update_user_persona(clean_existing, clean_name, clean_prompt)
        except Exception as exc:
            self._set_status(f"Failed to update persona: {exc}")
            return
        if not updated:
            self._set_status("Persona update did not complete.")
            return
        persona["name"] = clean_name
        persona["prompt"] = clean_prompt
        self._persona_assignments = rename_persona_assignments(self._persona_assignments, clean_existing, clean_name)
        sort_personas_inplace(self._personas)
        self._sync_model_persona_names()
        self._persist_runtime_settings()
        self.personasChanged.emit()
        self.modelsChanged.emit()
        self.stateChanged.emit()
        self._set_status(f"Updated persona '{clean_name}'.")

    @Slot(str)
    def delete_persona(self, name: str) -> None:
        clean_name = name.strip()
        persona = self._persona_entry(clean_name)
        if not persona:
            self._set_status("Persona not found.")
            return
        if persona.get("builtin", False) or clean_name == "None":
            self._set_status("Built-in personas cannot be deleted.")
            return
        try:
            removed = delete_user_persona(clean_name)
        except Exception as exc:
            self._set_status(f"Failed to delete persona: {exc}")
            return
        if not removed:
            self._set_status("Persona delete did not complete.")
            return
        self._personas = [entry for entry in self._personas if str(entry.get("name", "")) != clean_name]
        self._persona_assignments = clear_persona_assignment(self._persona_assignments, clean_name)
        self._sync_model_persona_names()
        self._persist_runtime_settings()
        self.personasChanged.emit()
        self.modelsChanged.emit()
        self.stateChanged.emit()
        self._set_status(f"Deleted persona '{clean_name}'.")

    @Slot(str)
    def add_attachment(self, file_value: str) -> None:
        path = self._normalize_file_path(file_value)
        if not path:
            self._set_status("Attachment not found.")
            return
        path_str = str(path)
        if any(item["path"] == path_str for item in self._attachments):
            self._set_status("Attachment already added.")
            return
        self._attachments.append({"path": path_str, "name": path.name})
        self.attachmentsChanged.emit()
        self.stateChanged.emit()
        self._set_status(f"Attached {path.name}.")

    @Slot(int)
    def remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self._attachments):
            removed = self._attachments.pop(index)
            self.attachmentsChanged.emit()
            self.stateChanged.emit()
            self._set_status(f"Removed {removed['name']}.")

    @Slot()
    def clear_attachments(self) -> None:
        if not self._attachments:
            return
        self._attachments = []
        self.attachmentsChanged.emit()
        self.stateChanged.emit()
        self._set_status("Attachments cleared.")

    @asyncSlot(str)
    async def run_council(self, prompt: str) -> None:
        question = prompt.strip()
        if self._is_busy:
            self._set_status("Wait for the current operation to finish.")
            return
        if not question:
            self._set_status("Enter a prompt before running the council.")
            return
        if not self._selected_models():
            self._set_status("Select at least one model.")
            return

        self._last_question = question
        self._clear_feed()
        self._append_feed_entry("user", "User Prompt", question)
        if self._attachments:
            self._append_feed_entry(
                "system",
                "Attachments",
                ", ".join(item["name"] for item in self._attachments),
            )
        self._set_busy(True)
        context_block, image_urls = self._collect_context_and_images()
        try:
            if self._mode == "discussion":
                await self._run_discussion(question, context_block, image_urls)
            else:
                await self._run_deliberation(question, context_block, image_urls)
        except Exception as exc:
            self._set_status(f"Run failed: {exc}")
            self._append_feed_entry("system", "Error", str(exc))
        finally:
            self._set_busy(False)

    @Slot()
    def export_json(self) -> None:
        if not self._last_session_record:
            self._set_status("No session data available to export.")
            return
        export_dir = app_data_dir() / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode = str(self._last_session_record.get("mode", "session"))
        path = export_dir / f"{timestamp}-{mode}.json"
        path.write_text(json.dumps(self._last_session_record, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_export_path = str(path)
        self.resultsReady.emit()
        self._set_status(f"Exported session to {path.name}.")

    @Slot(str)
    def replay_session(self, session_path: str) -> None:
        path = Path(session_path)
        if not path.exists():
            self._set_status("Session file not found.")
            return
        try:
            record = load_session(path)
        except Exception as exc:
            self._set_status(f"Failed to load session: {exc}")
            return
        if not record:
            self._set_status("Session file is empty.")
            return
        self._apply_session_record(record, append_feed=True)
        self._set_status(f"Loaded session {path.name}.")
