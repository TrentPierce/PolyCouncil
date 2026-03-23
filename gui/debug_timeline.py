from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6 import QtWidgets


class DebugTimelineWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Stage", "Details"])
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(False)
        self._items: Dict[str, QtWidgets.QTreeWidgetItem] = {}
        self.reset_timeline()

    def reset_timeline(self):
        self.clear()
        self._items = {}
        for key, title in (
            ("setup", "Setup"),
            ("answers", "Parallel execution"),
            ("voting", "Cross-scoring and ballots"),
            ("consensus", "Consensus"),
            ("discussion", "Discussion"),
            ("errors", "Errors"),
        ):
            item = QtWidgets.QTreeWidgetItem([title, "Waiting"])
            self.addTopLevelItem(item)
            self._items[key] = item
        self.expandAll()

    def add_status(self, text: str):
        lowered = text.lower()
        if "collecting answers" in lowered:
            self._items["answers"].setText(1, text)
        elif "voting" in lowered or "ballot" in lowered:
            self._items["voting"].setText(1, text)
        elif "discussion" in lowered:
            self._items["discussion"].setText(1, text)
        elif any(token in lowered for token in ("done", "complete", "winner", "valid")):
            self._items["consensus"].setText(1, text)
        elif any(token in lowered for token in ("error", "failed", "timed out", "warning", "invalid")):
            self.add_error(text)
        else:
            self._items["setup"].setText(1, text)

    def add_error(self, text: str):
        self._items["errors"].setText(1, "Attention required")
        self._items["errors"].addChild(QtWidgets.QTreeWidgetItem(["Issue", text]))
        self.expandItem(self._items["errors"])

    def show_deliberation_result(self, *, details: Dict[str, Any], tally: Dict[str, Any], short_id):
        self._items["answers"].takeChildren()
        timings = details.get("timings_ms", {}) or {}
        errors = details.get("errors", {}) or {}
        for model_id in sorted(timings.keys(), key=lambda key: (timings[key], key)):
            info = f"{int(timings[model_id])} ms"
            if model_id in errors:
                info = f"{info} · {errors[model_id]}"
            self._items["answers"].addChild(QtWidgets.QTreeWidgetItem([short_id(model_id), info]))
        self._items["answers"].setText(1, f"{len(timings)} model(s) responded")

        self._items["voting"].takeChildren()
        valid_votes = details.get("valid_votes", {}) or {}
        rubric_weights = details.get("rubric_weights", {}) or {}
        for voter, ballot in valid_votes.items():
            voter_item = QtWidgets.QTreeWidgetItem([short_id(voter), ballot.get("reasoning", "Valid ballot")])
            for candidate, score_map in (ballot.get("scores", {}) or {}).items():
                weighted = sum(int(score_map.get(key, 0)) * int(rubric_weights.get(key, 0)) for key in rubric_weights.keys())
                criteria = ", ".join(
                    f"{key} {score_map.get(key, 0)}×{rubric_weights.get(key, 0)}"
                    for key in rubric_weights.keys()
                )
                voter_item.addChild(QtWidgets.QTreeWidgetItem([short_id(candidate), f"{weighted} · {criteria}"]))
            self._items["voting"].addChild(voter_item)
        invalid_votes = details.get("invalid_votes", {}) or {}
        for voter, message in invalid_votes.items():
            self._items["voting"].addChild(QtWidgets.QTreeWidgetItem([short_id(voter), f"Invalid · {message}"]))
        self._items["voting"].setText(
            1,
            f"{len(valid_votes)} valid / {len(valid_votes) + len(invalid_votes)} total ballots",
        )

        self._items["consensus"].takeChildren()
        for model_id, score in sorted(tally.items(), key=lambda item: (-item[1], item[0])):
            self._items["consensus"].addChild(QtWidgets.QTreeWidgetItem([short_id(model_id), f"Score {score}"]))
        winner = details.get("winner", "")
        self._items["consensus"].setText(1, f"Winner: {short_id(winner) if winner else 'Unavailable'}")
        self.expandAll()

    def show_discussion_result(self, *, transcript: List[Dict[str, Any]], synthesis: Optional[str]):
        self._items["discussion"].takeChildren()
        for entry in transcript:
            summary = (entry.get("message", "") or "").strip().replace("\n", " ")
            if len(summary) > 120:
                summary = summary[:117] + "..."
            self._items["discussion"].addChild(
                QtWidgets.QTreeWidgetItem(
                    [f"Turn {entry.get('turn', '?')} · {entry.get('agent', 'Unknown')}", summary]
                )
            )
        self._items["discussion"].setText(1, f"{len(transcript)} turn(s) recorded")
        self._items["consensus"].setText(1, "Synthesis ready" if synthesis else "Synthesis unavailable")
        self.expandAll()
