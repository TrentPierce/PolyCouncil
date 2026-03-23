from __future__ import annotations

from typing import Callable, Iterable, Optional

from PySide6 import QtCore, QtGui, QtWidgets


def populate_persona_library_list(
    list_widget: QtWidgets.QListWidget,
    personas: Iterable[dict],
    *,
    query: str,
    current_name: str,
) -> None:
    list_widget.clear()
    normalized = query.strip().lower()
    for persona in personas:
        name = persona["name"]
        prompt = persona.get("prompt") or ""
        if normalized and normalized not in name.lower() and normalized not in prompt.lower():
            continue
        item = QtWidgets.QListWidgetItem(name)
        if persona.get("builtin", False):
            item.setForeground(QtGui.QColor("#666"))
        list_widget.addItem(item)

    if current_name:
        matches = list_widget.findItems(current_name, QtCore.Qt.MatchExactly)
        if matches:
            list_widget.setCurrentItem(matches[0])
    if not list_widget.currentItem() and list_widget.count():
        list_widget.setCurrentRow(0)


def build_persona_preview_html(
    persona: Optional[dict],
    *,
    assignment_count: int,
    colors: dict,
    render_markdown: Callable[[str], str],
    escape_text: Callable[[str], str],
    placeholder_html: Callable[[str, str], str],
) -> str:
    if not persona:
        return placeholder_html("Persona Preview", "Select a persona to inspect its prompt and assignment behavior.")

    prompt = persona.get("prompt") or "No prompt configured."
    persona_type = "Built-in" if persona.get("builtin", False) else "Custom"
    return (
        f"<div style='font-size:18px; font-weight:700; margin-bottom:6px;'>{escape_text(persona['name'])}</div>"
        f"<div style='margin-bottom:10px; color:{colors['text_secondary']};'>"
        f"{persona_type} persona &middot; Assigned to {assignment_count} model(s)</div>"
        f"{render_markdown(prompt)}"
    )
