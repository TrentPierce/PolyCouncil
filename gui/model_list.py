from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from PySide6 import QtCore, QtWidgets


@dataclass
class ModelListWidgets:
    checks: Dict[str, QtWidgets.QCheckBox]
    persona_buttons: Dict[str, QtWidgets.QPushButton]
    meta_labels: Dict[str, QtWidgets.QLabel]
    rows: Dict[str, QtWidgets.QWidget]


def clear_layout(layout: QtWidgets.QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
        del item


def make_unique_display_model_name(raw_model: str, provider_tag: str, existing_names: Iterable[str]) -> str:
    base_name = f"{provider_tag} :: {raw_model}"
    existing = set(existing_names)
    if base_name not in existing:
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name} ({suffix})"
        if candidate not in existing:
            return candidate
        suffix += 1


def build_model_badge_text(
    provider_badge: str,
    *,
    capabilities: Optional[Dict[str, bool]] = None,
    latency_ms: Optional[float] = None,
) -> str:
    cap_parts = []
    caps = capabilities or {}
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


def persona_button_state(persona_name: str) -> tuple[str, str]:
    display_text = persona_name if persona_name != "None" else "Persona"
    if len(display_text) > 12:
        display_text = display_text[:10] + ".."
    tooltip = persona_name if persona_name != "None" else "Select persona"
    return display_text, tooltip


def configure_persona_button(button: QtWidgets.QPushButton, persona_name: str, *, enabled: bool) -> None:
    display_text, tooltip = persona_button_state(persona_name)
    button.setText(display_text)
    button.setToolTip(tooltip)
    button.setVisible(enabled)
    button.setEnabled(enabled)
    button.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    button.setFocusPolicy(QtCore.Qt.StrongFocus)
    button.setAutoDefault(False)
    button.setDefault(False)
    button.raise_()


def build_model_row(
    *,
    model_id: str,
    provider_name: str,
    meta_text: str,
    assigned_persona: str,
    personas_enabled: bool,
    ui_available: bool,
    model_card_class,
) -> tuple[QtWidgets.QWidget, QtWidgets.QCheckBox, QtWidgets.QPushButton, QtWidgets.QLabel]:
    if ui_available and model_card_class is not None:
        row_widget = model_card_class(model_id, provider_name, meta_text)
        checkbox = row_widget.checkbox
        persona_button = row_widget.persona_btn
        meta_label = row_widget.meta_label
        row_widget.setPersonaText(assigned_persona)
        row_widget.showPersonaButton(personas_enabled)
    else:
        row_widget = QtWidgets.QWidget()
        row_widget.setMinimumHeight(34)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)

        checkbox = QtWidgets.QCheckBox(model_id)
        checkbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        checkbox.setAccessibleName(f"Model selector {model_id}")

        persona_button = QtWidgets.QPushButton("Persona")
        persona_button.setFixedWidth(118)
        persona_button.setFixedHeight(28)
        persona_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        meta_label = QtWidgets.QLabel(meta_text)
        meta_label.setObjectName("HintLabel")
        meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        meta_label.setMinimumWidth(170)

        row_layout.addWidget(checkbox, stretch=1)
        row_layout.addWidget(meta_label, stretch=0)
        row_layout.addWidget(persona_button, stretch=0)
        row_layout.setAlignment(persona_button, QtCore.Qt.AlignRight)

    configure_persona_button(persona_button, assigned_persona, enabled=personas_enabled)
    return row_widget, checkbox, persona_button, meta_label


def populate_model_rows(
    *,
    layout: QtWidgets.QVBoxLayout,
    models: Iterable[str],
    persona_assignments: Dict[str, str],
    persona_names: Iterable[str],
    personas_enabled: bool,
    provider_name_for_model: Callable[[str], str],
    meta_text_for_model: Callable[[str], str],
    on_persona_click: Callable[[str, QtWidgets.QPushButton], None],
    on_selection_changed: Callable[[bool], None],
    ui_available: bool,
    model_card_class,
) -> ModelListWidgets:
    clear_layout(layout)
    layout.addStretch(1)

    valid_personas = set(persona_names)
    checks: Dict[str, QtWidgets.QCheckBox] = {}
    persona_buttons: Dict[str, QtWidgets.QPushButton] = {}
    meta_labels: Dict[str, QtWidgets.QLabel] = {}
    rows: Dict[str, QtWidgets.QWidget] = {}

    for model in models:
        assigned = persona_assignments.get(model, "None")
        if assigned not in valid_personas:
            assigned = "None"
        row_widget, checkbox, persona_button, meta_label = build_model_row(
            model_id=model,
            provider_name=provider_name_for_model(model),
            meta_text=meta_text_for_model(model),
            assigned_persona=assigned,
            personas_enabled=personas_enabled,
            ui_available=ui_available,
            model_card_class=model_card_class,
        )
        persona_button.clicked.connect(
            lambda _checked=False, mid=model, btn=persona_button: on_persona_click(mid, btn)
        )
        checkbox.toggled.connect(on_selection_changed)
        layout.insertWidget(layout.count() - 1, row_widget)
        checks[model] = checkbox
        persona_buttons[model] = persona_button
        meta_labels[model] = meta_label
        rows[model] = row_widget

    return ModelListWidgets(
        checks=checks,
        persona_buttons=persona_buttons,
        meta_labels=meta_labels,
        rows=rows,
    )


def apply_persona_visibility(
    persona_buttons: Dict[str, QtWidgets.QPushButton],
    *,
    enabled: bool,
) -> None:
    for button in persona_buttons.values():
        if button is None:
            continue
        button.setVisible(enabled)
        button.setEnabled(enabled)
        if enabled:
            button.setStyleSheet("")
        else:
            button.setStyleSheet("QPushButton { color: #888; background-color: #333; }")


def filter_model_rows(
    model_rows: Dict[str, QtWidgets.QWidget],
    model_meta_labels: Dict[str, QtWidgets.QLabel],
    query: str,
) -> None:
    normalized = query.strip().lower()
    for model, row in model_rows.items():
        badge = model_meta_labels.get(model)
        haystack = f"{model} {badge.text() if badge else ''}".lower()
        row.setVisible((not normalized) or (normalized in haystack))
