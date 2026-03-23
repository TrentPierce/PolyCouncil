from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore, QtWidgets

from constants import INTERNAL_GAP, PADDING_LG, PADDING_MD, SECTION_GAP, STRINGS, dp
from ui.factories import (
    apply_form_grid,
    make_button,
    make_entry,
    make_hint_label,
    make_label,
)


@dataclass
class WorkspacePanelWidgets:
    dock: QtWidgets.QDockWidget
    tabs: QtWidgets.QTabWidget
    settings_debug_check: QtWidgets.QCheckBox
    settings_personas_check: QtWidgets.QCheckBox
    settings_storage_label: QtWidgets.QLabel
    settings_timeout_spin: QtWidgets.QSpinBox
    settings_reduce_motion_check: QtWidgets.QCheckBox
    rubric_weight_spins: dict[str, QtWidgets.QSpinBox]
    settings_shortcuts_btn: QtWidgets.QPushButton
    settings_issue_btn: QtWidgets.QPushButton
    persona_search_edit: QtWidgets.QLineEdit
    persona_library_list: QtWidgets.QListWidget
    persona_preview: QtWidgets.QTextBrowser
    persona_add_btn: QtWidgets.QPushButton
    persona_edit_btn: QtWidgets.QPushButton
    persona_delete_btn: QtWidgets.QPushButton


def build_workspace_panel(
    parent: QtWidgets.QMainWindow,
    *,
    create_output_view: Callable[[], QtWidgets.QTextBrowser],
) -> WorkspacePanelWidgets:
    dock = QtWidgets.QDockWidget("Workspace Panel", parent)
    dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable
    )

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
    layout.setSpacing(dp(INTERNAL_GAP))

    tabs = QtWidgets.QTabWidget()
    tabs.setDocumentMode(True)

    settings_page = QtWidgets.QWidget()
    settings_layout = QtWidgets.QVBoxLayout(settings_page)
    settings_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
    settings_layout.setSpacing(dp(SECTION_GAP))
    settings_intro = make_hint_label(STRINGS["settings_intro"])
    settings_layout.addWidget(settings_intro)

    settings_debug_check = QtWidgets.QCheckBox("Enable debug logs")
    settings_personas_check = QtWidgets.QCheckBox("Show persona controls in the workflow")
    settings_storage_label = make_hint_label("")
    timeout_row = QtWidgets.QWidget()
    timeout_layout = QtWidgets.QHBoxLayout(timeout_row)
    timeout_layout.setContentsMargins(0, 0, 0, 0)
    timeout_layout.setSpacing(dp(PADDING_MD))
    timeout_label = make_label("Request timeout", object_name="FieldLabel")
    settings_timeout_spin = QtWidgets.QSpinBox()
    settings_timeout_spin.setRange(15, 600)
    settings_timeout_spin.setSuffix(" s")
    timeout_layout.addWidget(timeout_label)
    timeout_layout.addStretch(1)
    timeout_layout.addWidget(settings_timeout_spin)
    settings_reduce_motion_check = QtWidgets.QCheckBox("Reduce motion and skip non-essential animations")
    rubric_group = QtWidgets.QGroupBox("Scoring Rubric")
    rubric_layout = QtWidgets.QGridLayout(rubric_group)
    rubric_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
    apply_form_grid(rubric_layout)
    rubric_weight_spins: dict[str, QtWidgets.QSpinBox] = {}
    for row, (label_text, key) in enumerate(
        [
            ("Correctness", "correctness"),
            ("Relevance", "relevance"),
            ("Specificity", "specificity"),
            ("Safety", "safety"),
            ("Conciseness", "conciseness"),
        ]
    ):
        rubric_label = make_label(label_text, object_name="FieldLabel")
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 10)
        rubric_layout.addWidget(rubric_label, row, 0)
        rubric_layout.addWidget(spin, row, 1)
        rubric_weight_spins[key] = spin
    settings_shortcuts_btn = make_button(STRINGS["action_shortcuts"], variant="secondary", action_id="toggle_shortcuts")
    settings_issue_btn = make_button(STRINGS["action_report_issue"], variant="secondary")

    settings_layout.addWidget(settings_debug_check)
    settings_layout.addWidget(settings_personas_check)
    settings_layout.addWidget(settings_storage_label)
    settings_layout.addWidget(timeout_row)
    settings_layout.addWidget(settings_reduce_motion_check)
    settings_layout.addWidget(rubric_group)
    settings_layout.addWidget(settings_shortcuts_btn)
    settings_layout.addWidget(settings_issue_btn)
    settings_layout.addStretch(1)

    personas_page = QtWidgets.QWidget()
    personas_layout = QtWidgets.QVBoxLayout(personas_page)
    personas_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG), dp(PADDING_LG))
    personas_layout.setSpacing(dp(INTERNAL_GAP))
    personas_intro = make_hint_label(STRINGS["personas_intro"])

    persona_search_edit = make_entry("Filter persona library")
    persona_library_list = QtWidgets.QListWidget()
    persona_preview = create_output_view()

    persona_actions = QtWidgets.QHBoxLayout()
    persona_actions.setSpacing(dp(PADDING_MD))
    persona_add_btn = make_button("Add Persona", variant="secondary")
    persona_edit_btn = make_button("Edit Persona", variant="secondary")
    persona_delete_btn = make_button("Delete Persona", variant="danger")
    persona_actions.addWidget(persona_add_btn)
    persona_actions.addWidget(persona_edit_btn)
    persona_actions.addWidget(persona_delete_btn)
    persona_actions.addStretch(1)

    personas_layout.addWidget(personas_intro)
    personas_layout.addWidget(persona_search_edit)
    personas_layout.addWidget(persona_library_list, 1)
    personas_layout.addLayout(persona_actions)
    personas_layout.addWidget(persona_preview, 1)

    tabs.addTab(settings_page, "Settings")
    tabs.addTab(personas_page, "Personas")
    layout.addWidget(tabs)

    dock.setWidget(container)

    return WorkspacePanelWidgets(
        dock=dock,
        tabs=tabs,
        settings_debug_check=settings_debug_check,
        settings_personas_check=settings_personas_check,
        settings_storage_label=settings_storage_label,
        settings_timeout_spin=settings_timeout_spin,
        settings_reduce_motion_check=settings_reduce_motion_check,
        rubric_weight_spins=rubric_weight_spins,
        settings_shortcuts_btn=settings_shortcuts_btn,
        settings_issue_btn=settings_issue_btn,
        persona_search_edit=persona_search_edit,
        persona_library_list=persona_library_list,
        persona_preview=persona_preview,
        persona_add_btn=persona_add_btn,
        persona_edit_btn=persona_edit_btn,
        persona_delete_btn=persona_delete_btn,
    )
