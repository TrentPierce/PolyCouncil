from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore, QtWidgets


@dataclass
class WorkspacePanelWidgets:
    dock: QtWidgets.QDockWidget
    tabs: QtWidgets.QTabWidget
    settings_debug_check: QtWidgets.QCheckBox
    settings_personas_check: QtWidgets.QCheckBox
    settings_storage_label: QtWidgets.QLabel
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
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    tabs = QtWidgets.QTabWidget()
    tabs.setDocumentMode(True)

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

    settings_debug_check = QtWidgets.QCheckBox("Enable debug logs")
    settings_personas_check = QtWidgets.QCheckBox("Show persona controls in the workflow")
    settings_storage_label = QtWidgets.QLabel("")
    settings_storage_label.setObjectName("HintLabel")
    settings_storage_label.setWordWrap(True)
    settings_shortcuts_btn = QtWidgets.QPushButton("Keyboard Shortcuts")
    settings_shortcuts_btn.setObjectName("SecondaryButton")
    settings_issue_btn = QtWidgets.QPushButton("Report an Issue")
    settings_issue_btn.setObjectName("SecondaryButton")

    settings_layout.addWidget(settings_debug_check)
    settings_layout.addWidget(settings_personas_check)
    settings_layout.addWidget(settings_storage_label)
    settings_layout.addWidget(settings_shortcuts_btn)
    settings_layout.addWidget(settings_issue_btn)
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

    persona_search_edit = QtWidgets.QLineEdit()
    persona_search_edit.setPlaceholderText("Filter persona library")
    persona_library_list = QtWidgets.QListWidget()
    persona_preview = create_output_view()

    persona_actions = QtWidgets.QHBoxLayout()
    persona_add_btn = QtWidgets.QPushButton("Add Persona")
    persona_edit_btn = QtWidgets.QPushButton("Edit Persona")
    persona_delete_btn = QtWidgets.QPushButton("Delete Persona")
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
        settings_shortcuts_btn=settings_shortcuts_btn,
        settings_issue_btn=settings_issue_btn,
        persona_search_edit=persona_search_edit,
        persona_library_list=persona_library_list,
        persona_preview=persona_preview,
        persona_add_btn=persona_add_btn,
        persona_edit_btn=persona_edit_btn,
        persona_delete_btn=persona_delete_btn,
    )
