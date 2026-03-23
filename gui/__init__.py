"""GUI composition helpers for PolyCouncil."""

from gui.model_list import (
    ModelListWidgets,
    apply_persona_visibility,
    build_model_badge_text,
    clear_layout as clear_model_layout,
    filter_model_rows,
    make_unique_display_model_name,
    populate_model_rows,
)
from gui.debug_timeline import DebugTimelineWidget
from gui.persona_panel import build_persona_preview_html, populate_persona_library_list
from gui.provider_profiles import build_provider_profile_row, clear_layout
from gui.workspace_panel import WorkspacePanelWidgets, build_workspace_panel

__all__ = [
    "ModelListWidgets",
    "WorkspacePanelWidgets",
    "apply_persona_visibility",
    "build_model_badge_text",
    "build_persona_preview_html",
    "DebugTimelineWidget",
    "build_workspace_panel",
    "build_provider_profile_row",
    "clear_model_layout",
    "clear_layout",
    "filter_model_rows",
    "make_unique_display_model_name",
    "populate_model_rows",
    "populate_persona_library_list",
]
