from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets


PADDING_SM = 4
PADDING_MD = 8
PADDING_LG = 16
PADDING_XL = 24
INTERNAL_GAP = 12
SECTION_GAP = 24
ROW_PADDING = 8
TOOLBAR_ICON_GAP = 6
TOOLBAR_BUTTON_GAP = 2

MIN_W = 640
MIN_H = 480
MIN_SIDEBAR = 180
MIN_TOUCH_TARGET = 44
FOCUS_RING_WIDTH = 2
CORNER_RADIUS_SM = 4
CORNER_RADIUS_MD = 8
CORNER_RADIUS_LG = 16
TOOLTIP_MAX_WIDTH = 260
TOOLTIP_SHOW_DELAY_MS = 600
TOOLTIP_HIDE_DELAY_MS = 200

FONT_XS = 10
FONT_SM = 12
FONT_BASE = 13
FONT_MD = 14
FONT_LG = 16
FONT_XL = 20
FONT_DISPLAY = 24
LINE_HEIGHT_MULTIPLIER = 1.4
MONOSPACE_FONT_FAMILY = "Consolas"

WINDOW_ICON_SIZE = 32
TOOLBAR_ICON_SIZE = 20
MENU_ICON_SIZE = 16
DIALOG_ICON_SIZE = 32


LIGHT_THEME: dict[str, str] = {
    "bg_primary": "#f4f7fb",
    "bg_secondary": "#ffffff",
    "bg_tertiary": "#e8eef5",
    "fg_primary": "#17212b",
    "fg_secondary": "#425466",
    "fg_muted": "#66778a",
    "accent": "#0f62fe",
    "accent_hover": "#0a53d6",
    "accent_fg": "#ffffff",
    "border": "#c9d5e3",
    "border_strong": "#8ea4bb",
    "danger": "#c62828",
    "danger_fg": "#ffffff",
    "warning": "#b26a00",
    "warning_fg": "#1f1300",
    "success": "#1d7a3f",
    "success_fg": "#ffffff",
}

DARK_THEME: dict[str, str] = {
    "bg_primary": "#0f1720",
    "bg_secondary": "#17212b",
    "bg_tertiary": "#22303d",
    "fg_primary": "#f2f6fb",
    "fg_secondary": "#c1cfdd",
    "fg_muted": "#8ea2b8",
    "accent": "#58a6ff",
    "accent_hover": "#7ab8ff",
    "accent_fg": "#08131d",
    "border": "#314255",
    "border_strong": "#51657a",
    "danger": "#ff6b6b",
    "danger_fg": "#1f0d0d",
    "warning": "#f4b860",
    "warning_fg": "#261800",
    "success": "#5dd39e",
    "success_fg": "#062116",
}

THEME: dict[str, str] = dict(DARK_THEME)


STRINGS: dict[str, str] = {
    "app_title": "PolyCouncil",
    "welcome_title": "Welcome",
    "welcome_message": "Choose a provider, load some models, and send a prompt to compare answers or run a collaborative discussion.",
    "settings_intro": "Settings are applied immediately. Use this panel for stable app preferences and support links.",
    "personas_intro": "Manage the persona library here. Assign personas from the model list in the workflow.",
    "empty_models": "Load models to unlock selection.",
    "empty_history": "No items found. Run a council round to start building history.",
    "empty_attachments": "Attach supporting files here before you run.",
    "status_ready": "Ready. Choose a provider, then load models.",
    "status_no_keychain": "Secure keychain unavailable. Hosted API keys will only persist for the current session.",
    "action_report_issue": "Report an Issue",
    "action_shortcuts": "Keyboard Shortcuts",
    "tooltip_prompt": "Enter a council prompt. Press Enter to run and Shift plus Enter for a new line.",
}

ACTIONS: dict[str, dict[str, str]] = {
    "load_models": {"label": "Load models", "shortcut": "Ctrl+R", "tooltip": "Load models from the current provider (Ctrl+R)."},
    "focus_prompt": {"label": "Focus prompt", "shortcut": "Ctrl+L", "tooltip": "Focus the prompt editor (Ctrl+L)."},
    "focus_filter": {"label": "Focus model filter", "shortcut": "Ctrl+F", "tooltip": "Focus the model filter (Ctrl+F)."},
    "select_all_models": {"label": "Select all models", "shortcut": "Ctrl+Shift+A", "tooltip": "Select every loaded model (Ctrl+Shift+A)."},
    "toggle_shortcuts": {"label": "Toggle shortcuts help", "shortcut": "Ctrl+?", "tooltip": "Open keyboard shortcuts help (Ctrl+?)."},
    "run_council": {"label": "Run council", "shortcut": "Ctrl+Enter", "tooltip": "Run the current council request (Ctrl+Enter)."},
    "stop_run": {"label": "Stop run", "shortcut": "Escape", "tooltip": "Stop the active operation (Escape)."},
    "toggle_theme": {"label": "Toggle theme", "shortcut": "Ctrl+Shift+T", "tooltip": "Toggle between light and dark theme (Ctrl+Shift+T)."},
}


def compute_dpi_scale(app: QtWidgets.QApplication | None = None) -> float:
    current_app = app or QtWidgets.QApplication.instance()
    if current_app is None:
        return 1.0
    screen = current_app.primaryScreen()
    if screen is None:
        return 1.0
    dpi = max(96.0, float(screen.logicalDotsPerInch()))
    return round(dpi / 96.0, 2)


DPI_SCALE = compute_dpi_scale()


def refresh_dpi_scale(app: QtWidgets.QApplication | None = None) -> float:
    global DPI_SCALE
    DPI_SCALE = compute_dpi_scale(app)
    return DPI_SCALE


def dp(value: int | float) -> int:
    return max(1, int(round(float(value) * DPI_SCALE)))


def line_height_for(font_size: int) -> int:
    return int(round(font_size * LINE_HEIGHT_MULTIPLIER))


def make_font(
    point_size: int,
    *,
    bold: bool = False,
    italic: bool = False,
    family: str | None = None,
) -> QtGui.QFont:
    font = QtGui.QFont(family or "")
    font.setPointSizeF(float(point_size))
    font.setBold(bold)
    font.setItalic(italic)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    return font


def make_monospace_font(point_size: int = FONT_SM, *, bold: bool = False) -> QtGui.QFont:
    return make_font(point_size, bold=bold, family=MONOSPACE_FONT_FAMILY)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    qcolor = QtGui.QColor(color)
    return (qcolor.redF(), qcolor.greenF(), qcolor.blueF())


def relative_luminance(color: str) -> float:
    def normalize(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = hex_to_rgb(color)
    return 0.2126 * normalize(r) + 0.7152 * normalize(g) + 0.0722 * normalize(b)


def contrast_ratio(foreground: str, background: str) -> float:
    fg = relative_luminance(foreground)
    bg = relative_luminance(background)
    lighter = max(fg, bg)
    darker = min(fg, bg)
    return round((lighter + 0.05) / (darker + 0.05), 2)


@dataclass(frozen=True)
class ContrastResult:
    name: str
    foreground: str
    background: str
    ratio: float
    minimum: float

    @property
    def passes(self) -> bool:
        return self.ratio >= self.minimum


def audit_theme_contrast(theme: dict[str, str]) -> list[ContrastResult]:
    pairs = [
        ("body", theme["fg_primary"], theme["bg_secondary"], 4.5),
        ("secondary", theme["fg_secondary"], theme["bg_secondary"], 4.5),
        ("muted", theme["fg_muted"], theme["bg_secondary"], 4.5),
        ("accent_button", theme["accent_fg"], theme["accent"], 4.5),
        ("danger_button", theme["danger_fg"], theme["danger"], 4.5),
        ("warning_text", theme["warning_fg"], theme["warning"], 3.0),
        ("success_text", theme["success_fg"], theme["success"], 3.0),
    ]
    return [
        ContrastResult(
            name=name,
            foreground=foreground,
            background=background,
            ratio=contrast_ratio(foreground, background),
            minimum=minimum,
        )
        for name, foreground, background, minimum in pairs
    ]


CONTRAST_AUDIT = {
    "light": audit_theme_contrast(LIGHT_THEME),
    "dark": audit_theme_contrast(DARK_THEME),
}


def tooltip_style(theme: dict[str, str]) -> str:
    return (
        "QToolTip {"
        f"background-color: {theme['bg_secondary']};"
        f"color: {theme['fg_primary']};"
        f"border: 1px solid {theme['border_strong']};"
        f"padding: {dp(PADDING_MD)}px;"
        f"max-width: {dp(TOOLTIP_MAX_WIDTH)}px;"
        f"border-radius: {dp(CORNER_RADIUS_SM)}px;"
        "}"
    )
