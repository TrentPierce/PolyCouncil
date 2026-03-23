"""
PolyCouncil Theme Engine
========================
Adaptive dark/light color palette, dynamic QSS generation, and font loading.
All color tokens are defined here so the rest of the UI never needs hardcoded hex values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Color Tokens
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColorTokens:
    """Semantic color tokens for the entire UI. Two instances: dark and light."""

    # Surfaces
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_elevated: str

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    text_inverse: str

    # Brand / accent
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_muted: str

    # Semantic
    success: str
    success_bg: str
    warning: str
    warning_bg: str
    danger: str
    danger_bg: str
    info: str
    info_bg: str

    # Borders & separators
    border: str
    border_subtle: str
    border_focus: str

    # Badge backgrounds
    badge_neutral_bg: str
    badge_neutral_fg: str
    badge_busy_bg: str
    badge_busy_border: str
    badge_success_bg: str
    badge_success_border: str
    badge_warn_bg: str
    badge_warn_border: str
    badge_error_bg: str
    badge_error_border: str

    # Misc
    shadow: str
    overlay: str
    scrollbar_thumb: str
    scrollbar_track: str


DARK_TOKENS = ColorTokens(
    # Surfaces
    bg_primary="#0f1419",
    bg_secondary="#161b22",
    bg_tertiary="#1c2128",
    bg_elevated="#21262d",

    # Text
    text_primary="#e6edf3",
    text_secondary="#b1bac4",
    text_muted="#6e7681",
    text_inverse="#0f1419",

    # Brand
    accent="#4493f8",
    accent_hover="#539bf5",
    accent_pressed="#316dca",
    accent_muted="#1a3a5c",

    # Semantic
    success="#3fb950",
    success_bg="#1b3a2a",
    warning="#d29922",
    warning_bg="#3d2e00",
    danger="#f85149",
    danger_bg="#3d1518",
    info="#58a6ff",
    info_bg="#12263a",

    # Borders
    border="#30363d",
    border_subtle="#21262d",
    border_focus="#4493f8",

    # Badges
    badge_neutral_bg="#21262d",
    badge_neutral_fg="#b1bac4",
    badge_busy_bg="#12263a",
    badge_busy_border="#316dca",
    badge_success_bg="#1b3a2a",
    badge_success_border="#3fb950",
    badge_warn_bg="#3d2e00",
    badge_warn_border="#d29922",
    badge_error_bg="#3d1518",
    badge_error_border="#f85149",

    # Misc
    shadow="rgba(0,0,0,0.4)",
    overlay="rgba(0,0,0,0.6)",
    scrollbar_thumb="#30363d",
    scrollbar_track="#161b22",
)

LIGHT_TOKENS = ColorTokens(
    # Surfaces
    bg_primary="#ffffff",
    bg_secondary="#f6f8fa",
    bg_tertiary="#eaeef2",
    bg_elevated="#ffffff",

    # Text
    text_primary="#1f2328",
    text_secondary="#57606a",
    text_muted="#8b949e",
    text_inverse="#ffffff",

    # Brand
    accent="#0969da",
    accent_hover="#0550ae",
    accent_pressed="#033d8b",
    accent_muted="#ddf4ff",

    # Semantic
    success="#1a7f37",
    success_bg="#dafbe1",
    warning="#9a6700",
    warning_bg="#fff8c5",
    danger="#d1242f",
    danger_bg="#ffebe9",
    info="#0969da",
    info_bg="#ddf4ff",

    # Borders
    border="#d0d7de",
    border_subtle="#eaeef2",
    border_focus="#0969da",

    # Badges
    badge_neutral_bg="#eaeef2",
    badge_neutral_fg="#57606a",
    badge_busy_bg="#ddf4ff",
    badge_busy_border="#54aeff",
    badge_success_bg="#dafbe1",
    badge_success_border="#1a7f37",
    badge_warn_bg="#fff8c5",
    badge_warn_border="#9a6700",
    badge_error_bg="#ffebe9",
    badge_error_border="#d1242f",

    # Misc
    shadow="rgba(31,35,40,0.12)",
    overlay="rgba(31,35,40,0.5)",
    scrollbar_thumb="#afb8c1",
    scrollbar_track="#f6f8fa",
)


# ---------------------------------------------------------------------------
# Theme Engine
# ---------------------------------------------------------------------------
class ThemeEngine(QtCore.QObject):
    """Generates and applies a dynamic QSS stylesheet based on system palette."""

    themeChanged = QtCore.Signal()

    def __init__(self, app: QtWidgets.QApplication, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._app = app
        self._dark = self._detect_dark()
        self._tokens = DARK_TOKENS if self._dark else LIGHT_TOKENS
        self._font_family = self._load_fonts()

    # -- public API --
    @property
    def is_dark(self) -> bool:
        return self._dark

    @property
    def tokens(self) -> ColorTokens:
        return self._tokens

    @property
    def font_family(self) -> str:
        return self._font_family

    def apply(self, window: QtWidgets.QMainWindow):
        """Generate and apply the stylesheet to the window."""
        self._dark = self._detect_dark()
        self._tokens = DARK_TOKENS if self._dark else LIGHT_TOKENS
        window.setStyleSheet(self._generate_qss())

    def refresh(self, window: QtWidgets.QMainWindow):
        """Re-detect theme and reapply."""
        new_dark = self._detect_dark()
        if new_dark != self._dark:
            self._dark = new_dark
            self._tokens = DARK_TOKENS if self._dark else LIGHT_TOKENS
            window.setStyleSheet(self._generate_qss())
            self.themeChanged.emit()

    # -- helpers --
    def _detect_dark(self) -> bool:
        return self._app.palette().color(QtGui.QPalette.Window).lightness() < 128

    def _load_fonts(self) -> str:
        """Try to load Inter font; fall back to system sans-serif."""
        try:
            font_id = QtGui.QFontDatabase.addApplicationFont(":/fonts/Inter-Variable.ttf")
            if font_id >= 0:
                families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    return families[0]
        except Exception:
            pass
        # Fallback chain
        for candidate in ("Inter", "Segoe UI", "Helvetica Neue", "Arial"):
            if QtGui.QFontDatabase.hasFamily(candidate):
                return candidate
        return "Segoe UI"

    def _generate_qss(self) -> str:
        t = self._tokens
        ff = self._font_family
        return f"""
        /* ===== Global ===== */
        * {{
            font-family: "{ff}", "Segoe UI", sans-serif;
        }}

        QWidget#Root {{
            background: {t.bg_primary};
        }}

        /* ===== Hero Card ===== */
        QFrame#HeroCard {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {t.accent_muted}, stop:1 {t.bg_secondary});
            border: 1px solid {t.border};
            border-radius: 16px;
        }}

        /* ===== Composer Card ===== */
        QFrame#ComposerCard {{
            background: {t.bg_secondary};
            border: 1px solid {t.border};
            border-radius: 14px;
        }}

        /* ===== Group Boxes ===== */
        QGroupBox {{
            background: {t.bg_secondary};
            border: 1px solid {t.border};
            border-radius: 14px;
            margin-top: 10px;
            font-weight: 600;
            color: {t.text_primary};
        }}
        QGroupBox::title {{
            left: 14px;
            padding: 0 6px;
            color: {t.text_secondary};
        }}

        /* ===== Collapsible Group Boxes ===== */
        QFrame#CollapsibleHeader {{
            background: transparent;
            border: none;
            border-bottom: 1px solid {t.border_subtle};
            padding: 8px 14px;
        }}
        QFrame#CollapsibleHeader:hover {{
            background: {t.bg_tertiary};
        }}

        /* ===== Labels ===== */
        QLabel {{
            color: {t.text_primary};
        }}
        QLabel#HeroTitle {{
            color: {t.text_primary};
            letter-spacing: 0.5px;
            font-size: 20px;
            font-weight: 800;
        }}
        QLabel#HeroSubtitle {{
            color: {t.text_secondary};
            font-size: 13px;
        }}
        QLabel#HintLabel {{
            color: {t.text_muted};
            font-size: 12px;
        }}
        QLabel#StatusText {{
            color: {t.text_secondary};
        }}
        QLabel#SectionTitle {{
            font-weight: 700;
            font-size: 14px;
            color: {t.text_primary};
        }}
        QLabel#MetricValue {{
            font-weight: 700;
            min-width: 40px;
            color: {t.accent};
        }}

        /* ===== Info / Status Badges ===== */
        QLabel#InfoBadge {{
            border-radius: 12px;
            padding: 4px 10px;
            font-weight: 600;
            font-size: 11px;
            background: {t.badge_neutral_bg};
            color: {t.badge_neutral_fg};
            border: 1px solid {t.border};
        }}

        QLabel#StatusBadge {{
            border-radius: 12px;
            padding: 5px 12px;
            font-weight: 600;
            font-size: 12px;
            background: {t.badge_neutral_bg};
            color: {t.badge_neutral_fg};
            border: 1px solid {t.border};
        }}
        QLabel#StatusBadge[tone="busy"] {{
            background: {t.badge_busy_bg};
            color: {t.info};
            border-color: {t.badge_busy_border};
        }}
        QLabel#StatusBadge[tone="success"] {{
            background: {t.badge_success_bg};
            color: {t.success};
            border-color: {t.badge_success_border};
        }}
        QLabel#StatusBadge[tone="warn"] {{
            background: {t.badge_warn_bg};
            color: {t.warning};
            border-color: {t.badge_warn_border};
        }}
        QLabel#StatusBadge[tone="error"] {{
            background: {t.badge_error_bg};
            color: {t.danger};
            border-color: {t.badge_error_border};
        }}

        /* ===== Inputs ===== */
        QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QScrollArea {{
            border-radius: 10px;
            border: 1px solid {t.border};
            padding: 7px 10px;
            background: {t.bg_elevated};
            color: {t.text_primary};
            selection-background-color: {t.accent_muted};
        }}
        QLineEdit:focus, QPlainTextEdit:focus {{
            border-color: {t.border_focus};
        }}
        QPlainTextEdit {{
            padding: 10px 12px;
        }}
        QListWidget {{
            padding: 8px;
        }}
        QListWidget::item {{
            padding: 4px 6px;
            border-radius: 6px;
        }}
        QListWidget::item:selected {{
            background: {t.accent_muted};
            color: {t.text_primary};
        }}
        QListWidget::item:hover {{
            background: {t.bg_tertiary};
        }}

        /* ===== Combo / Spin ===== */
        QComboBox, QSpinBox {{
            border-radius: 10px;
            border: 1px solid {t.border};
            padding: 6px 10px;
            background: {t.bg_elevated};
            color: {t.text_primary};
            min-height: 28px;
        }}
        QComboBox:focus, QSpinBox:focus {{
            border-color: {t.border_focus};
        }}
        QComboBox::drop-down {{
            border: none;
            padding-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background: {t.bg_elevated};
            color: {t.text_primary};
            border: 1px solid {t.border};
            border-radius: 8px;
            selection-background-color: {t.accent_muted};
        }}

        /* ===== Buttons ===== */
        QPushButton {{
            border-radius: 10px;
            padding: 7px 14px;
            min-height: 34px;
            background: {t.bg_tertiary};
            color: {t.text_primary};
            border: 1px solid {t.border};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background: {t.bg_elevated};
            border-color: {t.accent};
        }}
        QPushButton:pressed {{
            background: {t.accent_muted};
        }}
        QPushButton:disabled {{
            color: {t.text_muted};
            background: {t.bg_secondary};
            border-color: {t.border_subtle};
        }}

        QPushButton#PrimaryButton {{
            background: {t.accent};
            color: {t.text_inverse};
            border: 1px solid {t.accent_pressed};
            font-weight: 700;
        }}
        QPushButton#PrimaryButton:hover {{
            background: {t.accent_hover};
        }}
        QPushButton#PrimaryButton:pressed {{
            background: {t.accent_pressed};
        }}
        QPushButton#PrimaryButton:disabled {{
            background: {t.accent_muted};
            color: {t.text_muted};
            border-color: {t.border};
        }}

        QPushButton#SecondaryButton {{
            background: {t.bg_tertiary};
            border: 1px solid {t.border};
        }}
        QPushButton#SecondaryButton:hover {{
            border-color: {t.accent};
            background: {t.bg_elevated};
        }}

        QPushButton#DangerButton {{
            background: {t.danger_bg};
            color: {t.danger};
            border: 1px solid {t.danger};
            font-weight: 700;
        }}
        QPushButton#DangerButton:hover {{
            background: {t.danger};
            color: {t.text_inverse};
        }}

        /* ===== Tabs ===== */
        QTabWidget::pane {{
            border: 1px solid {t.border};
            border-radius: 12px;
            top: -1px;
            background: {t.bg_secondary};
        }}
        QTabBar::tab {{
            padding: 8px 14px;
            margin-right: 4px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            background: {t.bg_tertiary};
            color: {t.text_secondary};
            border: 1px solid {t.border_subtle};
            border-bottom: none;
        }}
        QTabBar::tab:selected {{
            background: {t.bg_secondary};
            color: {t.text_primary};
            font-weight: 700;
            border-color: {t.border};
        }}
        QTabBar::tab:hover:!selected {{
            background: {t.bg_elevated};
            color: {t.text_primary};
        }}

        /* ===== Progress Bar ===== */
        QProgressBar {{
            border-radius: 8px;
            border: 1px solid {t.border};
            background: {t.bg_tertiary};
            text-align: center;
            color: {t.text_secondary};
        }}
        QProgressBar::chunk {{
            border-radius: 8px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {t.accent}, stop:1 {t.accent_hover});
        }}

        /* ===== Splitter ===== */
        QSplitter::handle {{
            background: {t.border_subtle};
            width: 2px;
            margin: 4px 2px;
            border-radius: 1px;
        }}
        QSplitter::handle:hover {{
            background: {t.accent};
        }}

        /* ===== Scrollbars ===== */
        QScrollBar:vertical {{
            border: none;
            background: {t.scrollbar_track};
            width: 10px;
            border-radius: 5px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {t.scrollbar_thumb};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t.accent_muted};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            border: none;
            background: {t.scrollbar_track};
            height: 10px;
            border-radius: 5px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {t.scrollbar_thumb};
            border-radius: 5px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {t.accent_muted};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        /* ===== Check / Radio ===== */
        QCheckBox {{
            color: {t.text_primary};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid {t.border};
            background: {t.bg_elevated};
        }}
        QCheckBox::indicator:checked {{
            background: {t.accent};
            border-color: {t.accent};
        }}
        QCheckBox::indicator:hover {{
            border-color: {t.accent};
        }}

        /* ===== Slider ===== */
        QSlider::groove:horizontal {{
            height: 6px;
            background: {t.bg_tertiary};
            border-radius: 3px;
            border: 1px solid {t.border_subtle};
        }}
        QSlider::handle:horizontal {{
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
            background: {t.accent};
            border: 2px solid {t.accent_pressed};
        }}
        QSlider::handle:horizontal:hover {{
            background: {t.accent_hover};
        }}
        QSlider::sub-page:horizontal {{
            background: {t.accent};
            border-radius: 3px;
        }}

        /* ===== Dock ===== */
        QDockWidget {{
            color: {t.text_primary};
            titlebar-close-icon: none;
        }}
        QDockWidget::title {{
            background: {t.bg_tertiary};
            padding: 8px 12px;
            border-bottom: 1px solid {t.border};
        }}

        /* ===== Menu ===== */
        QMenu {{
            background: {t.bg_elevated};
            color: {t.text_primary};
            border: 1px solid {t.border};
            border-radius: 10px;
            padding: 6px;
        }}
        QMenu::item {{
            padding: 8px 24px 8px 12px;
            border-radius: 6px;
        }}
        QMenu::item:selected {{
            background: {t.accent_muted};
        }}

        /* ===== Tooltip ===== */
        QToolTip {{
            background: {t.bg_elevated};
            color: {t.text_primary};
            border: 1px solid {t.border};
            border-radius: 8px;
            padding: 6px 10px;
        }}

        /* ===== Model Row ===== */
        QWidget#ModelRow {{
            background: {t.bg_elevated};
            border: 1px solid {t.border_subtle};
            border-radius: 10px;
        }}
        QWidget#ModelRow:hover {{
            border-color: {t.accent};
            background: {t.bg_tertiary};
        }}

        /* ===== Dialog ===== */
        QDialog {{
            background: {t.bg_primary};
            color: {t.text_primary};
        }}
        """
