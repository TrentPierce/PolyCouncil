"""
PolyCouncil theme engine.

The UI consumes semantic tokens from `constants.py` so spacing, typography,
and color choices stay aligned with the design system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from constants import (
    CONTRAST_AUDIT,
    CORNER_RADIUS_MD,
    CORNER_RADIUS_SM,
    DARK_THEME,
    DPI_SCALE,
    BODY_FONT_FAMILY,
    FOCUS_RING_WIDTH,
    FONT_BASE,
    FONT_DISPLAY,
    FONT_LG,
    FONT_SM,
    FONT_XS,
    HEADING_FONT_FAMILY,
    INTERNAL_GAP,
    LIGHT_THEME,
    MIN_TOUCH_TARGET,
    PADDING_LG,
    PADDING_MD,
    PADDING_SM,
    SECTION_GAP,
    THEME,
    dp,
    make_font,
    tooltip_style,
)


@dataclass(frozen=True)
class ColorTokens:
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    fg_primary: str
    fg_secondary: str
    fg_muted: str
    accent: str
    accent_hover: str
    accent_fg: str
    border: str
    border_strong: str
    danger: str
    danger_fg: str
    warning: str
    warning_fg: str
    success: str
    success_fg: str

    @property
    def text_primary(self) -> str:
        return self.fg_primary

    @property
    def text_secondary(self) -> str:
        return self.fg_secondary

    @property
    def text_muted(self) -> str:
        return self.fg_muted

    @property
    def bg_elevated(self) -> str:
        return self.bg_secondary

    @property
    def accent_muted(self) -> str:
        return self.bg_tertiary

    @property
    def danger_bg(self) -> str:
        return self.bg_tertiary

    @property
    def success_bg(self) -> str:
        return self.bg_tertiary


def _tokens_from_theme(theme: dict[str, str]) -> ColorTokens:
    return ColorTokens(**theme)


def apply_theme(root: QtWidgets.QWidget, engine: Optional["ThemeEngine"] = None) -> None:
    theme_engine = engine or getattr(root, "_theme_engine", None)
    if theme_engine is None:
        return
    stylesheet = theme_engine.stylesheet()
    root.setStyleSheet(stylesheet)
    root.setFont(make_font(FONT_BASE))
    for widget in root.findChildren(QtWidgets.QWidget):
        if isinstance(widget, QtWidgets.QLabel) and widget.property("muted"):
            widget.setProperty("muted", True)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()


def toggle_theme(root: QtWidgets.QWidget, engine: Optional["ThemeEngine"] = None) -> dict[str, str]:
    theme_engine = engine or getattr(root, "_theme_engine", None)
    if theme_engine is None:
        return dict(THEME)
    theme_engine.toggle()
    apply_theme(root, theme_engine)
    return dict(THEME)


def apply_shadow(widget: QtWidgets.QWidget, engine: Optional["ThemeEngine"] = None, blur_radius: int = 24, y_offset: int = 8, alpha: int = 20) -> None:
    theme_engine = engine or getattr(widget.window(), "_theme_engine", None)
    is_dark = theme_engine.is_dark if theme_engine else True
    
    shadow = QtWidgets.QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(dp(blur_radius))
    shadow.setXOffset(0)
    shadow.setYOffset(dp(y_offset))
    shadow.setColor(QtGui.QColor(0, 0, 0, int(alpha * 2) if is_dark else alpha))
    widget.setGraphicsEffect(shadow)


class ThemeEngine(QtCore.QObject):
    themeChanged = QtCore.Signal()

    def __init__(self, app: QtWidgets.QApplication, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._app = app
        self._mode = "dark" if self._detect_dark() else "light"
        self._tokens = _tokens_from_theme(DARK_THEME if self._mode == "dark" else LIGHT_THEME)
        self._last_contrast_failures = {
            mode: [item for item in results if not item.passes]
            for mode, results in CONTRAST_AUDIT.items()
        }

    @property
    def is_dark(self) -> bool:
        return self._mode == "dark"

    @property
    def tokens(self) -> ColorTokens:
        return self._tokens

    @property
    def font_family(self) -> str:
        return BODY_FONT_FAMILY

    @property
    def contrast_failures(self):
        return self._last_contrast_failures

    def apply(self, window: QtWidgets.QWidget) -> None:
        theme = DARK_THEME if self._mode == "dark" else LIGHT_THEME
        THEME.clear()
        THEME.update(theme)
        self._tokens = _tokens_from_theme(theme)
        self._app.setFont(make_font(FONT_BASE))
        apply_theme(window, self)

    def refresh(self, window: QtWidgets.QWidget) -> None:
        new_mode = "dark" if self._detect_dark() else "light"
        if new_mode != self._mode:
            self._mode = new_mode
            self.apply(window)
            self.themeChanged.emit()

    def toggle(self) -> None:
        self._mode = "light" if self._mode == "dark" else "dark"
        theme = DARK_THEME if self._mode == "dark" else LIGHT_THEME
        THEME.clear()
        THEME.update(theme)
        self._tokens = _tokens_from_theme(theme)
        self.themeChanged.emit()

    def stylesheet(self) -> str:
        t = self._tokens
        base_font = self.font_family
        tooltip_qss = tooltip_style(THEME)
        return f"""
        {tooltip_qss}
        QWidget {{
            background-color: {t.bg_primary};
            color: {t.fg_primary};
            font-family: "{base_font}", "Segoe UI", sans-serif;
            font-size: {FONT_BASE}pt;
        }}
        QWidget#Root {{
            background-color: {t.bg_primary};
        }}
        QLabel#HeroTitle {{
            font-family: "{HEADING_FONT_FAMILY}", "{base_font}", sans-serif;
            font-size: {FONT_DISPLAY}pt;
            font-weight: 700;
            color: {t.fg_primary};
        }}
        QLabel#HeroSubtitle,
        QLabel#HintLabel,
        QLabel#StatusText,
        QLabel#FieldHelper {{
            font-size: {FONT_SM}pt;
            color: {t.fg_secondary};
        }}
        QLabel#HintLabel[error="true"],
        QLabel#FieldHelper[error="true"] {{
            color: {t.danger};
        }}
        QLabel#SectionEyebrow {{
            font-size: {FONT_XS}pt;
            letter-spacing: 1px;
            color: {t.fg_muted};
            font-weight: 700;
        }}
        QLabel#PanelTitle,
        QLabel#WorkflowStepTitle {{
            font-family: "{HEADING_FONT_FAMILY}", "{base_font}", sans-serif;
            font-size: {FONT_LG}pt;
            font-weight: 700;
            color: {t.fg_primary};
        }}
        QLabel#FieldLabel {{
            font-size: {FONT_SM}pt;
            font-weight: 600;
            color: {t.fg_secondary};
        }}
        QLabel#StatusBadge,
        QLabel#HeaderStatus,
        QPushButton#HeaderStatusButton,
        QLabel#InfoBadge {{
            background-color: {t.bg_tertiary};
            color: {t.fg_secondary};
            border: 1px solid {t.border};
            border-radius: {dp(MIN_TOUCH_TARGET // 2)}px;
            padding: {dp(PADDING_SM)}px {dp(PADDING_MD)}px;
            min-height: {dp(MIN_TOUCH_TARGET - PADDING_SM)}px;
            font-size: {FONT_SM}pt;
            font-weight: 600;
        }}
        QLabel#StatusBadge[tone="busy"],
        QPushButton#HeaderStatusButton[tone="busy"] {{
            background-color: {t.accent};
            color: {t.accent_fg};
            border-color: {t.accent_hover};
        }}
        QLabel#StatusBadge[tone="success"],
        QPushButton#HeaderStatusButton[tone="success"] {{
            background-color: {t.success};
            color: {t.success_fg};
            border-color: {t.success};
        }}
        QLabel#StatusBadge[tone="warn"],
        QPushButton#HeaderStatusButton[tone="warn"] {{
            background-color: {t.warning};
            color: {t.warning_fg};
            border-color: {t.warning};
        }}
        QLabel#StatusBadge[tone="error"],
        QPushButton#HeaderStatusButton[tone="error"] {{
            background-color: {t.danger};
            color: {t.danger_fg};
            border-color: {t.danger};
        }}
        QFrame#HeroCard,
        QFrame#ComposerCard,
        QFrame#PanelCard,
        QFrame#Toast,
        QFrame#OnboardingCard,
        QFrame#OverlayCard,
        QFrame#CollapsibleFrame,
        QWidget#WorkflowStepCard,
        QWidget#ModelRow,
        QDockWidget {{
            background-color: {t.bg_secondary};
            border: 1px solid {t.border};
            border-radius: {dp(CORNER_RADIUS_MD)}px;
        }}
        QFrame#HeroCard {{
            border-color: {t.border_strong};
        }}
        QFrame#PanelCard,
        QFrame#ComposerCard,
        QWidget#WorkflowStepCard,
        QWidget#ModelRow {{
            background-color: {t.bg_secondary};
        }}
        QDockWidget::title {{
            padding: {dp(PADDING_MD)}px {dp(PADDING_LG)}px;
            color: {t.fg_secondary};
            font-size: {FONT_SM}pt;
        }}
        QLineEdit,
        QPlainTextEdit,
        QTextBrowser,
        QListWidget,
        QScrollArea,
        QComboBox,
        QSpinBox,
        QTextEdit {{
            background-color: {t.bg_primary};
            color: {t.fg_primary};
            border: 1px solid {t.border};
            border-bottom: 2px solid {t.border_strong};
            border-radius: {dp(CORNER_RADIUS_SM)}px;
            padding: {dp(PADDING_SM)}px {dp(PADDING_MD)}px;
            selection-background-color: {t.accent};
            selection-color: {t.accent_fg};
            min-height: {dp(MIN_TOUCH_TARGET)}px;
        }}
        QListWidget::item {{
            padding-top: {dp(PADDING_MD)}px;
            padding-bottom: {dp(PADDING_MD)}px;
        }}
        QLineEdit:disabled,
        QPlainTextEdit:disabled,
        QTextBrowser:disabled,
        QListWidget:disabled,
        QComboBox:disabled,
        QSpinBox:disabled,
        QPushButton:disabled {{
            color: {t.fg_muted};
            background-color: {t.bg_tertiary};
            border-color: {t.border};
        }}
        QLineEdit:focus,
        QPlainTextEdit:focus,
        QTextBrowser:focus,
        QListWidget:focus,
        QComboBox:focus,
        QSpinBox:focus,
        QPushButton:focus,
        QCheckBox:focus,
        QRadioButton:focus {{
            border: 1px solid {t.accent};
            border-bottom: 2px solid {t.accent};
            outline: none;
        }}
        QPushButton {{
            min-height: {dp(MIN_TOUCH_TARGET)}px;
            min-width: {dp(MIN_TOUCH_TARGET)}px;
            padding: {dp(PADDING_MD)}px {dp(PADDING_LG)}px;
            border-radius: {dp(CORNER_RADIUS_SM)}px;
            border: 1px solid {t.border};
            background-color: {t.bg_secondary};
            color: {t.fg_primary};
            font-size: {FONT_BASE}pt;
        }}
        QPushButton:hover {{
            border-color: {t.border_strong};
        }}
        QPushButton#PrimaryButton,
        QPushButton#RunStateButton[runState="ready"],
        QPushButton#RunStateButton[runState="running"] {{
            background-color: {t.accent};
            color: {t.accent_fg};
            border-color: {t.accent_hover};
            font-weight: 700;
        }}
        QPushButton#PrimaryButton:hover,
        QPushButton#RunStateButton[runState="ready"]:hover,
        QPushButton#RunStateButton[runState="running"]:hover {{
            background-color: {t.accent_hover};
        }}
        QPushButton#SecondaryButton,
        QPushButton#GhostButton {{
            background-color: {t.bg_tertiary};
            color: {t.fg_primary};
            border-color: {t.border};
        }}
        QPushButton#GhostButton {{
            background-color: transparent;
            color: {t.fg_secondary};
        }}
        QPushButton#DangerButton {{
            background-color: {t.danger};
            color: {t.danger_fg};
            border-color: {t.danger};
            font-weight: 700;
        }}
        QWidget#WorkflowStepCard[stepState="complete"] QLabel#WorkflowStepBadge {{
            background-color: {t.success};
            color: {t.success_fg};
            border-color: {t.success};
        }}
        QWidget#WorkflowStepCard[stepState="active"] QLabel#WorkflowStepBadge {{
            background-color: {t.accent};
            color: {t.accent_fg};
            border-color: {t.accent_hover};
        }}
        QWidget#WorkflowStepCard[stepState="locked"] QLabel,
        QWidget#WorkflowStepCard[stepState="locked"] QPushButton {{
            color: {t.fg_muted};
        }}
        QLabel#WorkflowStepBadge {{
            min-width: {dp(MIN_TOUCH_TARGET)}px;
            min-height: {dp(MIN_TOUCH_TARGET)}px;
            max-width: {dp(MIN_TOUCH_TARGET)}px;
            max-height: {dp(MIN_TOUCH_TARGET)}px;
            border-radius: {dp(MIN_TOUCH_TARGET // 2)}px;
            border: 1px solid {t.border};
            background-color: {t.bg_tertiary};
            color: {t.fg_secondary};
            font-weight: 700;
        }}
        QTabWidget::pane {{
            border: 1px solid {t.border};
            border-radius: {dp(CORNER_RADIUS_MD)}px;
            background-color: {t.bg_secondary};
        }}
        QTabBar::tab {{
            background-color: {t.bg_tertiary};
            color: {t.fg_secondary};
            padding: {dp(PADDING_MD)}px {dp(PADDING_LG)}px;
            margin-right: {dp(PADDING_SM)}px;
            border-top-left-radius: {dp(CORNER_RADIUS_SM)}px;
            border-top-right-radius: {dp(CORNER_RADIUS_SM)}px;
            min-height: {dp(MIN_TOUCH_TARGET)}px;
        }}
        QTabBar::tab:selected {{
            color: {t.fg_primary};
            background-color: {t.bg_secondary};
            font-weight: 700;
        }}
        QProgressBar {{
            border: 1px solid {t.border};
            border-radius: {dp(CORNER_RADIUS_SM)}px;
            background-color: {t.bg_tertiary};
            text-align: center;
            min-height: {dp(PADDING_LG)}px;
        }}
        QProgressBar::chunk {{
            background-color: {t.accent};
            border-radius: {dp(CORNER_RADIUS_SM)}px;
        }}
        QCheckBox,
        QRadioButton {{
            spacing: {dp(PADDING_MD)}px;
            min-height: {dp(MIN_TOUCH_TARGET)}px;
            color: {t.fg_primary};
        }}
        QCheckBox::indicator,
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
        }}
        QListWidget#LeaderboardList::item {{
            margin-top: {dp(PADDING_SM)}px;
            margin-bottom: {dp(PADDING_SM)}px;
        }}
        QWidget#SidebarCol {{
            background-color: transparent;
        }}
        QScrollBar:vertical {{
            background-color: {t.bg_primary};
            width: {dp(PADDING_LG)}px;
            margin: {dp(PADDING_SM)}px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {t.border_strong};
            min-height: {dp(32)}px;
            border-radius: {dp(CORNER_RADIUS_SM)}px;
        }}
        """

    def _detect_dark(self) -> bool:
        return self._app.palette().color(QtGui.QPalette.Window).lightness() < 128
