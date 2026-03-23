from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from constants import (
    ACTIONS,
    CORNER_RADIUS_SM,
    FONT_BASE,
    FONT_LG,
    FONT_SM,
    FONT_XS,
    INTERNAL_GAP,
    MIN_TOUCH_TARGET,
    PADDING_LG,
    PADDING_MD,
    PADDING_SM,
    TOOLTIP_HIDE_DELAY_MS,
    TOOLTIP_MAX_WIDTH,
    TOOLTIP_SHOW_DELAY_MS,
    dp,
    line_height_for,
    make_font,
    make_monospace_font,
)


class StyleMixin:
    @staticmethod
    def apply_focus_policy(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        return widget


def configure_tooltip(widget: QtWidgets.QWidget, text: str) -> None:
    widget.setToolTip(text)
    widget.setProperty("tooltipShowDelay", TOOLTIP_SHOW_DELAY_MS)
    widget.setProperty("tooltipHideDelay", TOOLTIP_HIDE_DELAY_MS)
    widget.setProperty("tooltipMaxWidth", dp(TOOLTIP_MAX_WIDTH))


def make_label(
    text: str,
    *,
    object_name: str | None = None,
    heading: bool = False,
    muted: bool = False,
    wrap: bool = False,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(wrap)
    label.setFont(make_font(FONT_LG if heading else FONT_SM if muted else FONT_BASE, bold=heading))
    label.setMinimumHeight(dp(line_height_for(FONT_LG if heading else FONT_BASE)))
    if muted:
        label.setProperty("muted", True)
    return label


def make_button(
    text: str,
    *,
    variant: str = "secondary",
    action_id: str | None = None,
) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setObjectName(
        "PrimaryButton" if variant == "primary"
        else "DangerButton" if variant == "danger"
        else "GhostButton" if variant == "ghost"
        else "SecondaryButton"
    )
    button.setMinimumHeight(dp(MIN_TOUCH_TARGET))
    button.setMinimumWidth(dp(MIN_TOUCH_TARGET))
    button.setFont(make_font(FONT_BASE, bold=variant in {"primary", "danger"}))
    StyleMixin.apply_focus_policy(button)
    if action_id and action_id in ACTIONS:
        configure_tooltip(button, ACTIONS[action_id]["tooltip"])
    return button


def make_entry(placeholder: str = "") -> QtWidgets.QLineEdit:
    entry = QtWidgets.QLineEdit()
    entry.setPlaceholderText(placeholder)
    entry.setMinimumHeight(dp(MIN_TOUCH_TARGET))
    entry.setFont(make_font(FONT_BASE))
    StyleMixin.apply_focus_policy(entry)
    return entry


def make_combobox(items: Iterable[str] | None = None) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.setMinimumHeight(dp(MIN_TOUCH_TARGET))
    combo.setFont(make_font(FONT_BASE))
    StyleMixin.apply_focus_policy(combo)
    if items:
        combo.addItems(list(items))
    return combo


def make_text_browser(object_name: str = "OutputView") -> QtWidgets.QTextBrowser:
    browser = QtWidgets.QTextBrowser()
    browser.setObjectName(object_name)
    browser.setOpenExternalLinks(True)
    browser.setReadOnly(True)
    browser.setUndoRedoEnabled(False)
    browser.setFrameShape(QtWidgets.QFrame.NoFrame)
    browser.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
    browser.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
    browser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    browser.document().setDocumentMargin(dp(PADDING_LG))
    browser.setFont(make_font(FONT_BASE))
    return browser


def make_monospace_output() -> QtWidgets.QPlainTextEdit:
    edit = QtWidgets.QPlainTextEdit()
    edit.setObjectName("OutputLog")
    edit.setReadOnly(True)
    edit.setFont(make_monospace_font())
    edit.setMinimumHeight(dp(MIN_TOUCH_TARGET))
    return edit


def apply_form_grid(layout: QtWidgets.QGridLayout) -> None:
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(dp(PADDING_MD))
    layout.setVerticalSpacing(dp(INTERNAL_GAP))
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 1)


def set_box_layout_spacing(layout: QtWidgets.QBoxLayout, *, margins: int = 0, spacing: int = INTERNAL_GAP) -> None:
    layout.setContentsMargins(dp(margins), dp(margins), dp(margins), dp(margins))
    layout.setSpacing(dp(spacing))


def make_section_title(text: str) -> QtWidgets.QLabel:
    label = make_label(text, object_name="PanelTitle", heading=True)
    label.setFont(make_font(FONT_LG, bold=True))
    return label


def make_hint_label(text: str) -> QtWidgets.QLabel:
    label = make_label(text, object_name="HintLabel", muted=True, wrap=True)
    label.setFont(make_font(FONT_XS))
    return label


def make_monospace_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("MonoLabel")
    label.setFont(make_monospace_font())
    return label
