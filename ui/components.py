from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from constants import (
    ACTIONS,
    FONT_BASE,
    FONT_LG,
    FONT_MD,
    FONT_SM,
    FONT_XL,
    INTERNAL_GAP,
    MIN_TOUCH_TARGET,
    PADDING_LG,
    PADDING_MD,
    PADDING_SM,
    SECTION_GAP,
    STRINGS,
    TOOLTIP_HIDE_DELAY_MS,
    TOOLTIP_MAX_WIDTH,
    TOOLTIP_SHOW_DELAY_MS,
    dp,
    make_font,
    make_monospace_font,
)
from ui.animations import FadeIn, FadeOut, SmoothCollapse
from ui.factories import configure_tooltip


class CollapsibleGroupBox(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, start_collapsed: bool = False):
        super().__init__(parent)
        self._expanded = not start_collapsed

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._frame = QtWidgets.QFrame()
        self._frame.setObjectName("CollapsibleFrame")
        frame_layout = QtWidgets.QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        self._header = QtWidgets.QFrame()
        self._header.setObjectName("CollapsibleHeader")
        self._header.setCursor(QtCore.Qt.PointingHandCursor)
        self._header.setFocusPolicy(QtCore.Qt.StrongFocus)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_MD), dp(PADDING_LG), dp(PADDING_MD))
        header_layout.setSpacing(dp(INTERNAL_GAP))

        self._arrow_label = QtWidgets.QLabel("▾" if self._expanded else "▸")
        self._arrow_label.setFixedWidth(dp(MIN_TOUCH_TARGET))
        self._arrow_label.setAlignment(QtCore.Qt.AlignCenter)
        self._arrow_label.setFont(make_font(FONT_MD, bold=True))

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setObjectName("PanelTitle")
        self._title_label.setFont(make_font(FONT_LG, bold=True))

        header_layout.addWidget(self._arrow_label)
        header_layout.addWidget(self._title_label, 1)

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(dp(PADDING_LG), 0, dp(PADDING_LG), dp(PADDING_LG))
        self._content_layout.setSpacing(dp(INTERNAL_GAP))

        frame_layout.addWidget(self._header)
        frame_layout.addWidget(self._content)
        outer.addWidget(self._frame)

        self._header.mousePressEvent = self._on_header_click
        if start_collapsed:
            self._content.hide()
            self._content.setMaximumHeight(0)

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        return self._content_layout

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QtWidgets.QLayout) -> None:
        self._content_layout.addLayout(layout)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._arrow_label.setText("▾" if expanded else "▸")
        if animate:
            if expanded:
                SmoothCollapse.expand(self._content, duration_ms=180)
            else:
                SmoothCollapse.collapse(self._content, duration_ms=180)
        else:
            self._content.setVisible(expanded)
            self._content.setMaximumHeight(16777215 if expanded else 0)
        self.toggled.emit(expanded)

    def _on_header_click(self, event: QtGui.QMouseEvent) -> None:
        self.set_expanded(not self._expanded)
        event.accept()


class ToastNotification(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget, message: str, tone: str = "info", duration_ms: int = 3500):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_MD), dp(PADDING_LG), dp(PADDING_MD))
        layout.setSpacing(dp(INTERNAL_GAP))

        icon_label = QtWidgets.QLabel({"info": "i", "success": "✓", "warn": "!", "error": "×"}.get(tone, "i"))
        icon_label.setObjectName("StatusBadge")
        icon_label.setProperty("tone", "warn" if tone == "warn" else tone if tone in {"success", "error"} else "busy")
        icon_label.setFixedSize(dp(MIN_TOUCH_TARGET), dp(MIN_TOUCH_TARGET))
        icon_label.setAlignment(QtCore.Qt.AlignCenter)

        msg_label = QtWidgets.QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setMinimumWidth(dp(240))
        msg_label.setFont(make_font(FONT_BASE))
        msg_label.setMaximumWidth(dp(360))

        layout.addWidget(icon_label, 0)
        layout.addWidget(msg_label, 1)

        self._duration_ms = duration_ms
        self._dismiss_timer = QtCore.QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._dismiss)

    def show_toast(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(
            parent.width() - self.width() - dp(PADDING_LG),
            parent.height() - self.height() - dp(PADDING_LG),
        )
        self.show()
        self.raise_()
        FadeIn.run(self, duration_ms=180)
        self._dismiss_timer.start(self._duration_ms)

    def _dismiss(self) -> None:
        anim = FadeOut.run(self, duration_ms=220, hide_on_finish=True)
        if anim:
            anim.finished.connect(self.deleteLater)
        else:
            self.deleteLater()


class EnhancedPromptEditor(QtWidgets.QWidget):
    submitRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(dp(PADDING_SM))

        self._editor = _PromptTextEdit(self)
        self._editor.submitRequested.connect(self.submitRequested.emit)
        self._editor.setAccessibleName("Prompt editor")
        self._editor.setAccessibleDescription(STRINGS["tooltip_prompt"])
        configure_tooltip(self._editor, STRINGS["tooltip_prompt"])
        layout.addWidget(self._editor)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(dp(PADDING_SM), 0, dp(PADDING_SM), 0)
        footer.setSpacing(dp(INTERNAL_GAP))

        self._hint_label = QtWidgets.QLabel(
            "Enter sends. Shift+Enter inserts a new line. Ctrl+Shift+A selects all models."
        )
        self._hint_label.setObjectName("HintLabel")
        self._hint_label.setWordWrap(True)

        self._char_count = QtWidgets.QLabel("0")
        self._char_count.setObjectName("HintLabel")
        self._char_count.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        footer.addWidget(self._hint_label, 1)
        footer.addWidget(self._char_count)
        layout.addLayout(footer)

        self._editor.textChanged.connect(self._update_char_count)

    def toPlainText(self) -> str:
        return self._editor.toPlainText()

    def setPlainText(self, text: str) -> None:
        self._editor.setPlainText(text)

    def clear(self) -> None:
        self._editor.clear()

    def setPlaceholderText(self, text: str) -> None:
        self._editor.setPlaceholderText(text)

    def setFocus(self) -> None:
        self._editor.setFocus()

    def _update_char_count(self) -> None:
        self._char_count.setText(f"{len(self._editor.toPlainText()):,}")


class _PromptTextEdit(QtWidgets.QPlainTextEdit):
    submitRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._min_height = dp(80)
        self._max_height = dp(180)
        self.setTabChangesFocus(True)
        self.setPlaceholderText("Ask the council a question. Press Enter to send, Shift+Enter for a new line.")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.document().documentLayout().documentSizeChanged.connect(self._sync_height)
        self._sync_height()

    def _sync_height(self, *_args) -> None:
        target = int(self.document().size().height() + self.frameWidth() * 2 + dp(PADDING_LG))
        target = max(self._min_height, min(target, self._max_height))
        self.setFixedHeight(target)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not (event.modifiers() & QtCore.Qt.ShiftModifier):
            self.submitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_height)

    def clear(self) -> None:
        super().clear()
        self._sync_height()


class AnimatedStatusBar(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, dp(PADDING_SM), 0, dp(PADDING_SM))
        layout.setSpacing(dp(INTERNAL_GAP))

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximum(0)
        self.progress.setFixedWidth(dp(120))
        self.progress.hide()

        self.badge = QtWidgets.QLabel("Idle")
        self.badge.setObjectName("StatusBadge")

        self.message = QtWidgets.QLabel("Ready.")
        self.message.setObjectName("StatusText")
        self.message.setWordWrap(True)
        self.message.setFont(make_font(FONT_SM))

        self.footer_link = QtWidgets.QLabel('<a href="https://github.com/TrentPierce">Trent Pierce · GitHub</a>')
        self.footer_link.setOpenExternalLinks(True)

        layout.addWidget(self.progress, 0)
        layout.addWidget(self.badge, 0)
        layout.addWidget(self.message, 1)
        layout.addWidget(self.footer_link, 0)

    def set_status(self, text: str) -> None:
        self.message.setText(text)
        lowered = text.lower()
        if any(word in lowered for word in ("error", "failed")):
            self.badge.setText("Error")
            self._set_badge_tone("error")
        elif any(word in lowered for word in ("warning", "select", "unavailable", "no ")) and not self.progress.isVisible():
            self.badge.setText("Attention")
            self._set_badge_tone("warn")
        elif self.progress.isVisible():
            self.badge.setText("Working")
            self._set_badge_tone("busy")
        elif any(word in lowered for word in ("done", "ready", "complete", "saved", "added", "found", "loaded", "updated", "cleared")):
            self.badge.setText("Ready")
            self._set_badge_tone("success")
        else:
            self.badge.setText("Info")
            self._set_badge_tone("neutral")

    def set_busy(self, on: bool) -> None:
        self.progress.setVisible(on)
        self.progress.setMaximum(0 if on else 1)
        self.badge.setText("Working" if on else "Idle")
        self._set_badge_tone("busy" if on else "neutral")

    def _set_badge_tone(self, tone: str) -> None:
        self.badge.setProperty("tone", tone)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.badge.update()


class StatefulRunButton(QtWidgets.QPushButton):
    runRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()

    LOCKED = "locked"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._run_state = self.LOCKED
        self.setObjectName("RunStateButton")
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setMinimumHeight(dp(MIN_TOUCH_TARGET))
        self.clicked.connect(self._handle_click)
        configure_tooltip(self, ACTIONS["run_council"]["tooltip"])
        self.set_run_state(self.LOCKED)

    def run_state(self) -> str:
        return self._run_state

    def set_run_state(self, state: str) -> None:
        self._run_state = state
        self.setProperty("runState", state)
        if state == self.LOCKED:
            self.setText("Run council")
            self.setEnabled(False)
            self.setIcon(QtGui.QIcon())
        elif state == self.READY:
            self.setText("Run council")
            self.setEnabled(True)
            self.setIcon(QtGui.QIcon())
        elif state == self.RUNNING:
            self.setText("Running... Stop")
            self.setEnabled(True)
            self.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserStop))
        else:
            self.setText("Stopping...")
            self.setEnabled(False)
            self.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserStop))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _handle_click(self) -> None:
        if self._run_state == self.READY:
            self.runRequested.emit()
        elif self._run_state == self.RUNNING:
            self.stopRequested.emit()


class WorkflowStepCard(QtWidgets.QWidget):
    headerClicked = QtCore.Signal(int)
    stepCompleted = QtCore.Signal(int, bool)

    IDLE = "idle"
    ACTIVE = "active"
    COMPLETE = "complete"
    LOCKED = "locked"

    def __init__(
        self,
        step_number: int,
        title: str,
        description: str = "",
        *,
        collapsible: bool = True,
        start_collapsed: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.step_number = step_number
        self._collapsible = collapsible
        self._collapsed = bool(start_collapsed)
        self._state = self.IDLE

        self.setObjectName("WorkflowStepCard")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QtWidgets.QFrame()
        self.header.setObjectName("WorkflowStepHeader")
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        self.header.setFocusPolicy(QtCore.Qt.StrongFocus)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_MD), dp(PADDING_LG), dp(PADDING_MD))
        header_layout.setSpacing(dp(INTERNAL_GAP))

        self.badge = QtWidgets.QLabel(str(step_number))
        self.badge.setObjectName("WorkflowStepBadge")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)

        title_block = QtWidgets.QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(dp(PADDING_SM))

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("WorkflowStepTitle")
        self.title_label.setFont(make_font(FONT_LG, bold=True))

        self.summary_label = QtWidgets.QLabel(description)
        self.summary_label.setObjectName("HintLabel")
        self.summary_label.setWordWrap(True)

        self.chevron = QtWidgets.QLabel("▾")
        self.chevron.setObjectName("WorkflowStepChevron")
        self.chevron.setMinimumWidth(dp(MIN_TOUCH_TARGET))
        self.chevron.setAlignment(QtCore.Qt.AlignCenter)

        title_block.addWidget(self.title_label)
        title_block.addWidget(self.summary_label)

        header_layout.addWidget(self.badge)
        header_layout.addLayout(title_block, 1)
        header_layout.addWidget(self.chevron)

        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(dp(PADDING_LG), 0, dp(PADDING_LG), dp(PADDING_LG))
        self.content_layout.setSpacing(dp(INTERNAL_GAP))

        outer.addWidget(self.header)
        outer.addWidget(self.content)

        self.header.mousePressEvent = self._handle_header_click
        self.set_state(self.IDLE)
        self.set_collapsed(start_collapsed, animate=False)

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout: QtWidgets.QLayout) -> None:
        self.content_layout.addLayout(layout)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def set_state(self, state: str) -> None:
        previous_complete = self._state == self.COMPLETE
        self._state = state
        self.badge.setText("✓" if state == self.COMPLETE else str(self.step_number))
        for widget in (self, self.header, self.badge, self.title_label, self.summary_label, self.chevron):
            widget.setProperty("stepState", state)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()
        if (state == self.COMPLETE) != previous_complete:
            self.stepCompleted.emit(self.step_number, state == self.COMPLETE)

    def set_collapsed(self, collapsed: bool, *, animate: bool = True) -> None:
        self._collapsed = bool(collapsed)
        if not self._collapsible:
            self.chevron.hide()
            return
        self.chevron.setText("▸" if collapsed else "▾")
        if animate:
            if collapsed:
                SmoothCollapse.collapse(self.content, duration_ms=180)
            else:
                SmoothCollapse.expand(self.content, duration_ms=180)
        else:
            self.content.setVisible(not collapsed)
            self.content.setMaximumHeight(0 if collapsed else 16777215)

    def _handle_header_click(self, event: QtGui.QMouseEvent) -> None:
        if self._state == self.LOCKED:
            event.accept()
            return
        self.headerClicked.emit(self.step_number)
        if self._collapsible:
            self.set_collapsed(not self._collapsed)
        event.accept()


class OnboardingOverlay(QtWidgets.QWidget):
    dismissed = QtCore.Signal()

    STEPS = [
        {
            "title": "Welcome to PolyCouncil",
            "body": "PolyCouncil lets you run multiple AI models in parallel, compare their responses, and review the council result in one workspace.",
        },
        {
            "title": "Step 1: Connect a Provider",
            "body": "Choose a provider, confirm the endpoint details, and load models before you begin.",
        },
        {
            "title": "Step 2: Select Models",
            "body": "Pick the models you want in the council. You can assign personas to shape how they respond.",
        },
        {
            "title": "Step 3: Ask a Question",
            "body": "Write the prompt in the composer and run the council. Results and logs stay in the review panel.",
        },
    ]

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._step = 0
        parent.installEventFilter(self)

        self._card = QtWidgets.QFrame(self)
        self._card.setObjectName("OnboardingCard")
        card_layout = QtWidgets.QVBoxLayout(self._card)
        card_layout.setContentsMargins(dp(SECTION_GAP), dp(SECTION_GAP), dp(SECTION_GAP), dp(PADDING_LG))
        card_layout.setSpacing(dp(INTERNAL_GAP))

        self._step_indicator = QtWidgets.QLabel()
        self._step_indicator.setObjectName("HintLabel")
        self._step_indicator.setAlignment(QtCore.Qt.AlignCenter)

        self._title_label = QtWidgets.QLabel()
        self._title_label.setFont(make_font(FONT_XL, bold=True))
        self._title_label.setWordWrap(True)
        self._title_label.setAlignment(QtCore.Qt.AlignCenter)

        self._body_label = QtWidgets.QLabel()
        self._body_label.setFont(make_font(FONT_MD))
        self._body_label.setWordWrap(True)
        self._body_label.setAlignment(QtCore.Qt.AlignCenter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(dp(INTERNAL_GAP))
        self._skip_btn = QtWidgets.QPushButton("Skip")
        self._skip_btn.setObjectName("SecondaryButton")
        self._next_btn = QtWidgets.QPushButton("Next")
        self._next_btn.setObjectName("PrimaryButton")
        self._skip_btn.setMinimumHeight(dp(MIN_TOUCH_TARGET))
        self._next_btn.setMinimumHeight(dp(MIN_TOUCH_TARGET))
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._next_btn)

        card_layout.addWidget(self._step_indicator)
        card_layout.addWidget(self._title_label)
        card_layout.addWidget(self._body_label)
        card_layout.addStretch(1)
        card_layout.addLayout(btn_row)

        self._skip_btn.clicked.connect(self._dismiss)
        self._next_btn.clicked.connect(self._advance)
        self._render_step()

    def _render_step(self) -> None:
        step = self.STEPS[self._step]
        self._step_indicator.setText(f"Step {self._step + 1} of {len(self.STEPS)}")
        self._title_label.setText(step["title"])
        self._body_label.setText(step["body"])
        self._next_btn.setText("Get Started" if self._step == len(self.STEPS) - 1 else "Next")

    def _advance(self) -> None:
        self._step += 1
        if self._step >= len(self.STEPS):
            self._dismiss()
            return
        self._render_step()

    def _dismiss(self) -> None:
        anim = FadeOut.run(self, duration_ms=180, hide_on_finish=True)
        if anim:
            anim.finished.connect(self.dismissed.emit)
        else:
            self.dismissed.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._center_card()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.parentWidget() and event.type() == QtCore.QEvent.Resize:
            self.setGeometry(self.parentWidget().rect())
            self._center_card()
        return super().eventFilter(obj, event)

    def _center_card(self) -> None:
        card_w = min(dp(560), self.width() - dp(SECTION_GAP * 2))
        self._card.setFixedWidth(card_w)
        self._card.adjustSize()
        self._card.move((self.width() - card_w) // 2, max(dp(PADDING_LG), (self.height() - self._card.height()) // 2))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 160))
        painter.end()


class KeyboardShortcutOverlay(QtWidgets.QWidget):
    SHORTCUTS = [
        ("Enter", "Send prompt to council"),
        ("Shift+Enter", "Insert a new line"),
        ("Ctrl+Enter", "Run from anywhere"),
        ("Ctrl+Shift+A", "Select all models"),
        ("Ctrl+R", "Reload models"),
        ("Ctrl+L", "Focus prompt editor"),
        ("Ctrl+F", "Focus model filter"),
        ("Escape", "Stop current operation"),
        ("Ctrl+?", "Toggle this help"),
        ("Ctrl+Shift+T", "Toggle theme"),
    ]

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.hide()
        parent.installEventFilter(self)

        self._card = QtWidgets.QFrame(self)
        self._card.setObjectName("OverlayCard")
        card_layout = QtWidgets.QVBoxLayout(self._card)
        card_layout.setContentsMargins(dp(SECTION_GAP), dp(PADDING_LG), dp(SECTION_GAP), dp(PADDING_LG))
        card_layout.setSpacing(dp(PADDING_MD))

        title = QtWidgets.QLabel("Keyboard Shortcuts")
        title.setObjectName("PanelTitle")
        title.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(title)
        card_layout.addSpacing(dp(PADDING_MD))

        for key, desc in self.SHORTCUTS:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(dp(INTERNAL_GAP))
            key_label = QtWidgets.QLabel(key)
            key_label.setObjectName("InfoBadge")
            key_label.setFont(make_monospace_font())
            key_label.setAlignment(QtCore.Qt.AlignCenter)
            key_label.setMinimumWidth(dp(140))
            desc_label = QtWidgets.QLabel(desc)
            desc_label.setObjectName("HintLabel")
            row.addWidget(key_label)
            row.addWidget(desc_label, 1)
            card_layout.addLayout(row)

        card_layout.addStretch(1)
        close_hint = QtWidgets.QLabel("Press Escape or Ctrl+? to close")
        close_hint.setObjectName("HintLabel")
        close_hint.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(close_hint)

    def toggle(self) -> None:
        if self.isVisible():
            FadeOut.run(self, duration_ms=180, hide_on_finish=True)
            return
        self.setGeometry(self.parentWidget().rect())
        self._center_card()
        self.show()
        self.raise_()
        FadeIn.run(self, duration_ms=180)

    def _center_card(self) -> None:
        card_w = min(dp(520), self.width() - dp(SECTION_GAP * 2))
        card_h = min(dp(420), self.height() - dp(SECTION_GAP * 2))
        self._card.setFixedSize(card_w, card_h)
        self._card.move((self.width() - card_w) // 2, (self.height() - card_h) // 2)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._center_card()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.parentWidget() and event.type() == QtCore.QEvent.Resize:
            self.setGeometry(self.parentWidget().rect())
            self._center_card()
        return super().eventFilter(obj, event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 140))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._card.geometry().contains(event.pos()):
            self.toggle()
            event.accept()
            return
        super().mousePressEvent(event)


class ModelCard(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)
    personaClicked = QtCore.Signal()

    def __init__(self, model_id: str, provider_name: str, meta_text: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModelRow")
        self.setMinimumHeight(dp(MIN_TOUCH_TARGET))

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(dp(PADDING_LG), dp(PADDING_MD), dp(PADDING_LG), dp(PADDING_MD))
        layout.setSpacing(dp(INTERNAL_GAP))

        self.checkbox = QtWidgets.QCheckBox(model_id)
        self.checkbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.checkbox.setAccessibleName(f"Select model {model_id}")
        self.checkbox.setAccessibleDescription(f"Toggle participation for model {model_id}.")
        self.checkbox.toggled.connect(self.toggled.emit)

        self.provider_badge = QtWidgets.QLabel(provider_name)
        self.provider_badge.setObjectName("InfoBadge")

        self.meta_label = QtWidgets.QLabel(meta_text)
        self.meta_label.setObjectName("HintLabel")
        self.meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.meta_label.setMinimumWidth(dp(160))

        self.persona_btn = QtWidgets.QPushButton("Persona")
        self.persona_btn.setObjectName("SecondaryButton")
        self.persona_btn.setMinimumHeight(dp(MIN_TOUCH_TARGET))
        self.persona_btn.setMinimumWidth(dp(120))
        self.persona_btn.setAccessibleName(f"Persona selector for {model_id}")
        self.persona_btn.clicked.connect(self.personaClicked.emit)

        layout.addWidget(self.checkbox, 1)
        layout.addWidget(self.provider_badge, 0)
        layout.addWidget(self.meta_label, 0)
        layout.addWidget(self.persona_btn, 0)

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)

    def setPersonaText(self, text: str) -> None:
        display_text = text if text != "None" else "Persona"
        if len(display_text) > 12:
            display_text = f"{display_text[:10]}.."
        self.persona_btn.setText(display_text)
        self.persona_btn.setToolTip(text if text != "None" else "Select persona")

    def showPersonaButton(self, show: bool) -> None:
        self.persona_btn.setVisible(show)
