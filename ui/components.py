"""
PolyCouncil UI Components
=========================
Reusable widgets extracted from the monolithic council.py:
 - CollapsibleGroupBox: click-to-collapse with smooth animation
 - ModelCard: card-style model row with provider badge & capability icons
 - ToastNotification: slide-in overlay for status messages
 - EnhancedPromptEditor: improved prompt editor with char count
 - AnimatedStatusBar: footer with animated status badges
 - OnboardingOverlay: first-run transparent walkthrough
 - KeyboardShortcutOverlay: Ctrl+? help panel
"""

from __future__ import annotations

from typing import Optional, List

from PySide6 import QtCore, QtGui, QtWidgets

from ui.animations import FadeIn, FadeOut, SmoothCollapse


# ---------------------------------------------------------------------------
# CollapsibleGroupBox
# ---------------------------------------------------------------------------
class CollapsibleGroupBox(QtWidgets.QWidget):
    """
    A group box whose content area can be collapsed/expanded by clicking
    the header. Uses smooth height animation.
    """

    toggled = QtCore.Signal(bool)  # emits True when expanded

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, start_collapsed: bool = False):
        super().__init__(parent)
        self._expanded = not start_collapsed
        self._title = title
        self._anim: Optional[QtCore.QPropertyAnimation] = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Container frame styled like QGroupBox
        self._frame = QtWidgets.QFrame()
        self._frame.setObjectName("CollapsibleFrame")
        frame_layout = QtWidgets.QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # Header (clickable)
        self._header = QtWidgets.QFrame()
        self._header.setObjectName("CollapsibleHeader")
        self._header.setCursor(QtCore.Qt.PointingHandCursor)
        self._header.setFixedHeight(44)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(14, 0, 14, 0)
        header_layout.setSpacing(10)

        self._arrow_label = QtWidgets.QLabel("▾" if self._expanded else "▸")
        self._arrow_label.setFixedWidth(16)
        self._arrow_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setStyleSheet("font-weight: 700; font-size: 13px;")

        header_layout.addWidget(self._arrow_label)
        header_layout.addWidget(self._title_label, 1)

        # Content container
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(14, 10, 14, 14)
        self._content_layout.setSpacing(8)

        frame_layout.addWidget(self._header)
        frame_layout.addWidget(self._content)
        outer.addWidget(self._frame)

        # Install click handler on header
        self._header.mousePressEvent = self._on_header_click

        if start_collapsed:
            self._content.setMaximumHeight(0)
            self._content.hide()

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        """Return the layout to add child widgets to."""
        return self._content_layout

    def add_widget(self, widget: QtWidgets.QWidget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QtWidgets.QLayout):
        self._content_layout.addLayout(layout)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, animate: bool = True):
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._arrow_label.setText("▾" if expanded else "▸")
        if animate:
            if expanded:
                SmoothCollapse.expand(self._content, duration_ms=200)
            else:
                SmoothCollapse.collapse(self._content, duration_ms=200)
        else:
            if expanded:
                self._content.show()
                self._content.setMaximumHeight(16777215)
            else:
                self._content.hide()
                self._content.setMaximumHeight(0)
        self.toggled.emit(expanded)

    def _on_header_click(self, event):
        self.set_expanded(not self._expanded)

    def setStyleSheet(self, stylesheet: str):
        """Proxy to the inner frame for theming."""
        self._frame.setStyleSheet(stylesheet)


# ---------------------------------------------------------------------------
# ToastNotification
# ---------------------------------------------------------------------------
class ToastNotification(QtWidgets.QFrame):
    """
    A floating overlay notification that slides in from the bottom-right,
    stays for a few seconds, then fades out.
    """

    def __init__(self, parent: QtWidgets.QWidget, message: str, tone: str = "info", duration_ms: int = 3500):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Tone colors
        tone_colors = {
            "info": ("#ddf4ff", "#0969da", "#54aeff"),
            "success": ("#dafbe1", "#1a7f37", "#3fb950"),
            "warn": ("#fff8c5", "#9a6700", "#d29922"),
            "error": ("#ffebe9", "#d1242f", "#f85149"),
        }
        bg, fg, border = tone_colors.get(tone, tone_colors["info"])

        self.setStyleSheet(f"""
            QFrame#Toast {{
                background: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Tone icon
        icons = {"info": "ℹ️", "success": "✓", "warn": "⚠", "error": "✕"}
        icon_label = QtWidgets.QLabel(icons.get(tone, "ℹ️"))
        icon_label.setStyleSheet(f"font-size: 16px; color: {fg}; background: transparent; border: none;")
        layout.addWidget(icon_label)

        msg_label = QtWidgets.QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color: {fg}; background: transparent; border: none;")
        layout.addWidget(msg_label, 1)

        self.adjustSize()
        self.setFixedWidth(min(400, parent.width() - 40))

        self._duration_ms = duration_ms

    def show_toast(self):
        """Position and display the toast."""
        parent = self.parentWidget()
        if not parent:
            return

        # Position at bottom-right of parent
        x = parent.width() - self.width() - 20
        y = parent.height() - self.height() - 20
        self.move(x, y)
        self.show()
        self.raise_()

        FadeIn.run(self, duration_ms=200)

        # Auto-dismiss after duration
        QtCore.QTimer.singleShot(self._duration_ms, self._dismiss)

    def _dismiss(self):
        anim = FadeOut.run(self, duration_ms=300, hide_on_finish=True)
        if anim:
            anim.finished.connect(self.deleteLater)


# ---------------------------------------------------------------------------
# EnhancedPromptEditor
# ---------------------------------------------------------------------------
class EnhancedPromptEditor(QtWidgets.QWidget):
    """
    Enhanced prompt editor with character count, attachment indicator,
    and mode-aware placeholder.
    """
    submitRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._editor = _PromptTextEdit(self)
        self._editor.submitRequested.connect(self.submitRequested.emit)
        self._editor.setAccessibleName("Prompt editor")
        self._editor.setAccessibleDescription("Enter a council prompt. Press Enter to run and Shift plus Enter for a new line.")
        layout.addWidget(self._editor)

        # Footer row with char count and hints
        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(4, 0, 4, 0)
        footer.setSpacing(8)

        self._hint_label = QtWidgets.QLabel("Enter sends · Shift+Enter for new line · Ctrl+Shift+A selects all models")
        self._hint_label.setObjectName("HintLabel")

        self._char_count = QtWidgets.QLabel("0")
        self._char_count.setObjectName("HintLabel")
        self._char_count.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        footer.addWidget(self._hint_label, 1)
        footer.addWidget(self._char_count)
        layout.addLayout(footer)

        self._editor.textChanged.connect(self._update_char_count)

    def toPlainText(self) -> str:
        return self._editor.toPlainText()

    def setPlainText(self, text: str):
        self._editor.setPlainText(text)

    def clear(self):
        self._editor.clear()

    def setPlaceholderText(self, text: str):
        self._editor.setPlaceholderText(text)

    def setFocus(self):
        self._editor.setFocus()

    def _update_char_count(self):
        count = len(self._editor.toPlainText())
        self._char_count.setText(f"{count:,}")


class _PromptTextEdit(QtWidgets.QPlainTextEdit):
    """Inner text edit with Enter-to-submit and auto-resize."""
    submitRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._min_height = 74
        self._max_height = 180
        self.setTabChangesFocus(True)
        self.setPlaceholderText("Ask the council a question. Press Enter to send, Shift+Enter for a new line.")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.document().documentLayout().documentSizeChanged.connect(self._sync_height)
        self._sync_height()

    def _sync_height(self, *_args):
        doc_height = self.document().size().height()
        frame = self.frameWidth() * 2
        padding = 14
        target = int(doc_height + frame + padding)
        target = max(self._min_height, min(target, self._max_height))
        self.setFixedHeight(target)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if not (event.modifiers() & QtCore.Qt.ShiftModifier):
                self.submitRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_height)

    def clear(self):
        super().clear()
        self._sync_height()


# ---------------------------------------------------------------------------
# AnimatedStatusBar
# ---------------------------------------------------------------------------
class AnimatedStatusBar(QtWidgets.QWidget):
    """Footer status bar with animated badge, message, and progress indicator."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximum(0)
        self.progress.setFixedWidth(120)
        self.progress.setVisible(False)

        self.badge = QtWidgets.QLabel("Idle")
        self.badge.setObjectName("StatusBadge")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)

        self.message = QtWidgets.QLabel("Ready.")
        self.message.setObjectName("StatusText")
        self.message.setWordWrap(True)

        self.footer_link = QtWidgets.QLabel(
            '<a href="https://github.com/TrentPierce" style="color: inherit; text-decoration: none;">'
            'Trent Pierce · GitHub</a>'
        )
        self.footer_link.setOpenExternalLinks(True)
        self.footer_link.setObjectName("HintLabel")

        layout.addWidget(self.progress, 0)
        layout.addWidget(self.badge, 0)
        layout.addWidget(self.message, 1)
        layout.addWidget(self.footer_link, 0)

    def set_status(self, text: str):
        self.message.setText(text)
        lowered = text.lower()
        if any(w in lowered for w in ("error", "failed")):
            tone, badge_text = "error", "Error"
        elif any(w in lowered for w in ("warning", "select", "unavailable", "no ")) and not self.progress.isVisible():
            tone, badge_text = "warn", "Attention"
        elif self.progress.isVisible():
            tone, badge_text = "busy", "Working"
        elif any(w in lowered for w in (
            "done", "ready", "complete", "saved", "added", "found", "loaded", "updated", "cleared"
        )):
            tone, badge_text = "success", "Ready"
        else:
            tone, badge_text = "neutral", "Info"
        self.badge.setText(badge_text)
        self._set_badge_tone(tone)

    def set_busy(self, on: bool):
        self.progress.setVisible(on)
        self.progress.setMaximum(0 if on else 1)
        if on:
            self.badge.setText("Working")
            self._set_badge_tone("busy")

    def _set_badge_tone(self, tone: str):
        self.badge.setProperty("tone", tone)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.badge.update()


# ---------------------------------------------------------------------------
# OnboardingOverlay
# ---------------------------------------------------------------------------
class OnboardingOverlay(QtWidgets.QWidget):
    """
    Semi-transparent first-run overlay with step-by-step guidance.
    Shown once on first launch, then suppressed via settings.
    """
    dismissed = QtCore.Signal()

    STEPS = [
        {
            "title": "Welcome to PolyCouncil! 👋",
            "body": "PolyCouncil lets you run multiple AI models in parallel, "
                    "have them evaluate each other, and pick the best answer through consensus.",
        },
        {
            "title": "Step 1: Connect a Provider",
            "body": "Choose a provider (LM Studio, Ollama, or a hosted API) from the "
                    "Provider Connection panel, then click Load Models.",
        },
        {
            "title": "Step 2: Select Models",
            "body": "Check the models you want to compare. You can mix models from "
                    "different providers and assign personas for varied perspectives.",
        },
        {
            "title": "Step 3: Ask a Question",
            "body": "Type your prompt in the composer at the bottom and press Enter. "
                    "The council will deliberate and show you the winning answer!",
        },
    ]

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self._step = 0

        # Semi-transparent backdrop
        self.setStyleSheet("background: rgba(0, 0, 0, 160);")

        # Center card
        self._card = QtWidgets.QFrame(self)
        self._card.setStyleSheet("""
            QFrame {
                background: #1c2128;
                border: 1px solid #30363d;
                border-radius: 20px;
                padding: 0;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(self._card)
        card_layout.setContentsMargins(32, 28, 32, 24)
        card_layout.setSpacing(16)

        self._step_indicator = QtWidgets.QLabel()
        self._step_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self._step_indicator.setStyleSheet("color: #6e7681; font-size: 12px; background: transparent; border: none;")

        self._title_label = QtWidgets.QLabel()
        self._title_label.setStyleSheet(
            "color: #e6edf3; font-size: 22px; font-weight: 800; background: transparent; border: none;"
        )
        self._title_label.setWordWrap(True)
        self._title_label.setAlignment(QtCore.Qt.AlignCenter)

        self._body_label = QtWidgets.QLabel()
        self._body_label.setStyleSheet(
            "color: #b1bac4; font-size: 14px; line-height: 1.6; background: transparent; border: none;"
        )
        self._body_label.setWordWrap(True)
        self._body_label.setAlignment(QtCore.Qt.AlignCenter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        self._skip_btn = QtWidgets.QPushButton("Skip")
        self._skip_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #6e7681; border: none; font-size: 13px; }
            QPushButton:hover { color: #b1bac4; }
        """)
        self._next_btn = QtWidgets.QPushButton("Next →")
        self._next_btn.setStyleSheet("""
            QPushButton {
                background: #4493f8; color: white; border: none;
                border-radius: 10px; padding: 10px 28px; font-size: 14px; font-weight: 700;
            }
            QPushButton:hover { background: #539bf5; }
        """)
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

    def _render_step(self):
        step = self.STEPS[self._step]
        self._step_indicator.setText(f"Step {self._step + 1} of {len(self.STEPS)}")
        self._title_label.setText(step["title"])
        self._body_label.setText(step["body"])
        if self._step == len(self.STEPS) - 1:
            self._next_btn.setText("Get Started! 🚀")
        else:
            self._next_btn.setText("Next →")

    def _advance(self):
        self._step += 1
        if self._step >= len(self.STEPS):
            self._dismiss()
        else:
            self._render_step()

    def _dismiss(self):
        FadeOut.run(self, duration_ms=250, hide_on_finish=True)
        self.dismissed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Center the card
        card_w = min(520, self.width() - 60)
        card_h = min(380, self.height() - 80)
        self._card.setFixedSize(card_w, card_h)
        self._card.move(
            (self.width() - card_w) // 2,
            (self.height() - card_h) // 2,
        )

    def paintEvent(self, event):
        # Draw semi-transparent background
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 160))
        painter.end()


# ---------------------------------------------------------------------------
# KeyboardShortcutOverlay
# ---------------------------------------------------------------------------
class KeyboardShortcutOverlay(QtWidgets.QWidget):
    """Shows all keyboard shortcuts in a floating overlay."""

    SHORTCUTS = [
        ("Enter", "Send prompt to council"),
        ("Shift+Enter", "New line in prompt"),
        ("Ctrl+Enter", "Send from anywhere"),
        ("Ctrl+Shift+A", "Select all models"),
        ("Ctrl+R", "Reload models"),
        ("Ctrl+L", "Focus prompt editor"),
        ("Ctrl+F", "Focus model filter"),
        ("Escape", "Stop current operation"),
        ("Ctrl+?", "Toggle this help"),
    ]

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.hide()

        # Card
        self._card = QtWidgets.QFrame(self)
        self._card.setStyleSheet("""
            QFrame {
                background: #1c2128;
                border: 1px solid #30363d;
                border-radius: 20px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(self._card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(6)

        title = QtWidgets.QLabel("⌨ Keyboard Shortcuts")
        title.setStyleSheet(
            "color: #e6edf3; font-size: 18px; font-weight: 800; background: transparent; border: none;"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(title)
        card_layout.addSpacing(12)

        for key, desc in self.SHORTCUTS:
            row = QtWidgets.QHBoxLayout()
            key_label = QtWidgets.QLabel(key)
            key_label.setStyleSheet(
                "background: #30363d; color: #e6edf3; padding: 4px 10px; "
                "border-radius: 6px; font-family: monospace; font-weight: 600; "
                "font-size: 12px; border: 1px solid #484f58;"
            )
            key_label.setFixedWidth(140)
            key_label.setAlignment(QtCore.Qt.AlignCenter)
            desc_label = QtWidgets.QLabel(desc)
            desc_label.setStyleSheet("color: #b1bac4; font-size: 13px; background: transparent; border: none;")
            row.addWidget(key_label)
            row.addWidget(desc_label, 1)
            card_layout.addLayout(row)

        card_layout.addStretch(1)

        close_hint = QtWidgets.QLabel("Press Escape or Ctrl+? to close")
        close_hint.setStyleSheet("color: #6e7681; font-size: 11px; background: transparent; border: none;")
        close_hint.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(close_hint)

    def toggle(self):
        if self.isVisible():
            FadeOut.run(self, duration_ms=200, hide_on_finish=True)
        else:
            self.setGeometry(self.parentWidget().rect())
            self._center_card()
            self.show()
            self.raise_()
            FadeIn.run(self, duration_ms=200)

    def _center_card(self):
        card_w = min(440, self.width() - 60)
        card_h = min(420, self.height() - 80)
        self._card.setFixedSize(card_w, card_h)
        self._card.move(
            (self.width() - card_w) // 2,
            (self.height() - card_h) // 2,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_card()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 140))
        painter.end()

    def mousePressEvent(self, event):
        # Clicking outside the card dismisses the overlay
        if not self._card.geometry().contains(event.pos()):
            self.toggle()
        else:
            super().mousePressEvent(event)

# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------
class ModelCard(QtWidgets.QWidget):
    """
    Card-style model row with provider badge, capability icons, and persona selector.
    """
    toggled = QtCore.Signal(bool)
    personaClicked = QtCore.Signal()

    def __init__(self, model_id: str, provider_name: str, meta_text: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModelRow")
        self.setMinimumHeight(44)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        self.checkbox = QtWidgets.QCheckBox(model_id)
        self.checkbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.checkbox.setAccessibleName(f"Select model {model_id}")
        self.checkbox.setAccessibleDescription(f"Toggle participation for model {model_id}.")
        self.checkbox.toggled.connect(self.toggled.emit)

        # Provider Badge
        self.provider_badge = QtWidgets.QLabel(provider_name)
        self.provider_badge.setObjectName("InfoBadge")
        
        # Meta Label (for capabilities/latency)
        self.meta_label = QtWidgets.QLabel(meta_text)
        self.meta_label.setObjectName("HintLabel")
        self.meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.meta_label.setMinimumWidth(150)

        # Persona Button
        self.persona_btn = QtWidgets.QPushButton("Persona")
        self.persona_btn.setFixedWidth(120)
        self.persona_btn.setFixedHeight(28)
        self.persona_btn.setAccessibleName(f"Persona selector for {model_id}")
        self.persona_btn.clicked.connect(self.personaClicked.emit)

        layout.addWidget(self.checkbox, 1)
        layout.addWidget(self.provider_badge, 0)
        layout.addWidget(self.meta_label, 0)
        layout.addWidget(self.persona_btn, 0)

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool):
        self.checkbox.setChecked(checked)

    def setPersonaText(self, text: str):
        display_text = text if text != "None" else "Persona"
        if len(display_text) > 12:
            display_text = display_text[:10] + ".."
        self.persona_btn.setText(display_text)
        self.persona_btn.setToolTip(text if text != "None" else "Select persona")

    def showPersonaButton(self, show: bool):
        self.persona_btn.setVisible(show)

