"""
PolyCouncil Animations
======================
Reusable animation helpers for smooth UI transitions.
Built on QPropertyAnimation to stay within PySide6.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

REDUCE_MOTION = False


def set_reduce_motion(enabled: bool):
    global REDUCE_MOTION
    REDUCE_MOTION = bool(enabled)


class FadeIn:
    """Fade a widget in from transparent to fully opaque."""

    @staticmethod
    def run(widget: QtWidgets.QWidget, duration_ms: int = 300, start_val: float = 0.0, end_val: float = 1.0):
        if REDUCE_MOTION:
            widget.show()
            return None
        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(start_val)
        anim.setEndValue(end_val)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        return anim


class FadeOut:
    """Fade a widget out and optionally hide it on finish."""

    @staticmethod
    def run(widget: QtWidgets.QWidget, duration_ms: int = 250, hide_on_finish: bool = True):
        if REDUCE_MOTION:
            if hide_on_finish:
                widget.hide()
            return None
        effect = widget.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
        if hide_on_finish:
            anim.finished.connect(widget.hide)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        return anim


class SlideIn:
    """Slide a widget in from a direction."""

    @staticmethod
    def run(
        widget: QtWidgets.QWidget,
        direction: str = "right",
        duration_ms: int = 350,
        distance: int = 80,
    ):
        if REDUCE_MOTION:
            widget.show()
            return None
        start_pos = widget.pos()
        if direction == "right":
            offset_pos = QtCore.QPoint(start_pos.x() + distance, start_pos.y())
        elif direction == "left":
            offset_pos = QtCore.QPoint(start_pos.x() - distance, start_pos.y())
        elif direction == "up":
            offset_pos = QtCore.QPoint(start_pos.x(), start_pos.y() - distance)
        else:  # down
            offset_pos = QtCore.QPoint(start_pos.x(), start_pos.y() + distance)

        widget.move(offset_pos)
        widget.show()
        anim = QtCore.QPropertyAnimation(widget, b"pos", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(offset_pos)
        anim.setEndValue(start_pos)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

        # Also fade in
        FadeIn.run(widget, duration_ms)
        return anim


class PulseEffect:
    """Subtle pulsing glow on a widget (e.g. a progress indicator)."""

    @staticmethod
    def run(widget: QtWidgets.QWidget, duration_ms: int = 1200):
        if REDUCE_MOTION:
            return None
        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.6)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        anim.setLoopCount(-1)  # infinite loop
        anim.start()
        return anim


class SmoothCollapse:
    """Animate a widget's maximumHeight for smooth collapse/expand."""

    @staticmethod
    def collapse(widget: QtWidgets.QWidget, duration_ms: int = 250):
        if REDUCE_MOTION:
            widget.hide()
            widget.setMaximumHeight(0)
            return None
        anim = QtCore.QPropertyAnimation(widget, b"maximumHeight", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(widget.sizeHint().height())
        anim.setEndValue(0)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        anim.finished.connect(widget.hide)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        return anim

    @staticmethod
    def expand(widget: QtWidgets.QWidget, target_height: int = 0, duration_ms: int = 250):
        if REDUCE_MOTION:
            widget.show()
            widget.setMaximumHeight(16777215)
            return None
        if target_height <= 0:
            target_height = widget.sizeHint().height()
        widget.setMaximumHeight(0)
        widget.show()
        anim = QtCore.QPropertyAnimation(widget, b"maximumHeight", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(0)
        anim.setEndValue(target_height)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        # Remove constraint when done so the widget can resize naturally
        anim.finished.connect(lambda: widget.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        return anim
