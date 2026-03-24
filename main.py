from __future__ import annotations

import asyncio
import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

import qasync

from qml_bridge import QmlBridge


ROOT = Path(__file__).resolve().parent


def runtime_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return ROOT


QML_PATH = runtime_root() / "gui" / "main.qml"
ICON_PATH = runtime_root() / "PolyCouncilIco.ico"


def configure_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PolyCouncil.App")
    except Exception:
        pass


def app_icon() -> QIcon:
    return QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon()


def create_runtime() -> tuple[QGuiApplication, qasync.QEventLoop, QQmlApplicationEngine, QmlBridge]:
    configure_windows_app_id()
    QQuickStyle.setStyle("Material")
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    bridge = QmlBridge()
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    context.setContextProperty("polyBridge", bridge)
    context.setContextProperty("polyBackend", bridge)
    engine.load(str(QML_PATH))

    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML: {QML_PATH}")
    for root in engine.rootObjects():
        if hasattr(root, "setIcon") and not icon.isNull():
            root.setIcon(icon)

    return app, loop, engine, bridge


def main() -> int:
    app, loop, _engine, _bridge = create_runtime()
    app.lastWindowClosed.connect(app.quit)
    app.aboutToQuit.connect(loop.stop)
    with loop:
        loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
