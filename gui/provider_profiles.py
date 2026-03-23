from __future__ import annotations

from typing import Callable

from PySide6 import QtWidgets


def clear_layout(layout: QtWidgets.QLayout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
        del item


def build_provider_profile_row(
    *,
    summary: str,
    profile_id: str,
    on_use: Callable[[str], None],
    on_load: Callable[[str], None],
    on_remove: Callable[[str], None],
) -> QtWidgets.QWidget:
    row = QtWidgets.QWidget()
    row_layout = QtWidgets.QHBoxLayout(row)
    row_layout.setContentsMargins(2, 2, 2, 2)
    row_layout.setSpacing(6)

    label = QtWidgets.QLabel(summary)
    use_btn = QtWidgets.QPushButton("Use")
    load_btn = QtWidgets.QPushButton("Load")
    remove_btn = QtWidgets.QPushButton("Remove")

    use_btn.clicked.connect(lambda _checked=False, pid=profile_id: on_use(pid))
    load_btn.clicked.connect(lambda _checked=False, pid=profile_id: on_load(pid))
    remove_btn.clicked.connect(lambda _checked=False, pid=profile_id: on_remove(pid))

    row_layout.addWidget(label, 1)
    row_layout.addWidget(use_btn)
    row_layout.addWidget(load_btn)
    row_layout.addWidget(remove_btn)
    return row
