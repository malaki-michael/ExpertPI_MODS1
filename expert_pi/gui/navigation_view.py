from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.gui import main_window

from PySide6 import QtCore


class NavigationView(QtCore.QObject):
    show_signal = QtCore.Signal()
    hide_signal = QtCore.Signal()

    def __init__(self, window: main_window.MainWindow):
        super().__init__()
        self.window = window

    def show(self):
        self.window.central_layout.set_toolbar_item_names(None, type="left")
        self.window.central_layout.set_toolbar_item_names(None, type="right")
        self.window.central_layout.set_central_layout("navigation")

        self.show_signal.emit()

    def hide(self):
        self.hide_signal.emit()
