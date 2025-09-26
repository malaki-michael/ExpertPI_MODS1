from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.gui.main_window import MainWindow
    from expert_pi.app import app
    from expert_pi.app.states import states_holder

from PySide6 import QtWidgets


class MainView:
    def __init__(
        self, window: MainWindow, controller: app.MainApp, state: states_holder.StatesHolder
    ):
        self.window = window
        self.controller = controller
        self.state = state

        self.test_widget = QtWidgets.QLabel("TOMOGRAPHY")

        self.window.central_layout.central.add_item("tomo", self.test_widget)
        # self.central_layout.right_toolbars.add_toolbar("customToolbar",CcustomToolbar)

    # need to be implemented
    def show(self):
        self.window.central_layout.set_toolbar_item_names(None, type="left")
        self.window.central_layout.set_toolbar_item_names(None, type="right")
        self.window.central_layout.set_central_layout("tomo")

    # need to be implemented
    def hide():
        # optional control at going out from measurement
        pass
