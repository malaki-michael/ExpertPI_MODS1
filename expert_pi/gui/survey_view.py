from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from expert_pi.gui.main_window import MainWindow


class SurveyView(QtCore.QObject):
    show_signal = QtCore.Signal()
    hide_signal = QtCore.Signal()

    def __init__(self, window: MainWindow):
        super().__init__()
        self.window = window

        self.left_toolbars = [
            window.optics.name,
            window.stem_tools.name,
            window.scanning.name,
            window.detectors.name,
            window.stem_4d.name,
            window.stem_adjustments.name,
        ]

        self.right_toolbars = [
            window.camera.name,
            window.diffraction_tools.name,
            window.diffraction.name,
            window.precession.name,
            window.xray.name,
        ]

        self.central_default_layout = {
            "vertical": False,
            "position": 0.618,
            "first": window.image_view.name,
            "second": {
                "vertical": True,
                "position": 0.618,
                "first": window.diffraction_view.name,
                "second": window.spectrum_view.name,
            },
        }

    def show(self):
        self.window.central_layout.set_toolbar_item_names(self.left_toolbars, type="left")
        self.window.central_layout.set_toolbar_item_names(self.right_toolbars, type="right")
        self.window.central_layout.set_central_layout(self.central_default_layout)

        self.show_signal.emit()

    def hide(self):
        self.hide_signal.emit()
