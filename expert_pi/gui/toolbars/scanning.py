from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class BlankerButton(QtWidgets.QPushButton):
    clicked = QtCore.Signal(str)

    def __init__(self, name, selected=False, name_off=None):
        if not selected and name_off is not None:
            super().__init__(name_off)
        else:
            super().__init__(name)
        self.names = [name, name_off]
        self.setProperty("class", "toolbarButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.state = "BeamOff"
        self.set_state("BeamOff")

    def set_state(self, value):
        self.state = value
        if self.state == "BeamOff":
            self.setStyleSheet(
                "background-color:black;color:white;border-color:black;"
            )  # need to redraw the selected property
            self.setToolTip("beam blanked")
        elif self.state == "BeamAcq":
            self.setStyleSheet(
                "background-color:qlineargradient(spread:repeat, x1:0, y1:0, x2:1, y2:0, stop: 0 #fff, stop: 0.7 #fff, stop: 0.8 #000, stop: 1 #000);color:black"
            )
            self.setToolTip("blanker driven by scanning recipe")
        elif self.state == "BeamOn":
            self.setStyleSheet("background-color:white;color:blank;")  # need to redraw the selected property
            self.setToolTip("beam on")

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.state in {"BeamOn", "BeamAcq"}:
                self.set_state("BeamOff")
            else:
                self.set_state("BeamOn")
            self.clicked.emit(self.state)
        else:
            super().mousePressEvent(e)


class Scanning(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("SCANNING", images_dir + "section_icons/scanning.svg", panel_size)
        self.fov_spin = spin_box.SpinBoxWithUnits(20, [0, 1000], 1, "um")
        self.fov_spin.setToolTip("fov")

        self.pixel_time_spin = spin_box.SpinBoxWithUnits(0.5, [0.1, 100_000_000], 1, "us")
        self.pixel_time_spin.setToolTip("pixel time")

        self.rotation_spin = spin_box.SpinBoxWithUnits(0, [-180, 180], 1, "deg")
        self.rotation_spin.setToolTip("STEM rotation")

        self.size_combo = combo_box.ToolbarComboBox()
        for i in range(3, 14):
            self.size_combo.addItem(f"{2**i} px")
        self.size_combo.setCurrentIndex(10 - 3)
        self.size_combo.setToolTip("number of pixels of STEM image")

        control_options = {
            "start": ("", images_dir + "tools_icons/start.svg"),
            "1x": ("1x", None),
            "stop": ("", images_dir + "tools_icons/stop.svg"),
        }
        self.control_buttons = buttons.ToolbarMultiButton(control_options, default_option="stop")

        self.blanker_button = BlankerButton("blanker")

        self.off_axis_butt = buttons.ToolbarStateButton("off-axis")

        self.content_layout.addWidget(self.fov_spin, 0, 0, 1, 2)
        self.content_layout.addWidget(self.pixel_time_spin, 1, 0, 1, 1)
        self.content_layout.addWidget(self.size_combo, 1, 1, 1, 1)
        self.content_layout.addWidget(self.control_buttons, 3, 0, 1, 2)
        self.content_layout.addWidget(self.blanker_button, 4, 0, 1, 1)
        self.content_layout.addWidget(self.off_axis_butt, 4, 1, 1, 1)

        self.content_layout.addWidget(self.rotation_spin, 0, 2, 1, 1)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.rotation_spin.show()
        else:
            self.rotation_spin.hide()
