import numpy as np
from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class StateButton(buttons.ToolbarPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.last_data = None

    def change_state(self, text: str, selected: bool, busy: bool, error: bool):
        self.setText(text)
        self.setProperty("selected", selected)
        self.setProperty("busy", busy)
        self.setProperty("error", error)
        self.setStyleSheet(self.styleSheet())


class Optics(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("OPTICS", images_dir + "section_icons/optics.svg", panel_size)
        self.energy = buttons.ToolbarPushButton("100 keV")
        self.energy.setToolTip("Beam Energy")
        self.state_button = StateButton("STANDBY")
        self.state_button.setToolTip("System State")

        self.mode = combo_box.ToolbarComboBox()
        self.mode.setToolTip("Optical Mode")

        self.current = spin_box.SpinBoxWithUnits(1, [0.001, 50], np.nan, "nA", send_immediately=False)
        self.current.setValue(1.0012)
        self.current.setToolTip("Beam current")

        self.angle = spin_box.SpinBoxWithUnits(9, [0.1, 20], np.nan, "mrad", send_immediately=False)
        self.angle.setValue(9)
        self.angle.setToolTip("Illumination angle (mrad) ")

        self.diameter = spin_box.SpinBoxWithUnits(1, [0.01, 100000], np.nan, "nm", send_immediately=False)
        self.diameter.setValue(9)
        self.diameter.setToolTip("d50 - theoretical spot size")

        self.keep_adjustment = QtWidgets.QCheckBox("keep adjust")
        self.keep_adjustment.setChecked(True)

        self.content_layout.addWidget(self.energy, 0, 0, 1, 2)
        self.content_layout.addWidget(self.mode, 1, 0, 1, 2)
        self.content_layout.addWidget(self.state_button, 2, 0, 1, 2)
        self.content_layout.addWidget(self.current, 3, 0, 1, 2)
        self.widget = self.content_layout.addWidget(self.angle, 4, 0, 1, 1)
        self.widget = self.content_layout.addWidget(self.diameter, 4, 1, 1, 1)

        self.content_layout.addWidget(self.keep_adjustment, 4, 2, 1, 2)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.keep_adjustment.show()
        else:
            self.keep_adjustment.hide()

    def set_buttons_state(self, current: bool, angle: bool, diameter: bool):
        self.current.setEnabled(current)
        self.angle.setEnabled(angle)
        self.diameter.setEnabled(diameter)

    def set_buttons_tool_tip(self, angle: str, diameter: str):
        self.angle.setToolTip(angle)
        self.diameter.setToolTip(diameter)

    def set_read_sent_values(self, current: float, angle: float, diameter: float):
        self.current.set_read_sent_value(current)
        self.angle.set_read_sent_value(angle)
        self.diameter.set_read_sent_value(diameter)
