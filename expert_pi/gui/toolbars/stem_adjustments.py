import json

from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class StemAdjustments(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("STEM ADJUST", images_dir + "section_icons/adjustments.svg", panel_size)
        # self.focus = QtWidgets.QLabel("FOCUS  +200nm")
        self.stigmators = QtWidgets.QLabel("STIGMATORS +20mA  +1mA")

        stem_shift_options = {"combined": ("COMB", ""), "electronic": ("EL", ""), "mechanic": ("MECH", "")}
        self.stem_shift_type = buttons.ToolbarMultiButton(stem_shift_options, default_option="combined")
        self.stem_shift_type.buttons["combined"].setToolTip(
            "Combined Electronic and Mechanical shift in image window (double click)"
        )
        self.stem_shift_type.buttons["electronic"].setToolTip("Electronic shift only (double click)")
        self.stem_shift_type.buttons["mechanic"].setToolTip("Mechanical shift only (double click)")

        self.content_layout.addWidget(self.stem_shift_type, 0, 0, 1, 2)

        self.stage_backlash = QtWidgets.QCheckBox("XYZ backlash")
        self.stage_backlash.setToolTip("Use backlash on stage XYZ movements")
        self.stage_x = spin_box.SpinBoxWithUnits(0, [-2000, 2000], 1, "um", send_immediately=True)
        self.stage_y = spin_box.SpinBoxWithUnits(0, [-2000, 2000], 1, "um", send_immediately=True)

        self.electronic_type = combo_box.ToolbarComboBox()
        self.electronic_type.addItem("scan")
        self.electronic_type.addItem("precession")
        self.electronic_type.setCurrentIndex(0)
        self.shift_x = spin_box.SpinBoxWithUnits(0, [-10000, 10000], 1, "nm", send_immediately=True)
        self.shift_y = spin_box.SpinBoxWithUnits(0, [-10000, 10000], 1, "nm", send_immediately=True)

        self.mechanic_label = QtWidgets.QLabel("Mechanic XY:")
        self.mechanic_stop = buttons.ToolbarPushButton("Stop")
        self.content_layout.addWidget(self.mechanic_label, 1, 0, 1, 1)
        self.content_layout.addWidget(self.mechanic_stop, 1, 1, 1, 1)
        self.content_layout.addWidget(self.stage_x, 2, 0, 1, 1)
        self.content_layout.addWidget(self.stage_y, 2, 1, 1, 1)
        self.content_layout.addWidget(self.stage_backlash, 2, 2, 1, 2)

        self.electronic_label = QtWidgets.QLabel("Electronic XY:")

        self.content_layout.addWidget(self.electronic_label, 3, 0, 1, 1)
        self.content_layout.addWidget(self.electronic_type, 3, 1, 1, 1)
        self.content_layout.addWidget(self.shift_x, 4, 0, 1, 1)
        self.content_layout.addWidget(self.shift_y, 4, 1, 1, 1)

        # ----------------------------

        self.tilt_wobbler = buttons.ToolbarPushButton("Tilt wobbler", selectable=True)
        self.content_layout.addWidget(self.tilt_wobbler, 5, 0, 1, 1)
        self.tilt_wobbler_direction = combo_box.ToolbarComboBox()
        self.tilt_wobbler_direction.addItem("X")
        self.tilt_wobbler_direction.addItem("Y")
        self.tilt_wobbler_direction.setCurrentIndex(0)
        self.content_layout.addWidget(self.tilt_wobbler_direction, 5, 2, 1, 1)
        self.tilt_wobbler_angle = spin_box.SpinBoxWithUnits(10, [-200, 200], 1, "mrad", send_immediately=True)
        self.content_layout.addWidget(self.tilt_wobbler_angle, 5, 3, 1, 1)

        self.tilt_wobbler_set = buttons.ToolbarPushButton("")
        self.tilt_wobbler_set.setEnabled(False)
        self.tilt_wobbler_set.set_value = 0
        self.tilt_wobbler_set.clicked.connect(self.set_z_measured)
        self.content_layout.addWidget(self.tilt_wobbler_set, 5, 1, 1, 1)

        self.content_layout.addWidget(QtWidgets.QLabel("Mechanics Z:"), 6, 0, 1, 2)

        self.stage_z = spin_box.SpinBoxWithUnits(0, [-2000, 2000], 1, "um", send_immediately=True)

        self.z_set_eucentric = buttons.ToolbarPushButton("set")
        self.z_set_eucentric.setToolTip("Use this as eucentric z - position")

        self.content_layout.addWidget(self.stage_z, 7, 0, 1, 1)
        self.content_layout.addWidget(self.z_set_eucentric, 7, 1, 1, 1)

        # ----------------------------

        focus_label = QtWidgets.QLabel("Focus:")
        self.content_layout.addWidget(focus_label, 8, 0, 1, 1)
        focus_label.setToolTip("ctrl+zoom on image and diffraction view")

        self.focus_auto = buttons.ToolbarPushButton("auto", selectable=True)
        self.content_layout.addWidget(self.focus_auto, 8, 1, 1, 1)

        self.focus = spin_box.SpinBoxWithUnits(0, [-2000, 2000], 0.2, "um", send_immediately=True)
        self.focus.setToolTip("ctrl+zoom on image and diffraction view")
        self.content_layout.addWidget(self.focus, 9, 0, 1, 1)

        self.focus_type = combo_box.ToolbarComboBox()
        self.focus_type.addItem("C3")
        self.focus_type.addItem("OB")
        self.focus_type.setCurrentIndex(0)
        self.content_layout.addWidget(self.focus_type, 9, 1, 1, 1)

        # ----------------------------

        stigmator_label = QtWidgets.QLabel("Stigmators:")
        stigmator_label.setToolTip("ctrl+drag on diffraction view")
        self.content_layout.addWidget(stigmator_label, 10, 0, 1, 1)

        self.stigmators_auto = buttons.ToolbarPushButton("auto", selectable=True)
        self.content_layout.addWidget(self.stigmators_auto, 10, 1, 1, 1)

        self.stigmator_normal = spin_box.SpinBoxWithUnits(
            0, [-10000, 10000], 1, "mA", send_immediately=True
        )  # TODO units
        self.stigmator_skew = spin_box.SpinBoxWithUnits(0, [-10000, 10000], 1, "mA", send_immediately=True)
        self.stigmator_normal.setToolTip("ctrl+drag on diffraction view")
        self.stigmator_skew.setToolTip("ctrl+drag on diffraction view")

        self.content_layout.addWidget(self.stigmator_normal, 11, 0, 1, 1)
        self.content_layout.addWidget(self.stigmator_skew, 11, 1, 1, 1)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.stage_backlash.show()
            self.tilt_wobbler_direction.show()
            self.tilt_wobbler_angle.show()
        else:
            self.stage_backlash.hide()
            self.tilt_wobbler_direction.hide()
            self.tilt_wobbler_angle.hide()

    def stage_update(self, data):
        try:
            decoded = json.loads(data.decode())

            self.stage_x.update_read_signal.emit(decoded["x"])
            self.stage_y.update_read_signal.emit(decoded["y"])
            self.stage_z.update_read_signal.emit(decoded["z"])

            if "on_target" in decoded:
                for item in [self.stage_x, self.stage_y, self.stage_z]:
                    item.setProperty("busy", not decoded["on_target"])
                    item.update_style()
        except:
            for item in [self.stage_x, self.stage_y, self.stage_z]:
                item.setProperty("error", True)
                item.update_style()

    def set_z_measured(self):
        self.stage_z.emit_set_signal(self.stage_z.value() - self.tilt_wobbler_set.set_value)
        # self.tilt_wobbler_set.value = 0
        self.tilt_wobbler_set.setText("")
