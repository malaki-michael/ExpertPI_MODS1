import json

from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base
from expert_pi.gui.tools import tilt_axes_tuner


class Diffraction(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__(
            "DIFFRACTION ADJUST",
            images_dir + "section_icons/diffraction.svg",
            expand_type="right",
            panel_size=panel_size,
        )
        stem_tilt_options = {
            "zoom": ("ZOOM", ""),
            "electronic": ("EL", ""),
            "mechanic": ("MECH", ""),
            "aperture": ("AP", ""),
        }
        self.stem_tilt_type = buttons.ToolbarMultiButton(stem_tilt_options, default_option="zoom")
        self.stem_tilt_type.buttons["zoom"].setToolTip("Zooming in diffraction view only (double click)")
        self.stem_tilt_type.buttons["electronic"].setToolTip(
            "Illumination tilt (mouse drag) + projectionn tilt (double click)"
        )
        self.stem_tilt_type.buttons["mechanic"].setToolTip("Mechanical tilt (double click)")
        self.stem_tilt_type.buttons["aperture"].setToolTip("Electronic aperture shift (double click)")

        self.content_layout.addWidget(self.stem_tilt_type, 0, 0 + 2, 1, 2)

        self.tilt_label = QtWidgets.QLabel("Electronic tilts:")

        self.tilt_type = combo_box.ToolbarComboBox()
        self.tilt_type.addItem("illumination")
        self.tilt_type.addItem("projection")
        # self.tilt_type.addItem("aperture")
        self.tilt_type.setCurrentIndex(0)

        self.tilt_x = spin_box.SpinBoxWithUnits(0, [-1000, 1000], 1, "mrad", send_immediately=True)
        self.tilt_y = spin_box.SpinBoxWithUnits(0, [-1000, 1000], 1, "mrad", send_immediately=True)

        self.stage_label = QtWidgets.QLabel("Stage tilts:")
        self.stage_alpha = spin_box.SpinBoxWithUnits(0, [-360, 360], 1, "deg", send_immediately=True)
        self.stage_beta = spin_box.SpinBoxWithUnits(0, [-360, 360], 1, "deg", send_immediately=True)

        self.stage_backlash = QtWidgets.QCheckBox("backlash")
        self.stage_backlash.setChecked(True)
        self.stage_backlash.setToolTip("Use backlash on stage tilt movements")

        self.show_tilt_axis_tuner = buttons.ToolbarPushButton("tilt tuner", selectable=True)
        self.tilt_axis_tuner = tilt_axes_tuner.TiltAxisTuner()
        self.tilt_axis_tuner.hide()

        self.content_layout.addWidget(self.tilt_label, 1, 0 + 2, 1, 1)
        self.content_layout.addWidget(self.tilt_type, 1, 1 + 2, 1, 1)

        self.content_layout.addWidget(self.tilt_x, 2, 0 + 2, 1, 1)
        self.content_layout.addWidget(self.tilt_y, 2, 1 + 2, 1, 1)

        self.content_layout.addWidget(self.stage_label, 3, 0 + 2, 1, 2)
        self.content_layout.addWidget(self.show_tilt_axis_tuner, 3, 0, 1, 2)

        self.content_layout.addWidget(self.stage_backlash, 4, 0, 1, 2)
        self.content_layout.addWidget(self.stage_alpha, 4, 0 + 2, 1, 1)
        self.content_layout.addWidget(self.stage_beta, 4, 1 + 2, 1, 1)

        self.show_tilt_axis_tuner.clicked.connect(self.show_tuner)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.stage_backlash.show()
            self.show_tilt_axis_tuner.show()
        else:
            self.stage_backlash.hide()
            self.show_tilt_axis_tuner.hide()

    def show_tuner(self):
        if self.show_tilt_axis_tuner.selected:
            self.tilt_axis_tuner.show()
        else:
            self.tilt_axis_tuner.hide()

    def stage_update(self, data):
        try:
            decoded = json.loads(data.decode())
            self.stage_alpha.update_read_signal.emit(decoded["alpha"])
            self.stage_beta.update_read_signal.emit(decoded["beta"])

            if "on_target" in decoded:
                for item in [self.stage_alpha, self.stage_beta]:
                    item.setProperty("busy", not decoded["on_target"])
                    item.update_style_signal.emit()
        except:
            for item in [self.stage_alpha, self.stage_beta]:
                item.setProperty("error", True)
                item.update_style_signal.emit()
