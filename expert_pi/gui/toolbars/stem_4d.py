from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class Stem4D(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("4DSTEM", images_dir + "section_icons/stem_4d.svg", panel_size)

        self.pixel_time_spin = spin_box.SpinBoxWithUnits(0.22, [0.014, 100_000_000], 1, "ms")  # TODO roi mod higher fps
        self.pixel_time_spin.setToolTip("pixel time")

        self.size_combo = combo_box.ToolbarComboBox()
        for i in range(3, 14):
            self.size_combo.addItem(f"{2**i} px")
        self.size_combo.setCurrentIndex(7 - 3)
        self.size_combo.setToolTip("number of pixels of 4DSTEM image")

        self.auto_retract_detectors = QtWidgets.QCheckBox("auto retract")
        self.auto_retract_detectors.setChecked(True)

        self.acquire_button = buttons.ToolbarStateButton("EXIT 4DSTEM", name_off="ACQUIRE 3.6s <8GB")

        # used as also progress bar and visualize acquisition time/GB
        self.progress_button = buttons.ProgressButton("previous data")
        self.progress_button.setEnabled(False)

        self.progress_button.setToolTip("click to stop when running")

        self.content_layout.addWidget(self.pixel_time_spin, 0, 0, 1, 1)
        self.content_layout.addWidget(self.size_combo, 0, 1, 1, 1)
        self.content_layout.addWidget(self.acquire_button, 1, 0, 1, 2)
        self.content_layout.addWidget(self.progress_button, 2, 0, 1, 2)
        self.content_layout.addWidget(self.auto_retract_detectors, 0, 2, 1, 1)
        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.auto_retract_detectors.show()
        else:
            self.auto_retract_detectors.hide()
