from expert_pi.gui.elements import buttons, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class Detectors(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("DETECTORS", images_dir + "section_icons/detectors.svg", panel_size)
        self.BF_insert = buttons.ToolbarStateButton("BF IN", name_off="BF OUT")
        self.HAADF_insert = buttons.ToolbarStateButton("DF IN", name_off="DF OUT")

        self.BF_scale = spin_box.SpinBoxWithUnits(1, [1e-06, 10.0], 0.1, "", send_immediately=True)
        self.BF_scale.setValue(1)
        self.BF_scale.setToolTip("Set to portion of current which hits BF detector")

        self.HAADF_scale = spin_box.SpinBoxWithUnits(1, [1e-06, 10.0], 0.1, "", send_immediately=True)
        self.HAADF_scale.setValue(1)
        self.HAADF_scale.setToolTip("Set to portion of current which hits HAADF detector")

        self.content_layout.addWidget(self.BF_insert, 0, 0, 1, 1)
        self.content_layout.addWidget(self.HAADF_insert, 1, 0, 1, 1)

        self.content_layout.addWidget(self.BF_scale, 0, 1, 1, 1)
        self.content_layout.addWidget(self.HAADF_scale, 1, 1, 1, 1)

        self.expand(False)

    def expand(self, value):
        super().expand(value)

    def update_detectors_state(self, detector: str, state: bool):
        print("updating", detector, state)
        if detector == "BF":
            self.BF_insert.update_selected_signal.emit(state)
        else:
            self.HAADF_insert.update_selected_signal.emit(state)
