from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class Camera(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("CAMERA", images_dir + "section_icons/camera.svg", expand_type="right", panel_size=panel_size)
        self.exposure = spin_box.SpinBoxWithUnits(20, [0, 100000], 1, "ms")
        self.exposure.setToolTip("Camera exposure")

        roi_options = {
            "full": ("FULL", None),
            "roi256": ("ROI 256", None),
            "roi128": ("ROI 128", None),
        }
        self.roi_buttons = buttons.ToolbarMultiButton(roi_options, default_option="full")

        self.fov = spin_box.SpinBoxWithUnits(50, [1, 1000], 1, "mrad")
        self.fov.setToolTip("Angular field of view")

        control_options = {
            "start": ("", images_dir + "tools_icons/start.svg"),
            "1x": ("1x", None),
            "stop": ("", images_dir + "tools_icons/stop.svg"),
        }
        self.control_buttons = buttons.ToolbarMultiButton(control_options, default_option="1x")

        self.HAADF_angles = QtWidgets.QLabel("HAADF: ?-?mrad")
        self.BF_angles = QtWidgets.QLabel("BF: ?-?mrad")

        self.use_rotation = buttons.ToolbarPushButton("rotated", True)
        self.use_rotation.setToolTip("match rotation with STEM image")
        self.use_rotation.setProperty("selected", True)
        self.use_rotation.selected = True
        self.use_rotation.setStyleSheet(self.use_rotation.styleSheet())

        self.content_layout.addWidget(self.roi_buttons, 0, 1, 1, 1)
        self.content_layout.addWidget(self.exposure, 1, 1, 1, 1)
        self.content_layout.addWidget(self.fov, 2, 1, 1, 1)
        self.content_layout.addWidget(self.control_buttons, 3, 1, 1, 1)

        self.content_layout.addWidget(self.use_rotation, 1, 0, 1, 1)
        self.content_layout.addWidget(self.HAADF_angles, 2, 0, 1, 1)
        self.content_layout.addWidget(self.BF_angles, 3, 0, 1, 1)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            for item in [self.HAADF_angles, self.BF_angles, self.use_rotation]:
                item.show()
        else:
            for item in [self.HAADF_angles, self.BF_angles, self.use_rotation]:
                item.hide()
