from PySide6 import QtCore, QtWidgets

from expert_pi.gui.elements import buttons, combo_box
from expert_pi.gui.style import images_dir


class XYZTrackingWidget(QtWidgets.QWidget):
    show_signal = QtCore.Signal()
    hide_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.start_tracking_button = buttons.ToolbarStateButton(
            "", icon=images_dir + "tools_icons/pause.svg", icon_off=images_dir + "tools_icons/start.svg"
        )
        self.start_tracking_button.setToolTip("Start/stop comparison stage regulation")
        layout.addWidget(self.start_tracking_button, 0, 0, 1, 6)

        self.regulate_button = buttons.ToolbarPushButton("REGULATE", selectable=True)
        self.regulate_button.setToolTip("Enable/disable stage regulation")
        layout.addWidget(self.regulate_button, 0, 6, 1, 6)

        self.start_tracking_button.setFixedHeight(25)
        self.regulate_button.setFixedHeight(25)

        self.refresh_reference_button = buttons.ToolbarPushButton(
            f"{10:0.2g} um", icon=images_dir + "tools_icons/xyz_tracking_take.svg"
        )
        self.refresh_reference_button.setIconSize(QtCore.QSize(20, 20))
        layout.addWidget(self.refresh_reference_button, 1, 0, 1, 6)

        self.customize_button = buttons.ToolbarPushButton("CUSTOMIZE", selectable=True, tooltip="draw reference region")
        layout.addWidget(self.customize_button, 1, 6, 1, 6)

        self.goto_button = buttons.ToolbarPushButton("GOTO")
        self.goto_button.setToolTip("Set stage back to the reference - only XYZ coordinates")
        layout.addWidget(self.goto_button, 2, 0, 1, 3)
        self.goto_button.setStyleSheet("text-align:right")

        self.goto_ab_button = buttons.HoverSignalsButton("+ α,β")
        self.goto_ab_button.setToolTip("Set stage back to the reference including alpha and beta tilt")
        layout.addWidget(self.goto_ab_button, 2, 3, 1, 3)
        self.goto_ab_button.setStyleSheet("text-align:left")

        self.goto_ab_button.hover_enter.connect(lambda: self.goto_button.setProperty("hover", True))
        self.goto_ab_button.hover_leave.connect(lambda: self.goto_button.setProperty("hover", False))

        self.refresh_reference_button.setFixedHeight(25)
        self.customize_button.setFixedHeight(25)

        self.debug_button = buttons.ToolbarPushButton("Debug", selectable=True)
        layout.addWidget(self.debug_button, 2, 6, 1, 6)

        self.axes = ["X", "Y", "Z"]
        self.axes_labels = {}
        for i, name in enumerate(self.axes):
            self.axes_labels[name] = QtWidgets.QLabel(f"{name}:0.0")
            layout.addWidget(self.axes_labels[name], 3, 3 * i + 3, 1, 3)

        self.actual_fov = combo_box.ToolbarComboBox()
        self.actual_fov_values = []

        layout.addWidget(self.actual_fov, 3, 0, 1, 3)

    def show(self):
        super().show()
        self.show_signal.emit()

    def hide(self):
        super().hide()
        self.hide_signal.emit()
