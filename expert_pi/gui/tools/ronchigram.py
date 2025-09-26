from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons


class RonchigramWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tilt_checkbox = QtWidgets.QCheckBox("tilt")
        self.astigmatismus_checkbox = QtWidgets.QCheckBox("astigmatism")
        self.defocus_checkbox = QtWidgets.QCheckBox("defocus")
        self.empty = QtWidgets.QLabel("")

        self.tilt_checkbox.setChecked(True)
        self.astigmatismus_checkbox.setChecked(True)
        self.defocus_checkbox.setChecked(True)

        layout.addWidget(self.tilt_checkbox, 0, 0, 1, 1)
        layout.addWidget(self.astigmatismus_checkbox, 0, 1, 1, 1)
        layout.addWidget(self.defocus_checkbox, 1, 0, 1, 1)
        layout.addWidget(self.empty, 1, 1, 1, 1)

        self.hint_button = buttons.ToolbarStateButton("Hint", selected=True)
        self.optimize_button = buttons.ToolbarStateButton("Optimize")

        layout.addWidget(self.hint_button, 2, 0, 1, 1)
        layout.addWidget(self.optimize_button, 2, 1, 1, 1)
