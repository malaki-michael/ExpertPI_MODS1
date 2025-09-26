from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, spin_box


class OptionWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.mask_type = buttons.ToolbarMultiButton(
            {"full": ("Full", ""), "angular": ("Angular", "")}, default_option="full"
        )
        for button in self.mask_type.buttons.values():
            button.setFixedHeight(23)

        self.segments = spin_box.SpinBoxWithUnits(1, [1, 16], 1, units="x", decimals=0, tooltip="number of segments")
        self.segments.setFixedWidth(50)

        layout.addWidget(self.mask_type, 0, 0, 1, 1)
        layout.addWidget(self.segments, 0, 1, 1, 1)

        self.setFixedHeight(23)
