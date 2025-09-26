from PySide6 import QtWidgets


class MaskSelectorWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text = QtWidgets.QLabel("mask selector")
        layout.addWidget(self.text, 0, 0, 1, 1)
