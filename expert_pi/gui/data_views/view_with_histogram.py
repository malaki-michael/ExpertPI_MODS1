from PySide6 import QtWidgets


class ViewWithHistogram(QtWidgets.QWidget):
    def __init__(self, view, histogram):
        super().__init__()
        self.name = view.name
        self.view = view
        self.histogram = histogram

        self.histogram.setFixedHeight(100)

        self.setLayout(QtWidgets.QVBoxLayout())

        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.view)
        self.layout().addWidget(self.histogram)
