from PySide6 import QtWidgets, QtCore


class ThreadedLabel(QtWidgets.QLabel):
    update_text_signal = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_text_signal.connect(self.setText)
