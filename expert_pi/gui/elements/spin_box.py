from PySide6 import QtWidgets, QtCore, QtGui


class SpinBoxWithUnits(QtWidgets.QDoubleSpinBox):
    set_signal = QtCore.Signal(float)
    update_read_signal = QtCore.Signal(float)
    set_read_sent_signal = QtCore.Signal(float)
    update_style_signal = QtCore.Signal()

    def __init__(
        self, initial_value, range, step, units, decimals=3, send_immediately=False, update_read=True, tooltip=None
    ):
        self.read_value = initial_value
        self.sent_value = initial_value

        super().__init__()

        self.setStyleSheet(
            "QDoubleSpinBox::up-button {subcontrol-position: top right;}\
                           QDoubleSpinBox::down-button {subcontrol-position: bottom right;}"
        )

        self.units = units
        self.send_immediately = send_immediately
        self.update_read = update_read

        self.setDecimals(decimals)
        self.setMinimum(range[0])
        self.setMaximum(range[1])

        self.initial_step = step
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setProperty("class", "toolbarSpinBox")
        self.setValue(initial_value)
        if tooltip is not None:
            self.setToolTip(tooltip)

        self.update_read_signal.connect(self.update_read_value)
        self.set_read_sent_signal.connect(self.set_read_sent_value)
        self.editingFinished.connect(self.editing_finished)
        self.update_style_signal.connect(self.update_style)

    def emit_set_signal(self, value):
        if self.sent_value == value:
            return
        if self.update_read:
            self.read_value = value
        self.setValue(value)
        self.sent_value = value
        self.set_signal.emit(value)

    def update_read_value(self, value):
        self.read_value = value
        if not self.hasFocus():
            self.setValue(self.value())  # trigger update text

    def set_read_sent_value(self, value):
        self.read_value = value
        self.sent_value = value
        self.setValue(value)

    def editing_finished(self):
        self.emit_set_signal(self.value())
        self.clearFocus()

    def textFromValue(self, value):
        if self.hasFocus():
            return f"{value:{self.decimals() + 4}.{self.decimals()}f}"
        else:
            return f"{self.read_value:{self.decimals() + 4}.{self.decimals()}f} {self.units}"

    def focusInEvent(self, ev):
        super().focusInEvent(ev)
        self.setValue(self.value())  # trigger update text

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        if self.send_immediately:
            self.emit_set_signal(self.value())

    def wheelEvent(self, e: QtGui.QWheelEvent, force=False) -> None:
        if self.hasFocus() or force:
            super().wheelEvent(e)
            if self.send_immediately:
                self.emit_set_signal(self.value())
        else:
            e.ignore()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Q:
            self.setSingleStep(self.singleStep() / 2.0)
            event.accept()
        elif event.key() == QtCore.Qt.Key.Key_W:
            self.setSingleStep(self.singleStep() * 2.0)
            event.accept()
        elif event.key() == QtCore.Qt.Key.Key_R:
            self.setSingleStep(self.initial_step)
            event.accept()
        elif event.key() == QtCore.Qt.Key.Key_Return or event.key() == QtCore.Qt.Key.Key_Enter:
            self.setValue(self.valueFromText(self.text()))  # trigger update text
            self.editing_finished()
            event.accept()
        else:
            super().keyPressEvent(event)

    def update_style(self):
        self.setStyleSheet(self.styleSheet())


class PrefixSpinBoxWithUnits(SpinBoxWithUnits):
    def textFromValue(self, value):
        if self.hasFocus():
            return f"{value:{self.decimals() + 4}.{self.decimals()}f}"
        else:
            return f"±{self.read_value:{self.decimals() + 4}.{self.decimals()}f} {self.units}"
