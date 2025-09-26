from PySide6 import QtWidgets, QtCore, QtGui


class ToolbarComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        self.setProperty("class", "toolbarComboBox")


class SelectableComboBox(QtWidgets.QComboBox):
    clicked = QtCore.Signal(bool)
    update_selected_signal = QtCore.Signal(bool)

    def __init__(self, options, selected=False):
        super().__init__()
        self.setProperty("class", "toolbarSelectableComboBox")
        self.options = options

        for option in self.options:
            self.addItem(option)

        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.selected = selected
        self.busy = False
        self.setProperty("selected", selected)
        self.setProperty("busy", False)

        self.update_selected_signal.connect(self.set_selected)

    def mousePressEvent(self, ev):
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        control = self.style().hitTestComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt, ev.pos(), self)

        if control == QtWidgets.QStyle.SubControl.SC_ScrollBarAddPage or self.currentIndex() == -1:
            super().mousePressEvent(ev)

        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.set_selected(not self.selected)
            self.clicked.emit(self.selected)

    def set_busy(self, value):
        self.setProperty("busy", value)
        self.busy = value
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def set_error(self, value):
        self.setProperty("error", value)
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def set_selected(self, value, set_busy=False):
        self.setProperty("selected", value)
        self.selected = value
        self.set_busy(set_busy)
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def update_style(self):
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property
