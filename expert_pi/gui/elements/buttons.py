from PySide6 import QtWidgets, QtCore, QtGui


class ToolbarMultiButton(QtWidgets.QWidget):
    clicked = QtCore.Signal(str)

    def __init__(self, options, multi_select=False, no_select=False, default_option=None, vertical=False):
        super().__init__()
        self.multi_select = multi_select
        self.no_select = no_select
        layout = QtWidgets.QVBoxLayout() if vertical else QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.options = options
        self.buttons = {}
        self.selected: list = [] if default_option is None else default_option
        for name, option in self.options.items():
            widget = QtWidgets.QPushButton(option[0])
            self.layout().addWidget(widget)
            self.buttons[name] = widget
            if option[1] is not None:
                self.buttons[name].setIcon(QtGui.QIcon(option[1]))
            self.buttons[name].setProperty("class", "toolbarButton")
            if self.multi_select and self.selected:
                self.buttons[name].setProperty("selected", name in self.selected)
            else:
                self.buttons[name].setProperty("selected", name == self.selected)
            self.buttons[name].setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            self.buttons[name].clicked.connect((lambda index: (lambda: self.option_clicked(index)))(name))

    def option_clicked(self, name, emit=True):
        if self.multi_select:
            if not self.selected:
                self.buttons[name].setProperty("selected", True)
                self.selected.append(name)
            elif name in self.selected:
                self.selected.remove(name)
                self.buttons[name].setProperty("selected", False)
            else:
                self.selected.append(name)
                self.buttons[name].setProperty("selected", True)
        elif self.no_select:
            if name == self.selected:
                self.selected = []
                self.buttons[name].setProperty("selected", False)
            else:
                if self.selected:
                    self.buttons[self.selected].setProperty("selected", False)
                self.selected = name
                self.buttons[name].setProperty("selected", True)
        else:
            if self.selected:
                self.buttons[self.selected].setProperty("selected", False)
            self.selected = name
            self.buttons[self.selected].setProperty("selected", True)

        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property
        if emit:
            self.clicked.emit(name)


class ToolbarPushButton(QtWidgets.QPushButton):
    update_selected_signal = QtCore.Signal(bool)
    update_text_signal = QtCore.Signal(str)
    update_style_signal = QtCore.Signal()
    update_enabled_signal = QtCore.Signal(bool)

    def __init__(self, name, selectable=False, icon=None, tooltip=None):
        super().__init__(name)
        if icon is not None:
            self.setIcon(QtGui.QIcon(icon))
        self.setProperty("class", "toolbarButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.selectable = selectable
        self.selected = False
        self.set_value = 0.0

        if tooltip:
            self.setToolTip(tooltip)

        self.update_selected_signal.connect(self.set_selected)  # use this from different threads
        self.update_text_signal.connect(self.setText)
        self.update_style_signal.connect(self.update_style)
        self.update_enabled_signal.connect(self.setEnabled)

    def update_style(self):
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def mousePressEvent(self, e):
        if self.selectable:
            self.set_selected(not self.selected)
        super().mousePressEvent(e)

    def set_selected(self, value):
        self.setProperty("selected", value)
        self.selected = value
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property


class ToolbarStateButton(QtWidgets.QPushButton):
    clicked = QtCore.Signal(bool)
    update_selected_signal = QtCore.Signal(bool)
    update_text_signal = QtCore.Signal(str)

    def __init__(self, name, selected=False, name_off=None, icon=None, icon_off=None, tooltip=None):
        if not selected and name_off is not None:
            super().__init__(name_off)
        else:
            super().__init__(name)
        self.names = [name, name_off]
        self.icons = [icon, icon_off]
        if (selected and icon is not None) or (not selected and icon_off is None):
            self.setIcon(QtGui.QIcon(icon))
        elif not selected and icon_off is not None:
            self.setIcon(QtGui.QIcon(icon_off))

        self.setProperty("class", "toolbarButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.selected = selected
        self.busy = False
        self.setProperty("selected", selected)
        self.setProperty("busy", False)
        if tooltip:
            self.setToolTip(tooltip)

        self.update_selected_signal.connect(self.set_selected)
        self.update_text_signal.connect(self.setText)

    def set_busy(self, value):
        self.setProperty("busy", value)
        self.busy = value
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def set_error(self, value):
        self.setProperty("error", value)
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def set_selected(self, value, set_busy=False):
        if value:
            if self.names[0] is not None and self.names[1] is not None:
                self.setText(self.names[0])
            if self.icons[0] is not None:
                self.setIcon(QtGui.QIcon(self.icons[0]))
        else:
            if self.names[0] is not None and self.names[1] is not None:
                self.setText(self.names[1])
            if self.icons[1] is not None:
                self.setIcon(QtGui.QIcon(self.icons[1]))

        self.setProperty("selected", value)
        self.selected = value
        self.set_busy(set_busy)
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def update_style(self):
        self.setStyleSheet(self.styleSheet())  # need to redraw the selected property

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.set_selected(not self.selected)
            self.clicked.emit(self.selected)
        else:
            super().mousePressEvent(e)


class HoverSignalsButton(ToolbarPushButton):
    hover_enter = QtCore.Signal()
    hover_leave = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.installEventFilter(self)

    def eventFilter(self, watched, event):
        result = super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.HoverEnter:
            self.hover_enter.emit()
        elif event.type() == QtCore.QEvent.HoverLeave:
            self.hover_leave.emit()
        return result


class IconButton(QtWidgets.QPushButton):
    hover_enter = QtCore.Signal()
    hover_leave = QtCore.Signal()

    def __init__(self, icon, icon_hover=None, tooltip=None):
        super().__init__()
        self.icon = icon
        self.icon_hover = icon_hover
        self.installEventFilter(self)
        self.setProperty("class", "iconButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        if icon is not None:
            self.setIcon(QtGui.QIcon(self.icon))

        self.setToolTip(tooltip)

    def eventFilter(self, watched, event):
        result = super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.HoverEnter:
            if self.icon_hover is not None:
                self.setIcon(QtGui.QIcon(self.icon_hover))
            self.hover_enter.emit()
        elif event.type() == QtCore.QEvent.HoverLeave:
            self.setIcon(QtGui.QIcon(self.icon))
            self.hover_leave.emit()
        return result


class IconSwitchableButton(QtWidgets.QPushButton):
    hover_enter = QtCore.Signal()
    hover_leave = QtCore.Signal()
    update_selected_signal = QtCore.Signal(bool)

    def __init__(self, icon, icon_selected, icon_hover=None, icon_selected_hover=None, tooltip=None):
        super().__init__()
        self.setProperty("class", "iconButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.update_selected_signal.connect(self.set_selected)
        self.installEventFilter(self)
        self.icon = icon
        self.icon_hover = icon_hover

        self.linked_button = None

        self.icon_selected = icon_selected
        self.icon_selected_hover = icon_selected_hover

        self.selected = False
        if icon is not None:
            self.setIcon(QtGui.QIcon(self.icon))

        self.setToolTip(tooltip)

    def eventFilter(self, watched, event):
        result = super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.HoverEnter:
            self.update_icon(True)
            self.hover_enter.emit()
        elif event.type() == QtCore.QEvent.HoverLeave:
            self.update_icon()
            self.hover_leave.emit()
        return result

    def update_icon(self, hover=False):
        if self.selected:
            if hover:
                self.setIcon(QtGui.QIcon(self.icon_selected_hover))
            else:
                self.setIcon(QtGui.QIcon(self.icon_selected))
        else:
            if hover:
                self.setIcon(QtGui.QIcon(self.icon_hover))
            else:
                self.setIcon(QtGui.QIcon(self.icon))

    def set_selected(self, value):
        self.selected = value
        self.update_icon()

    def mousePressEvent(self, event):
        if not self.selected or self.linked_button is None:
            self.selected = not self.selected
        if self.linked_button is not None:
            self.linked_button.set_selected(not self.selected)
        self.update_icon(True)
        super().mousePressEvent(event)


class ProgressButton(ToolbarPushButton):
    set_progress_signal = QtCore.Signal(float, str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_progress_signal.connect(self.set_progress)
        self.progress = 0
        self.hover = False

    def leaveEvent(self, e) -> None:
        self.hover = False
        self.update_style()

    def enterEvent(self, e) -> None:
        self.hover = True
        self.update_style()

    def update_style(self):
        if self.hover:
            self.setStyleSheet(
                f"background: qlineargradient( x1:{self.progress} y1:0, x2:{self.progress + 1e-6} y2:0,"
                + " stop:0 #8888ff ,stop:1 #363548)"
            )

        else:
            self.setStyleSheet(
                f"background: qlineargradient( x1:{self.progress} y1:0, x2:{self.progress + 1e-6} y2:0,"
                + " stop:0 #6666dd ,stop:1 #302F2F)"
            )

    def set_progress(self, progress, text):
        self.progress = progress
        self.update_style()
        self.setText(text)
