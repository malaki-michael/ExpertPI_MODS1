from PySide6 import QtCore, QtGui, QtSvgWidgets, QtWidgets


class TitleWithIcon(QtWidgets.QWidget):
    def __init__(self, name, icon_file):
        super().__init__()
        self.name = name
        self.icon_file = icon_file

        icon = QtSvgWidgets.QSvgWidget(icon_file)
        icon.setFixedSize(25, 25)
        self.setStyleSheet(
            "background-color:#3f3e3e;" "font-size:18;" "font-weight:bold;" "margin:0px;" "padding-left:5px;"
        )

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        icon_wrapper = QtWidgets.QWidget()
        icon_wrapper.setLayout(QtWidgets.QHBoxLayout())
        icon_wrapper.setStyleSheet("padding:0px;margin:0px;")
        icon_wrapper.layout().setContentsMargins(0, 0, 0, 0)
        icon_wrapper.layout().setSpacing(0)
        icon_wrapper.layout().addWidget(icon)
        icon_wrapper.setFixedSize(30, 30)

        self.layout().addWidget(icon_wrapper)
        self.label = QtWidgets.QLabel(name)
        self.label.setStyleSheet("color:silver;")
        self.layout().addWidget(self.label)


class Toolbar(QtWidgets.QWidget):
    def __init__(self, name, icon, panel_size: int, expand_type="left"):
        self.name = name
        self.icon = icon
        self.expanded = False
        self.expand_type = expand_type
        self.panel_size = panel_size
        super().__init__()

        self.setMinimumHeight(50)

        # self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))

        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.setLayout(container_layout)
        self.main_layout = QtWidgets.QVBoxLayout()

        self.container_widget = QtWidgets.QWidget()

        container_layout.addWidget(self.container_widget)
        self.container_widget.setLayout(self.main_layout)

        self.shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.shadow.setColor(QtGui.QColor(0, 0, 0, 255))
        self.shadow.setBlurRadius(5)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)

        self.setGraphicsEffect(self.shadow)

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.header = TitleWithIcon(self.name, self.icon)
        self.main_layout.addWidget(self.header)
        # self.header.mouseEnter = lambda x: self.expand(True)
        # self.header.mouseLeave = lambda x: self.expand(False)

        self.header.setToolTip("Double click to expand functionality")

        self.content = QtWidgets.QWidget()
        self.main_layout.addWidget(self.content)

        self.content_layout = QtWidgets.QGridLayout()
        self.content.setLayout(self.content_layout)
        self.content.layout().setContentsMargins(0, 0, 0, 0)
        self.content.layout().setSpacing(0)
        self.setMaximumWidth(self.panel_size * 2)

        self.slider_width = 0
        self.toolbar_manager = None

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.expand(not self.expanded)

    def expand(self, value):
        if self.expand_type == "right":
            if value:
                self.setFixedWidth(self.panel_size * 2 - self.slider_width)
                self.move(self.pos().x() - self.panel_size, self.pos().y())
            else:
                self.setFixedWidth(self.panel_size - self.slider_width)
                self.move(self.pos().x() + self.panel_size, self.pos().y())
        elif value:
            self.setFixedWidth(self.panel_size * 2 - self.slider_width)
        else:
            self.setFixedWidth(self.panel_size - self.slider_width)
        self.expanded = value
