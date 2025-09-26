from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets


class ElementItem:
    def __init__(
        self,
        atomic_number: int,
        name: str,
        lines: dict[str, float],
        update_function: Callable[["ElementItem"], None],
        get_auto_color: Callable,
    ):
        self.atomic_number = atomic_number
        self.name = name
        self.lines = lines
        self._update_function = update_function

        self.color = ""
        self.selected = False
        self.active = False
        self.hover = False
        self.get_auto_color = get_auto_color

    def update_state(
        self,
        selected: bool | None = None,
        active: bool | None = None,
        hover: bool | None = None,
        color: str | None = None,
    ):
        if selected is not None:
            self.selected = selected
        if active is not None:
            self.active = active
        if hover is not None:
            self.hover = hover
        if color is not None:
            if color == "auto":
                color = self.get_auto_color()
            self.color = color

        self._update_function(self)


class ElementButton(QtWidgets.QWidget):
    def __init__(self, element_item: ElementItem):
        super().__init__()

        self.element_item = element_item
        self.name = element_item.name

        self.setStyleSheet("QLabel{background-color:none;color:black}")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self.background = QtWidgets.QWidget()
        self.background.setStyleSheet("border:0px solid black;")
        self.background.setParent(self)
        self.background.setFixedSize(40, 40)
        self.update_style()
        self.setFixedSize(40, 40)

        self.name_label = QtWidgets.QLabel(self.name)
        self.name_label.setParent(self)
        self.name_label.move(11, 0)
        self.name_label.setStyleSheet("font-size:12pt")

        self.atomic_label = QtWidgets.QLabel(str(self.element_item.atomic_number))
        self.atomic_label.setParent(self)
        self.atomic_label.move(1, 0)
        self.atomic_label.setStyleSheet("font-size:6pt")

        atomic_number = self.element_item.atomic_number
        lines = self.element_item.lines
        self.line_labels = []
        if atomic_number < 4:
            pass
        elif atomic_number < 20:
            if "Ka1" in lines:
                self.line_labels.append(QtWidgets.QLabel(f"Ka {(lines['Ka1'] / 1000):6.2f}"))
        elif atomic_number < 58:
            if "Ka1" in lines:
                self.line_labels.append(QtWidgets.QLabel(f"Ka {(lines['Ka1'] / 1000):6.2f}"))
            if "La1" in lines:
                self.line_labels.append(QtWidgets.QLabel(f"La {(lines['La1'] / 1000):6.2f}"))
        elif atomic_number < 98:
            if "La1" in lines:
                self.line_labels.append(QtWidgets.QLabel(f"La {(lines['La1'] / 1000):6.2f}"))
            if "Ma" in lines:
                self.line_labels.append(QtWidgets.QLabel(f"Ma {(lines['Ma'] / 1000):6.2f}"))
        else:
            pass

        for i, item in enumerate(self.line_labels):
            item.setParent(self)
            item.move(2, 20 + i * 9)
            item.setStyleSheet("font-size:6pt")

    def update_style(self):
        if self.element_item.hover:
            if self.element_item.selected:
                self.background.setStyleSheet("QWidget{background-color:#aaaaff;color:black}")
            else:
                self.background.setStyleSheet("QWidget{background-color:#6666dd;color:black}")
        elif self.element_item.selected:
            self.background.setStyleSheet("QWidget{background-color:#6666dd;color:black}")
        else:
            self.background.setStyleSheet("QWidget{background-color:#666;color:black}")

    def mousePressEvent(self, e) -> None:
        select = not self.element_item.selected
        self.element_item.update_state(selected=select, active=select, color="auto")

    def leaveEvent(self, e) -> None:
        self.element_item.update_state(hover=False)

    def enterEvent(self, e) -> None:
        self.element_item.update_state(hover=True)


class PeriodicTable(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Periodic table")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, on=True)

        self.setStyleSheet(open("expert_pi/gui/style/style.qss").read())
        self.button_layout = QtWidgets.QGridLayout()
        self.setLayout(self.button_layout)
        self.button_layout.setContentsMargins(2, 2, 2, 2)
        self.button_layout.setSpacing(1)

        self.elements_buttons = {}

        self.button_layout.addWidget(QtWidgets.QLabel(" "), 8, 0)

    def add_element_button(self, element_item: ElementItem):
        atomic_number = element_item.atomic_number
        position = atomic_number - 1
        if atomic_number > 1:
            position += (5 + 3) * 2
        if atomic_number > 4:
            position += 5 * 2
        if atomic_number > 12:
            position += 5 * 2
        if atomic_number > 57:
            position -= 7 * 2
        if atomic_number > 89:
            position -= 7 * 2
        if 56 < atomic_number < 72:
            position = 9 * 18 + 3 + atomic_number - 58

        if 88 < atomic_number < 104:
            position = 10 * 18 + 3 + atomic_number - 90

        i, j = (position // 18, position % 18)

        element = ElementButton(element_item)
        self.elements_buttons[element_item.name] = element
        self.button_layout.addWidget(element, i, j)

    def update_table(self, element_name):
        self.elements_buttons[element_name].update_style()
