# ruff: noqa: N802
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.elements import buttons
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base
from expert_pi.gui.tools import periodic_table
from expert_pi.gui.tools.periodic_table import ElementItem


class ElementButton(QtWidgets.QPushButton):
    def __init__(self, element_item: ElementItem, width=20, parent=None):
        super().__init__(element_item.name, parent=parent)
        self.element_item = element_item
        self.name = element_item.name
        self.color_off = "#3f3e3e"
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setProperty("class", "elementButton")
        self.update_style()
        self.setFixedWidth(width)

    def update_style(self):
        self.setProperty("selected", self.element_item.active)

        if self.element_item.active:
            if not self.element_item.color:
                color = "#ffffff"
            else:
                color = self.element_item.color
            self.setStyleSheet(f"background-color:{color}")
        else:
            self.setStyleSheet(f"background-color:{self.color_off}")

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.element_item.update_state(active=not self.element_item.active)

        if e.button() == QtCore.Qt.MouseButton.RightButton:
            self.element_item.update_state(selected=False, active=False, color="")


class ElementsSelection(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.buttons_layout = QtWidgets.QGridLayout()
        self.setLayout(self.buttons_layout)
        self.buttons_layout.setSpacing(1)
        self.buttons_layout.setContentsMargins(2, 2, 2, 2)
        self.buttons_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

        self.setFixedHeight(50)
        self.element_width = 25

        self.elements_buttons: dict[str, ElementButton] = {}

    def update_elements(self, element_item: ElementItem):
        element_name = element_item.name
        if element_name in self.elements_buttons:
            if element_item.selected:
                self.elements_buttons[element_name].update_style()
            else:
                del self.elements_buttons[element_name]
                self.update_layout()

        elif element_item.selected:
            self.elements_buttons[element_name] = ElementButton(element_item, width=self.element_width, parent=self)
            self.update_layout()

    def update_layout(self):
        while not self.buttons_layout.isEmpty():
            self.buttons_layout.takeAt(0).widget().setParent(None)

        n = int(self.width() / self.element_width)

        for i, item in enumerate(self.elements_buttons.values()):
            self.buttons_layout.addWidget(
                item,
                i // n,
                i % n,
                1,
                1,
            )


class AddElementEdit(QtWidgets.QLineEdit):
    emit_add_element = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Return or event.key() == QtCore.Qt.Key.Key_Enter:
            self.emit_add_element.emit(self.text())
            self.setText("")
        super().keyPressEvent(event)


class Xray(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("XRAY", images_dir + "section_icons/xray.svg", expand_type="right", panel_size=panel_size)
        self.expand(False)

        edx_detectors_options = {"EDX0": ("EDX0", ""), "EDX1": ("EDX1", "")}
        self.edx_detectors = buttons.ToolbarMultiButton(
            edx_detectors_options, multi_select=True, default_option=["EDX0", "EDX1"]
        )

        self.content_layout.addWidget(self.edx_detectors, 0, 0, 1, 2)

        self.calibrate_button = buttons.ToolbarPushButton("Calibrate")
        self.calibrate_info = QtWidgets.QLabel("138/138 eV  +2.4/+3.1%")

        self.content_layout.addWidget(self.calibrate_button, 1, 0, 1, 2)
        self.content_layout.addWidget(self.calibrate_info, 2, 0, 1, 1)

        self.add_element_edit = AddElementEdit()
        self.add_element_edit.setPlaceholderText("add element")
        self.content_layout.addWidget(self.add_element_edit, 3, 0, 1, 1)

        self.periodic_table = periodic_table.PeriodicTable()
        self.element_selection = ElementsSelection()

        self.open_periodic_table = buttons.ToolbarPushButton("+")
        self.open_periodic_table.clicked.connect(self.periodic_table.showNormal)
        self.content_layout.addWidget(self.open_periodic_table, 3, 1, 1, 1)

        self.content_layout.addWidget(self.element_selection, 4, 0, 1, 2)

    def get_selected_detectors(self) -> list[str]:
        result = []
        if "EDX0" in self.edx_detectors.selected:
            result.append("EDX0")
        if "EDX1" in self.edx_detectors.selected:
            result.append("EDX1")
        return result
