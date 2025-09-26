# ruff: noqa: N802, PLR6301
from PySide6 import QtCore, QtWidgets

from expert_pi.gui.central_layout.data_manager import DataManager
from expert_pi.gui.central_layout.toolbar_managers import ToolbarManager


class CentralLayout(QtWidgets.QLayout):
    def __init__(self, parent: QtWidgets.QWidget, panel_size: int):
        super().__init__()

        self.layout_parent = parent
        self.panel_size = panel_size

        self.left_toolbars = ToolbarManager(parent, panel_size, "left")
        self.right_toolbars = ToolbarManager(parent, panel_size, "right")
        self.central = DataManager(parent, self.left_toolbars, self.right_toolbars, panel_size)

    def setGeometry(self, rect: QtCore.QRect | None = None) -> None:
        rect_use = self.layout_parent.geometry() if rect is None else rect

        self.left_toolbars.parent_resized(rect_use.size())
        self.right_toolbars.parent_resized(rect_use.size())
        self.central.parent_resized(rect_use)

    def minimumSize(self) -> QtCore.QSize:
        return QtCore.QSize(self.panel_size * 4, self.panel_size * 2)

    def set_toolbar_item_names(self, items: list[str] | None = None, type: str = "left"):
        """If None entire toolbar will be hidden. Use this for also changing the order of items."""
        if type == "left":
            if items is None:
                self.left_toolbars.hide()
                self.left_toolbars.visible_toolbars = []
            else:
                self.left_toolbars.show()
                self.left_toolbars.visible_toolbars = items
            self.left_toolbars.parent_resized()
        else:
            if items is None:
                self.right_toolbars.hide()
                self.right_toolbars.visible_toolbars = []
            else:
                self.right_toolbars.show()
                self.right_toolbars.visible_toolbars = items
            self.right_toolbars.parent_resized()

    def set_central_layout(self, layout):
        """See layout definition in DataManager."""
        self.central.rebuild_layout(layout)

    def count(self):
        """We must implement this virtual method."""
        return 0

    def itemAt(self, arg__1: int):
        """We must implement this virtual method."""
        return None

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.panel_size * 4, self.panel_size * 2)
