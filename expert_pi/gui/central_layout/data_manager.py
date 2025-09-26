# ruff: noqa: N802
from PySide6 import QtCore, QtWidgets


class SpliterHandle(QtWidgets.QWidget):
    def __init__(self, orientation: QtCore.Qt.Orientation):
        super().__init__()
        self.orientation = orientation
        self.splitter = None

        self.size_px = 1
        self.setOrientation(orientation)
        self.setStyleSheet("background-color:red;")

    def setSplitter(self, splitter):
        self.splitter = splitter

    def setOrientation(self, orientation):
        self.orientation = orientation
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            self.setCursor(QtCore.Qt.CursorShape.SplitVCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.SplitHCursor)

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapToParent(event.position())
            if self.splitter is not None and self.splitter.geometry is not None:
                if self.orientation == QtCore.Qt.Orientation.Horizontal:
                    f = (pos.x() - self.splitter.geometry.left()) / self.splitter.geometry.width()
                else:
                    f = (pos.y() - self.splitter.geometry.top()) / self.splitter.geometry.height()
                self.splitter.position = max(0, min(f, 1))
                self.splitter.update()


class Splitter:
    def __init__(self, first, second, handle: SpliterHandle, position=0.618):
        self.first = first
        self.second = second
        self.handle = handle
        self.geometry = None
        self.position = position

        self.handle.setSplitter(self)

    def update(self):
        if self.geometry is not None:
            self.setGeometry(self.geometry)

    def show(self):
        pass

    def setGeometry(self, rect: QtCore.QRect):
        self.geometry = rect
        if self.handle.orientation == QtCore.Qt.Orientation.Vertical:
            rect0 = QtCore.QRect(rect.left(), rect.top(), rect.width(), int(rect.height() * self.position))
            rect_split = QtCore.QRect(rect.left(), rect0.bottom() + 1, rect.width(), self.handle.size_px)
            rect1 = QtCore.QRect(
                rect.left(), rect_split.bottom() + 1, rect.width(), rect.height() - self.handle.size_px - rect0.height()
            )
        else:
            rect0 = QtCore.QRect(rect.left(), rect.top(), int(rect.width() * self.position), rect.height())
            rect_split = QtCore.QRect(rect0.right() + 1, rect.top(), self.handle.size_px, rect.height())
            rect1 = QtCore.QRect(
                rect_split.right() + 1, rect.top(), rect.width() - self.handle.size_px - rect0.width(), rect.height()
            )

        self.first.setGeometry(rect0)
        self.first.show()

        self.handle.setGeometry(rect_split)
        self.handle.show()

        self.second.setGeometry(rect1)
        self.second.show()


class DataManager:
    def __init__(self, parent, left_toolbars, right_toolbars, panel_size: int):
        super().__init__()
        self.parent = parent
        self.left_toolbars = left_toolbars
        self.right_toolbars = right_toolbars
        self.panel_size = panel_size

        self.items = {}
        self._handles = []

        self.main_item: QtWidgets.QWidget | None = None
        self.layout = None

    def add_item(self, name, item, set_parent=True):
        if name in self.items:
            raise Exception(name + " already present")
        self.items[name] = item
        if set_parent:
            item.setParent(self.parent)
        item.lower()  # central items are with lowest  value
        return item

    def extract_item_names(self, layout):
        if isinstance(layout, str):
            return [layout]
        names = []
        for p in ["first", "second"]:
            if isinstance(layout[p], dict):
                names += self.extract_item_names(layout[p])
            else:
                names += [layout[p]]
        return names

    def rebuild_layout(self, layout):
        """Rebuild layout.

        recursive layout form:
        {
            "vertical":False,
            "position":0.5,
            "first":"name" or {layout},
            "second""name" or {layout}
        }
        or single item name
        """
        names = self.extract_item_names(layout)

        while len(self._handles) < len(names) - 1:
            self._handles.append(SpliterHandle(QtCore.Qt.Orientation.Vertical))
            self._handles[-1].setParent(self.parent)

        for handle in self._handles:
            handle.hide()

        for item in self.items.values():
            item.hide()

        self.layout = layout
        self.main_item, _ = self.build_splitters(layout, 0)
        self.main_item.show()
        self.parent_resized()

    def build_splitters(self, layout, handle_index) -> tuple[QtWidgets.QWidget, int]:
        if isinstance(layout, dict):
            handle = self._handles[handle_index]
            handle_index += 1

            first, handle_index = self.build_splitters(layout["first"], handle_index)
            second, handle_index = self.build_splitters(layout["second"], handle_index)

            if layout["vertical"]:
                handle.setOrientation(QtCore.Qt.Orientation.Vertical)
            else:
                handle.setOrientation(QtCore.Qt.Orientation.Horizontal)

            position = layout["position"]
            return Splitter(first, second, handle, position), handle_index
        else:
            return self.items[layout], handle_index

    def parent_resized(self, rect: QtCore.QRect | None = None):
        if rect is None:
            rect2 = self.parent.geometry()
            rect2 = QtCore.QRect(0, 0, rect2.width(), rect2.height())  # widgets with respect to parent
        else:
            rect2 = rect

        if self.left_toolbars.visible:
            rect2 = QtCore.QRect(
                rect2.left() + self.panel_size, rect2.top(), rect2.width() - self.panel_size, rect2.height()
            )

        if self.right_toolbars.visible:
            rect2 = QtCore.QRect(rect2.left(), rect2.top(), rect2.width() - self.panel_size, rect2.height())

        if self.main_item is not None:
            self.main_item.setGeometry(rect2)
