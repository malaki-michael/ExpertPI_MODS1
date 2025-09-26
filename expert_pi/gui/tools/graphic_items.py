import numpy as np
from PySide6 import QtGui, QtWidgets

color = QtGui.QColor(100, 100, 255, 255)
color_hover = QtGui.QColor(200, 200, 255, 255)


pen = QtGui.QPen(color, 0)
# pen.setCosmetic(True) #do not use cosmetic it will mess up itemAt selection
pen_hover = QtGui.QPen(color_hover, 0)
# pen_hover.setCosmetic(True)


class ItemGroup:
    def __init__(self):
        self.items = {}

    def has_item(self, item):
        if item is None:
            return False
        for child in self.items.values():
            if item == child:
                return True
            if hasattr(child, "has_item"):
                result = child.has_item(item)
                if result:
                    return True
        return False


class DragPoint(QtWidgets.QGraphicsItemGroup, ItemGroup):
    def __init__(self, on_drag_callback):
        super().__init__()
        self.on_drag_callback = on_drag_callback
        self.point_size = 6  # px
        self.select_size = 8  # px
        self.items = {
            "visible_point": QtWidgets.QGraphicsRectItem(
                -self.point_size / 2, -self.point_size / 2, self.point_size, self.point_size
            ),
            "selection_rect": QtWidgets.QGraphicsRectItem(
                -self.select_size / 2, -self.select_size / 2, self.select_size, self.select_size
            ),
        }

        self.items["selection_rect"].setBrush(QtGui.QColor(0, 0, 0, 0))

        self.items["selection_rect"].setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0), 0))

        for name, item in self.items.items():
            item.setParentItem(self)
            item.hover = self.hover
            item.hover_leave = self.hover_leave
            item.drag = self.drag

        self.hover_leave()

    def hover(self):
        self.items["visible_point"].setPen(pen_hover)

    def hover_leave(self):
        self.items["visible_point"].setPen(pen)

    def drag(self, x, y):
        self.setPos(x, y)
        self.on_drag_callback(x, y)


class DragLine(QtWidgets.QGraphicsItemGroup, ItemGroup):
    def __init__(self, x1, y1, x2, y2, on_drag_callback, linked_hover=None):
        super().__init__()
        self.on_drag_callback = on_drag_callback
        self.linked_hover = linked_hover
        self.px_select_size = 8  # px
        self.select_size = self.px_select_size * 1  # modify this during rescaling of scene
        self.items = {
            "visible_line": QtWidgets.QGraphicsLineItem(x1, y1, x2, y2),
            "selection_rect": QtWidgets.QGraphicsRectItem(0, 0, 0, 0),
        }

        self.items["selection_rect"].setBrush(QtGui.QColor(0, 0, 0, 0))
        self.items["selection_rect"].setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0), 0))

        for name, item in self.items.items():
            item.setParentItem(self)
            item.hover = self.hover
            item.hover_leave = self.hover_leave
            item.drag = self.drag

        self.interactivity_enabled = True

        self.hover_leave()

    def setLine(self, x1, y1, x2, y2):
        self.items["visible_line"].setLine(x1, y1, x2, y2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        self.items["selection_rect"].setTransformOriginPoint(x1, y1)
        self.items["selection_rect"].setRotation(angle / np.pi * 180)
        self.items["selection_rect"].setRect(
            x1, y1 - self.select_size, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2 * self.select_size
        )

    def line(self):
        return self.items["visible_line"].line()

    def hover(self, link_emit=True):
        if self.interactivity_enabled:
            self.items["visible_line"].setPen(pen_hover)
        if self.linked_hover is not None and link_emit:
            self.linked_hover.hover(link_emit=False)

    def hover_leave(self, link_emit=True):
        if self.interactivity_enabled:
            self.items["visible_line"].setPen(pen)
        if self.linked_hover is not None and link_emit:
            self.linked_hover.hover_leave(link_emit=False)

    def drag(self, x, y):
        if self.interactivity_enabled:
            self.on_drag_callback(x, y)

    def set_scale(self, scale):
        self.select_size = self.px_select_size * scale
        line = self.items["visible_line"].line()
        x1, y1, x2, y2 = line.x1(), line.y1(), line.x2(), line.y2()
        self.items["selection_rect"].setRect(
            x1, y1 - self.select_size, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2 * self.select_size
        )
