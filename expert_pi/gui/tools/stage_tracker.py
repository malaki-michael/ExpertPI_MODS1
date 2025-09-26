import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.tools import base


class StageTracker(base.Tool):
    def __init__(self, view):
        super().__init__(view)

        self.size = 20
        self.text_offset_factor = 3

        triangle = QtGui.QPolygonF()
        triangle.append(QtCore.QPointF(-self.size / 2, 0))
        triangle.append(QtCore.QPointF(self.size / 2, 0))
        triangle.append(QtCore.QPointF(0, self.size))

        self.items = {
            "circle": QtWidgets.QGraphicsEllipseItem(-self.size / 2, -self.size / 2, self.size, self.size),
            "hline": QtWidgets.QGraphicsLineItem(0, -self.size, 0, self.size),
            "wline": QtWidgets.QGraphicsLineItem(-self.size, 0, self.size, 0),
            "arrow": QtWidgets.QGraphicsPolygonItem(triangle),
            "text": QtWidgets.QGraphicsTextItem("text"),
        }

        for name, item in self.items.items():
            if name != "text":
                item.setPen(QtGui.QPen(QtGui.QColor(102, 102, 221, 255), 0))
            if name == "arrow":
                item.setBrush(QtGui.QColor(102, 102, 221, 255))
            item.setParentItem(self)

        self.hide()
        self.view.graphics_area.addItem(self)
        self.dragging = False
        self.moved = False

    def show(self):
        super().show()
        self.is_active = True
        self.update()

    def hide(self):
        self.is_active = False
        super().hide()

    def calculate_position(self, pos, rectangle):
        center = rectangle[:2] + rectangle[2:] / 2
        w = rectangle[2]
        h = rectangle[3]
        d = pos - center

        angle = np.arctan2(d[1], d[0])
        rect_angle = np.arctan2(h, w)

        if angle > (np.pi - rect_angle) or angle <= (rect_angle - np.pi):
            factor = -d[0] / (w / 2)
        elif angle > rect_angle:
            factor = d[1] / (h / 2)
        elif angle > -rect_angle:
            factor = d[0] / (w / 2)
        else:
            factor = -d[1] / (h / 2)

        if factor < 1:
            return pos, factor, d, angle
        else:
            return center + d / factor, factor, d, angle

    def setPos(self, *args, **kwargs):
        super().setPos(*args, **kwargs)
        self.update()

    def update(self, *args, **kwargs):
        # sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        rectangle = self.view.get_area_rectangle()

        position, distance_factor, vector, angle = self.calculate_position(
            np.array([self.pos().x(), self.pos().y()]), np.array(rectangle)
        )

        if distance_factor < 1:
            self.items["circle"].setRect(
                -self.size / 2 / scale, -self.size / 2 / scale, self.size / scale, self.size / scale
            )
            self.items["hline"].setLine(0, -self.size / scale, 0, self.size / scale)
            self.items["wline"].setLine(-self.size / scale, 0, self.size / scale, 0)

            self.items["circle"].show()
            self.items["wline"].show()
            self.items["hline"].show()
            self.items["arrow"].hide()
            self.items["text"].hide()

        else:
            triangle = QtGui.QPolygonF()
            triangle.append(QtCore.QPointF(-self.size / 2 / scale, -self.size / scale))
            triangle.append(QtCore.QPointF(self.size / 2 / scale, -self.size / scale))
            triangle.append(QtCore.QPointF(0, 0))

            distance = np.sqrt((position[0] - self.pos().x()) ** 2 + (position[1] - self.pos().y()) ** 2)
            position -= np.array([self.pos().x(), self.pos().y()])  # remove central offset

            self.items["arrow"].setPolygon(triangle)
            self.items["arrow"].setPos(position[0], position[1])
            self.items["arrow"].setRotation(angle / np.pi * 180 - 90)

            label = ""
            if distance < 1:
                label = f"{distance * 1000:.1f} nm"
            elif distance < 100:
                label = f"{distance:.1f} um"
            else:
                label = f"{int(distance)} um"

            self.items["text"].setHtml(f"<span style='color:#6666dd;font-size:16px'>{label}</span>")
            self.items["text"].setScale(1 / scale)

            text_offset = (
                -vector / np.sqrt(vector[0] ** 2 + vector[1] ** 2) * self.size / scale * self.text_offset_factor
            )

            br = self.items["text"].boundingRect()
            text_offset2 = np.array([-br.width() / 2 / scale, -br.height() / scale / 2])
            text_position = position + text_offset + text_offset2

            self.items["text"].setPos(text_position[0], text_position[1])

            self.items["circle"].hide()
            self.items["wline"].hide()
            self.items["hline"].hide()
            self.items["arrow"].show()
            self.items["text"].show()

        super().update()
