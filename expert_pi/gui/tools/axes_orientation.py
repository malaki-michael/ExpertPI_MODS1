from PySide6 import QtGui, QtWidgets


class AxesOrientation(QtWidgets.QGraphicsItemGroup):
    def __init__(self, view):
        super().__init__()

        self.view = view

        self.f = 1 / 30
        self.m = 0.5
        f, m = self.f, self.m

        x_pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 0)
        y_pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 255), 0)

        self.lines = [
            QtWidgets.QGraphicsLineItem(m * (1 - f), -f * m, m, 0),
            QtWidgets.QGraphicsLineItem(0, 0, m, 0),
            QtWidgets.QGraphicsLineItem(m * (1 - f), f * m, m, 0),
            QtWidgets.QGraphicsLineItem(-f * m, m * (1 - f), 0, m),
            QtWidgets.QGraphicsLineItem(0, 0, 0, m),
            QtWidgets.QGraphicsLineItem(f * m, m * (1 - f), 0, m),
        ]

        for line in self.lines:
            line.setParentItem(self)

        for line in self.lines[:3]:
            line.setPen(x_pen)
        for line in self.lines[3:]:
            line.setPen(y_pen)

        self.hide()
        self.view.graphics_area.addItem(self)

    def update(self, *args, **kwargs):
        f, m = self.f, self.m
        m *= self.view.image_item.fov
        self.lines[0].setLine(m * (1 - f), -f * m, m, 0)
        self.lines[1].setLine(0, 0, m, 0)
        self.lines[2].setLine(m * (1 - f), f * m, m, 0)
        self.lines[3].setLine(-f * m, -m * (1 - f), 0, -m)
        self.lines[4].setLine(0, 0, 0, -m)
        self.lines[5].setLine(f * m, -m * (1 - f), 0, -m)
