import numpy as np
from PySide6 import QtGui, QtWidgets


class CrossSymbol(QtWidgets.QGraphicsItemGroup):
    def __init__(self, view, size=3):
        super().__init__()

        self.view = view

        self.lines = [QtWidgets.QGraphicsLineItem(-size, 0, size, 0), QtWidgets.QGraphicsLineItem(0, -size, 0, size)]
        for line in self.lines:
            line.setPen(QtGui.QPen(QtGui.QColor(0, 200, 100, 100), 0))
            line.setParentItem(self)

        self.view.graphics_area.addItem(self)

    def update(self, *args, **kwargs):
        sr = self.view.sceneRect()
        size = max(sr.width(), sr.height())
        scale = self.view.graphics_area.scale()
        if scale != 0:
            fov = size / scale
        else:
            fov = 0
        self.lines[0].setLine(2 * -fov / 2.0, 0, 2 * fov / 2.0, 0)
        self.lines[1].setLine(0, 2 * -fov / 2.0, 0, 2 * fov / 2.0)

    def setPen(self, pen):
        for line in self.lines:
            line.setPen(pen)


class DiffractionPattern(QtWidgets.QGraphicsItemGroup):
    def __init__(self, view):
        super().__init__()
        self.view = view

        self.hide()
        self.view.graphics_area.addItem(self)

        self.model_spots = []  # [x,y,size]
        self.model_items = []
        self.fitted_items = []

    def clear(self, model=True, fit=True):
        if model:
            while len(self.model_items) > 0:
                self.model_items[0].setParentItem(None)
                self.view.scene().removeItem(self.model_items[0])
                del self.model_items[0]
            self.model_items = []
        if fit:
            while len(self.fitted_items) > 0:
                self.fitted_items[0].setParentItem(None)
                self.view.scene().removeItem(self.fitted_items[0])
                del self.fitted_items[0]
            self.fitted_items = []

    def generate_model(self, model_spots):
        self.model_spots = model_spots

        self.clear(fit=False)
        for x, y, d in self.model_spots:
            d = d / 2
            self.model_items.append(QtWidgets.QGraphicsEllipseItem(x - d / 2, y - d / 2, d, d))  # flip y axis
            self.model_items[-1].setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            self.model_items[-1].setParentItem(self)

    def generate_fit(self, x, y, s, angular_error):
        center_index = np.argmin(x**2 + y**2)

        self.clear(model=False)

        green_color = QtGui.QColor(0, 200, 0, 255)
        green_color_alpha = QtGui.QColor(0, 200, 0, 30)
        orange_color = QtGui.QColor(255, 100, 0, 255)
        orange_color_alpha = QtGui.QColor(255, 100, 0, 30)

        for i in range(len(x)):
            d = s[i] * 2
            symbol = QtWidgets.QGraphicsEllipseItem(-d / 2, -d / 2, d, d)
            symbol.setPen(QtGui.QPen(green_color, 0))
            symbol.setBrush(green_color)
            if i == center_index:
                symbol.setBrush(orange_color)

            self.fitted_items.append(symbol)
            self.fitted_items[-1].setPos(x[i], y[i])
            self.fitted_items[-1].setParentItem(self)

            # non highlighted:
            d = angular_error * 2
            symbol = QtWidgets.QGraphicsEllipseItem(-d / 2, -d / 2, d, d)
            if i == center_index:
                symbol.setPen(QtGui.QPen(orange_color_alpha, 0))
            else:
                symbol.setPen(QtGui.QPen(green_color_alpha, 0))
            self.fitted_items.append(symbol)
            self.fitted_items[-1].setPos(x[i], y[i])
            self.fitted_items[-1].setParentItem(self)

        self.setRotation(self.view.image_item.rotation / np.pi * 180)

    # def update(self, *args, **kwargs):
    #     sr = self.view.sceneRect()
    #     scale = self.view.graphics_area.scale()
    #     for item in self.items:
    #         item.set_scale(1./scale)
