from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base


class Cross(base.Tool):
    def __init__(self, view):
        super().__init__(view)

        self.lines = [QtWidgets.QGraphicsLineItem(-100, 0, 100, 0), QtWidgets.QGraphicsLineItem(0, -100, 0, 100)]
        for line in self.lines:
            line.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            line.setParentItem(self)

        self.hide()
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

    def show(self):
        self.is_active = True
        QtWidgets.QGraphicsItemGroup.show(self)
        self.update()

    def hide(self):
        self.is_active = False
        QtWidgets.QGraphicsItemGroup.hide(self)
