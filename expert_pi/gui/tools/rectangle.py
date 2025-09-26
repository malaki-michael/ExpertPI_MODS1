from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base


class Rectangle(base.Tool):
    def __init__(self, view):
        super().__init__(view)
        self.view = view

        self.items = {"rect": QtWidgets.QGraphicsRectItem(0, 0, 0, 0)}
        self.lines = []

        for item in self.items.values():
            item.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            item.setParentItem(self)

        self.hide()
        self.view.graphics_area.addItem(self)

    def set_rectangle(self, x, y, w, h, w_count, h_count):
        n = int(w_count - 1 + h_count - 1)
        while len(self.lines) < n:
            self.lines.append(QtWidgets.QGraphicsLineItem(0, 0, 0, 0))
            self.lines[-1].setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 100), 0))
            self.lines[-1].setParentItem(self)
        for i in range(len(self.lines)):
            if i < n:
                self.show()
            else:
                self.hide()

        self.items["rect"].setRect(x, y, w, h)

        for j in range(w_count - 1):
            self.lines[j].setLine(x + (j + 1) * w / w_count, y, x + (j + 1) * w / w_count, y + h)

        for j in range(h_count - 1):
            self.lines[j + w_count - 1].setLine(x, y + (j + 1) * h / h_count, x + w, y + (j + 1) * h / h_count)

    def update(self, *args, **kwargs):
        pass
