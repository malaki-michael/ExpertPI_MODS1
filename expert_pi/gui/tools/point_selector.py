from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base


class PointSelector(base.Tool):
    def __init__(self, view):
        super().__init__(view)

        self.lines = [
            QtWidgets.QGraphicsEllipseItem(-10, -10, 20, 20),
            QtWidgets.QGraphicsLineItem(0, -20, 0, 20),
            QtWidgets.QGraphicsLineItem(-20, 0, 20, 0),
        ]
        for line in self.lines:
            line.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            line.setParentItem(self)

        self.hide()
        self.view.graphics_area.addItem(self)
        self.dragging = False
        self.moved = False

    def view_mouse_pressed(self, event, focused_item=None):
        area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
        self.setPos(area_pos[0], area_pos[1])
        self.view.point_selection_signal.emit(area_pos)
        self.dragging = True
        self.moved = False

    def view_mouse_moved(self, event, focused_item=None):
        if self.dragging:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.setPos(area_pos[0], area_pos[1])
            self.view.point_selection_signal.emit(area_pos)
            self.moved = True

    def view_mouse_released(self, event, focused_item=None):
        if self.dragging and self.moved:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.setPos(area_pos[0], area_pos[1])
            self.view.point_selection_signal.emit(area_pos)
        self.dragging = False
        self.moved = False

    def show(self):
        super().show()
        self.is_active = True
        self.update()

    def hide(self):
        self.is_active = False
        super().hide()

    def update(self, *args, **kwargs):
        # sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        self.setScale(1.0 / scale)
        super().update()
