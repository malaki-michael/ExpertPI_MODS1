import numpy as np
from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base
from expert_pi.gui.tools.graphic_items import DragPoint, ItemGroup


class MaskSelector(base.Tool, ItemGroup):
    def __init__(self, view):
        super().__init__(view)

        self.items = {
            "r1": QtWidgets.QGraphicsEllipseItem(-10, -10, 20, 20),
            "r2": QtWidgets.QGraphicsEllipseItem(-10, -10, 20, 20),
            "r1_point": DragPoint(self.r1_moved),
            "r2_point": DragPoint(self.r2_moved),
            "center": DragPoint(self.center_moved),
        }
        self.segments_lines = []

        self.shape = (512, 512)

        self.center = [0, 0]
        self.radii = [0, 20]
        self.N = 0

        for name, item in self.items.items():
            if name in {"r1", "r2"}:
                pen = QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0)
                pen.setCosmetic(True)
                item.setPen(pen)

            item.setParentItem(self)
            item.hide()

        super().hide()
        self.view.graphics_area.addItem(self)
        self.dragging = False

    def view_mouse_pressed(self, event, focused_item=None):
        if not self.has_item(focused_item):
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.center = area_pos
            self.dragging = True
            self.items["center"].setPos(self.center[0], self.center[1])
            self.update()

    def view_mouse_moved(self, event, focused_item=None):
        if self.dragging:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.center = area_pos
            self.items["center"].setPos(self.center[0], self.center[1])
            self.update()

    def view_mouse_released(self, event, focused_item=None):
        if self.dragging:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.center = area_pos
            self.items["center"].setPos(self.center[0], self.center[1])
            self.update()
            self.view.mask_selector_changed.emit()
        self.dragging = False

    def generate_masks(self):
        fov = self.view.main_image_item.fov  # TODO roi
        rotation = self.view.main_image_item.rotation  # TODO roi
        Rm = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        center = np.dot(Rm, self.center)

        h, w = self.shape
        x = np.linspace(-fov / 2 * w / h, fov / 2 * w / h, num=w)
        y = np.linspace(-fov / 2, fov / 2, num=h)
        X, Y = np.meshgrid(x, y)
        R2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2

        mask = (R2 < np.max(self.radii) ** 2) & (R2 >= np.min(self.radii) ** 2)

        if self.N <= 1:
            if np.min(self.radii) == 0:
                return {"BF": mask}
            else:
                return {"BF": R2 < np.min(self.radii) ** 2, "HAADF": mask}
        else:
            alpha = np.arctan2(Y - center[1], X - center[0]) + np.pi
            results = {}
            for i in range(self.N):
                if np.min(self.radii) == 0:
                    results[f"BF_{i}"] = (
                        mask & (alpha >= 2 * np.pi * i / self.N) & (alpha < 2 * np.pi * (i + 1) / self.N)
                    )
                else:
                    results[f"BF_{i}"] = (
                        (R2 < np.min(self.radii) ** 2)
                        & (alpha >= 2 * np.pi * i / self.N)
                        & (alpha < 2 * np.pi * (i + 1) / self.N)
                    )
                    results[f"HAADF_{i}"] = (
                        mask & (alpha >= 2 * np.pi * i / self.N) & (alpha < 2 * np.pi * (i + 1) / self.N)
                    )
            return results

    def set_segments(self, N):
        N = int(N)
        if N <= 1:
            N = 0
        while len(self.segments_lines) < N:
            self.segments_lines.append(QtWidgets.QGraphicsLineItem(0, 0, 10, 10))
            self.segments_lines[-1].setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            self.segments_lines[-1].setParentItem(self)

        for i in range(len(self.segments_lines)):
            if i < N:
                self.segments_lines[i].show()
            else:
                self.segments_lines[i].hide()
        self.N = N
        self.view.mask_selector_changed.emit()
        self.update()

    def center_moved(self, x, y):
        self.center = [x, y]
        self.view.mask_selector_changed.emit()
        self.update()

    def r1_moved(self, x, y):
        self.radii[0] = np.sqrt((self.center[0] - x) ** 2 + (self.center[1] - y) ** 2)
        self.view.mask_selector_changed.emit()
        self.update()

    def r2_moved(self, x, y):
        self.radii[1] = np.sqrt((self.center[0] - x) ** 2 + (self.center[1] - y) ** 2)
        self.view.mask_selector_changed.emit()
        self.update()

    def update(self, *args, **kwargs):
        self.items["r1_point"].setPos(self.center[0] + self.radii[0], self.center[1])
        self.items["r2_point"].setPos(self.center[0] + self.radii[1], self.center[1])
        self.items["r1"].setRect(
            self.center[0] - self.radii[0], self.center[1] - self.radii[0], 2 * self.radii[0], 2 * self.radii[0]
        )
        self.items["r2"].setRect(
            self.center[0] - self.radii[1], self.center[1] - self.radii[1], 2 * self.radii[1], 2 * self.radii[1]
        )

        for i in range(self.N):
            s = np.sin(2 * np.pi * i / self.N)
            c = np.cos(2 * np.pi * i / self.N)
            self.segments_lines[i].setLine(
                self.center[0],
                self.center[1],
                self.center[0] + c * self.radii[1],
                self.center[1] + s * self.radii[1],
            )

        sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        for name in ["center", "r1_point", "r2_point"]:
            self.items[name].setScale(1.0 / scale)

    def show(self):
        self.is_active = True
        QtWidgets.QGraphicsItemGroup.show(self)
        for item in self.items.values():  # TODO wizard
            item.show()
        self.update()

    def hide(self):
        self.is_active = False
        QtWidgets.QGraphicsItemGroup.hide(self)
