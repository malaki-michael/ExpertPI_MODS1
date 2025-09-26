import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.tools import base
from expert_pi.gui.tools.graphic_items import DragLine, DragPoint, ItemGroup


class RectangleSelector(base.Tool, ItemGroup):
    def __init__(self, view):
        super().__init__(view)
        self.view = view

        self.items = {
            "rect": QtWidgets.QGraphicsRectItem(0, 0, 0, 0),
            "left": DragLine(0, 0, 0, 0, self.left_moved),
            "right": DragLine(0, 0, 0, 0, self.right_moved),
            "top": DragLine(0, 0, 0, 0, self.top_moved),
            "bottom": DragLine(0, 0, 0, 0, self.bottom_moved),
            "top_left": DragPoint(self.point_moved),
            "text": QtWidgets.QGraphicsTextItem(""),
        }

        for name, item in self.items.items():
            if name == "rect":
                item.setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                item.setBrush(QtGui.QColor(0, 0, 0, 0))
            item.setParentItem(self)
            item.hide()

        super().hide()
        self.view.graphics_area.addItem(self)

        self.wizard_step = 0

    def show(self):
        super().show()
        if not self.is_active:
            self.reset_wizard()
            self.is_active = True
        self.update()

    def hide(self):
        if self.is_active:
            self.reset_wizard()
            self.is_active = False
            self.view.selection_changed_signal.emit()
        super().hide()

    def rectangle_active(self):
        return self.wizard_step == 2 and self.is_active

    def get_scan_rectangle(self):
        fov = self.view.main_image_item.fov
        N = self.view.main_image_item.image.shape[0]
        x0, y0, x2, y2 = self.get_rectangle()

        return np.array(
            [
                (min(y0, y2) + fov / 2) / fov * N,
                (min(x0, x2) + fov / 2) / fov * N,
                max(1, abs(y2 - y0) / fov * N),
                max(1, abs(x2 - x0) / fov * N),
            ]
        ).astype("int")

    def set_pixel_size(self, pixel_size):
        self.pixel_size = pixel_size
        # in um

    def reset_wizard(self):
        for item in self.items.values():
            item.hide()
        self.wizard_step = 0

    def view_mouse_pressed(self, event, focused_item=None):
        if not self.has_item(focused_item) or focused_item == self.items["rect"]:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.wizard_step = 0
            self.use_wizard(area_pos[0], area_pos[1], click=True)

    def view_mouse_moved(self, event, focused_item=None):
        if self.wizard_step == 1:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.use_wizard(
                area_pos[0],
                area_pos[1],
                ctrl_modifier=(event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier),
            )

    def view_mouse_released(self, event, focused_item=None):
        area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
        self.use_wizard(
            area_pos[0],
            area_pos[1],
            click=True,
            ctrl_modifier=(event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier),
        )
        self.view.selection_changed_signal.emit()

    def use_wizard(self, x, y, click=False, ctrl_modifier=False):
        if self.wizard_step == 0:
            if click:
                self.items["top_left"].setPos(x, y)
                self.items["left"].setLine(x, y, x, y)
                self.items["right"].setLine(x, y, x, y)
                self.items["top"].setLine(x, y, x, y)
                self.items["bottom"].setLine(x, y, x, y)
                self.items["rect"].setRect(x, y, 0, 0)
                for name, item in self.items.items():
                    item.show()
                self.wizard_step = 1
                return
            else:
                self.items["top_left"].setPos(x, y)
                self.items["top_left"].show()
                return
        elif self.wizard_step == 1:
            x0, y0, x2, y2 = self.get_rectangle()

            if ctrl_modifier:
                d = max(abs(x0 - x), abs(y0 - y))
                self.set_rectangle(x0, y0, x0 + d, y0 + d)
            else:
                self.set_rectangle(x0, y0, x, y)
            if click:
                self.wizard_step = 2

    def get_rectangle(self):
        rect = self.items["rect"].rect()
        x0, y0, x2, y2 = rect.topLeft().x(), rect.topLeft().y(), rect.bottomRight().x(), rect.bottomRight().y()
        return x0, y0, x2, y2

    def set_rectangle(self, x0, y0, x2, y2):
        fov = self.view.main_image_item.fov
        x0 = np.clip(x0, -fov / 2, fov / 2)
        y0 = np.clip(y0, -fov / 2, fov / 2)
        x2 = np.clip(x2, -fov / 2, fov / 2)
        y2 = np.clip(y2, -fov / 2, fov / 2)

        self.items["top_left"].setPos(x0, y0)
        self.items["left"].setLine(x0, y0, x0, y2)
        self.items["right"].setLine(x2, y0, x2, y2)
        self.items["top"].setLine(x0, y0, x2, y0)
        self.items["bottom"].setLine(x0, y2, x2, y2)
        self.items["rect"].setRect(x0, y0, x2 - x0, y2 - y0)

        # text = self.items["text"]
        # w, h = x2 - x0, y2 - y0
        #
        # if w > 0.1 or h > 0.1:
        #     label = f"{w:.1f}x{h:.1f} um"
        # elif w > 1e-2 or h > 1 - 2:
        #     label = f"{w*1000:.0f}x{h*1000:.0f} nm"
        # elif w > 1e-4 or h > 1 - 4:
        #     label = f"{w*1000:.1f}x{h*1000:.1f} nm"
        #
        # text.setHtml(f"<span style='color:#0000ff;font-size:12px;font-weight:bold'>{label}</span>")
        # r = text.boundingRect()
        # text.setPos(x0, y0 - r.center().y()*2*text.scale())

    def point_moved(self, x, y):
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x, y, x2 - x0 + x, y2 - y0 + y)

    def left_moved(self, x, y):
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x, y0, x2, y2)

    def right_moved(self, x, y):
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0, y0, x, y2)

    def top_moved(self, x, y):
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0, y, x2, y2)

    def bottom_moved(self, x, y):
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0, y0, x2, y)

    def update(self, *args, **kwargs):
        sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        self.items["top_left"].setScale(1.0 / scale)
        for name in ["top", "bottom", "right", "left"]:
            self.items[name].set_scale(1.0 / scale)

        # x0, y0, x2, y2 = self.get_rectangle()
        # text = self.items["text"]
        # r = text.boundingRect()
        # text.setPos(x0, y0 - r.center().y()*2*text.scale())
        # self.items["text"].setScale(1./scale)
