import numpy as np
from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools.graphic_items import DragLine, ItemGroup


class AlignmentRectangle(QtWidgets.QGraphicsItemGroup, ItemGroup):
    def __init__(self, view=None):
        super().__init__()
        self.view = view
        self.cross_overlap = 10

        self.items = {
            "rect": QtWidgets.QGraphicsRectItem(0, 0, 0, 0),
            "left": DragLine(0, 0, 0, 0, self.left_moved),
            "right": DragLine(0, 0, 0, 0, self.right_moved),
            "top": DragLine(0, 0, 0, 0, self.top_moved),
            "bottom": DragLine(0, 0, 0, 0, self.bottom_moved),
            "vertical": DragLine(0, 0, 0, 0, self.cross_moved),
            "horizontal": DragLine(0, 0, 0, 0, self.cross_moved),
            "fit_vertical": QtWidgets.QGraphicsLineItem(0, 0, 0, 0),
            "fit_horizontal": QtWidgets.QGraphicsLineItem(0, 0, 0, 0),
        }

        self.items["vertical"].linked_hover = self.items["horizontal"]
        self.items["horizontal"].linked_hover = self.items["vertical"]

        for name, item in self.items.items():
            if name in {"rect"}:
                item.setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                item.setBrush(QtGui.QColor(0, 0, 0, 0))

            if name in {"fit_vertical", "fit_horizontal"}:
                item.setPen(QtGui.QPen(QtGui.QColor(0, 255, 100, 255), 1))

            item.setParentItem(self)
            item.hide()

        self.center = np.zeros(2)
        self.size = np.zeros(2)

        self.ref_center = np.zeros(2)
        self.ref_size = np.zeros(2)
        self.ref_index = 0

        self.hint_center = np.zeros(2)

        super().hide()
        if self.view is not None:
            self.view.graphics_area.addItem(self)

        self.wizard_step = 0

        self.interactivity_enabled = True
        self.wizard_finished_callback = None
        self.alignment_control = None

    def set_interactive(self, enabled):
        self.interactivity_enabled = enabled
        for item in self.items.values():
            if hasattr(item, "interactivity_enabled"):
                item.interactivity_enabled = enabled

    def hide(self):
        if self.rectangle_active():
            super().hide()
            if self.view is not None:
                self.view.main_window.scanning.rescan()
        super().hide()

    def rectangle_active(self):
        return self.wizard_step is None and self.isVisible()

    def reset_wizard(self):
        for item in self.items.values():
            item.hide()
        self.wizard_step = 0

    def use_wizard(self, x, y, click=False):
        if not self.interactivity_enabled:
            return
        if self.wizard_step == 0:
            if click:
                self.set_rectangle(x, y, x, y)

                for name, item in self.items.items():
                    if name in {"fit_vertical", "fit_horizontal"}:
                        item.hide()
                    else:
                        item.show()
                self.wizard_step += 1
                return
            else:
                self.set_rectangle(x, y, x, y)
                self.items["vertical"].show()
                self.items["horizontal"].show()

                return
        elif self.wizard_step == 1:
            self.set_rectangle(self.center[0] * 2 - x, self.center[1] * 2 - y, x, y)
            if click:
                self.wizard_step = None
                if self.wizard_finished_callback is not None:
                    self.wizard_finished_callback()
                self.ref_center = self.center * 1
                self.ref_size = self.size * 1
                if self.alignment_control is not None:
                    self.ref_index = self.alignment_control.get_current_index()

    def set_rectangle(self, x0, y0, x1, y1):
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.size = np.array([abs(x0 - x1), abs(y0 - y1)])
        center = self.center
        size = self.size
        self.items["left"].setLine(
            center[0] - size[0] / 2, center[1] - size[1] / 2, center[0] - size[0] / 2, center[1] + size[1] / 2
        )
        self.items["right"].setLine(
            center[0] + size[0] / 2, center[1] - size[1] / 2, center[0] + size[0] / 2, center[1] + size[1] / 2
        )

        self.items["top"].setLine(
            center[0] - size[0] / 2, center[1] - size[1] / 2, center[0] + size[0] / 2, center[1] - size[1] / 2
        )
        self.items["bottom"].setLine(
            center[0] - size[0] / 2, center[1] + size[1] / 2, center[0] + size[0] / 2, center[1] + size[1] / 2
        )

        self.items["rect"].setRect(center[0] - size[0] / 2, center[1] - size[1] / 2, size[0], size[1])
        self.items["vertical"].setLine(
            center[0] - size[0] / 2 - self.cross_overlap,
            center[1],
            center[0] + size[0] / 2 + self.cross_overlap,
            center[1],
        )
        self.items["horizontal"].setLine(
            center[0],
            center[1] - size[1] / 2 - self.cross_overlap,
            center[0],
            center[1] + size[1] / 2 + self.cross_overlap,
        )

    def get_rectangle(self):
        rect = self.items["rect"].rect()
        x0, y0, x2, y2 = rect.topLeft().x(), rect.topLeft().y(), rect.bottomRight().x(), rect.bottomRight().y()
        return x0, y0, x2, y2

    def get_ref_rectangle(self):
        x0, y0 = self.ref_center[0] - self.ref_size[0] / 2, self.ref_center[1] - self.ref_size[1] / 2
        x2, y2 = self.ref_center[0] + self.ref_size[0] / 2, self.ref_center[1] + self.ref_size[1] / 2
        return x0, y0, x2, y2

    def get_center(self):
        return self.center

    def cross_moved(self, x, y):
        if not self.interactivity_enabled:
            return
        x0, y0, x2, y2 = self.get_rectangle()
        dx = x2 - x0
        dy = y2 - y0
        self.set_rectangle(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2)

    def left_moved(self, x, y):
        if not self.interactivity_enabled:
            return
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0 - (x0 - x), y0, x2 + (x0 - x), y2)

    def right_moved(self, x, y):
        if not self.interactivity_enabled:
            return
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0 + (x2 - x), y0, x2 - (x2 - x), y2)

    def top_moved(self, x, y):
        if not self.interactivity_enabled:
            return
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0, y0 + (y2 - y), x2, y2 - (y2 - y))

    def bottom_moved(self, x, y):
        if not self.interactivity_enabled:
            return
        x0, y0, x2, y2 = self.get_rectangle()
        self.set_rectangle(x0, y0 - (y0 - y), x2, y2 + (y0 - y))

    def update(self, *args, **kwargs):
        if self.view is not None:
            sr = self.view.sceneRect()
            scale = self.view.graphics_area.scale()
            for name in ["top", "bottom", "right", "left", "vertical", "horizontal"]:
                self.items[name].set_scale(1.0 / scale)

        # x0, y0, x2, y2 = self.get_rectangle()
        # text = self.items["text"]
        # r = text.boundingRect()
        # text.setPos(x0, y0 - r.center().y()*2*text.scale())
        # self.items["text"].setScale(1./scale)

    def position_clicked(self, x, y, update_ref=False):
        if self.alignment_control is not None:
            if update_ref:
                self.ref_center = np.array([x, y])
                self.ref_size = self.size * 1
                self.ref_index = self.alignment_control.get_current_index()
            self.alignment_control.shift_actual_image(x - self.center[0], y - self.center[1])

    def previous_step(self):
        if self.alignment_control is not None:
            self.alignment_control.previous_step()

    def next_step(self):
        if self.alignment_control is not None:
            self.alignment_control.next_step()

    def accept_hint_position(self):
        if self.alignment_control is not None:
            self.alignment_control.accept_hint_position()

    def update_index(self, i):
        if self.isVisible():
            if self.alignment_control is not None:
                inp = np.array([
                    [self.ref_center[0] - self.ref_size[0] / 2, self.ref_center[0] + self.ref_size[0] / 2],
                    [self.ref_center[1] - self.ref_size[1] / 2, self.ref_center[1] + self.ref_size[1] / 2],
                ])

                r = self.alignment_control.transform(self.ref_index, i, inp)

                self.set_rectangle(r[0, 0], r[1, 0], r[0, 1], r[1, 1])

                if self.alignment_control.align_hinting:
                    hint = self.alignment_control.hint_position()
                    self.set_fit_marker_position(self.center[0] + hint[0], self.center[1] + hint[1])
                    self.show_fit_marker()

    def set_fit_marker_position(self, x, y):
        self.hint_center = np.array([x, y])
        self.items["fit_vertical"].setLine(x - self.cross_overlap, y, x + self.cross_overlap, y)
        self.items["fit_horizontal"].setLine(x, y - self.cross_overlap, x, y + self.cross_overlap)

    def show_fit_marker(self):
        self.items["fit_vertical"].show()
        self.items["fit_horizontal"].show()

    def hide_fit_marker(self):
        self.items["fit_vertical"].hide()
        self.items["fit_horizontal"].hide()
