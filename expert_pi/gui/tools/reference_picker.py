import numpy as np
from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base
from expert_pi.gui.tools.graphic_items import DragLine, ItemGroup


class ReferencePicker(base.Tool, ItemGroup):
    def __init__(self, view=None):
        super().__init__(view)
        self.cross_overlap = 0.05

        self.items = {
            "left": DragLine(0, 0, 0, 0, self.edge_moved),
            "right": DragLine(0, 0, 0, 0, self.edge_moved),
            "top": DragLine(0, 0, 0, 0, self.edge_moved),
            "bottom": DragLine(0, 0, 0, 0, self.edge_moved),
            "left_limit": DragLine(0, 0, 0, 0, self.left_limit_moved),
            "right_limit": DragLine(0, 0, 0, 0, self.right_limit_moved),
            "top_limit": DragLine(0, 0, 0, 0, self.top_limit_moved),
            "bottom_limit": DragLine(0, 0, 0, 0, self.bottom_limit_moved),
            "vertical": DragLine(0, 0, 0, 0, self.cross_moved),
            "horizontal": DragLine(0, 0, 0, 0, self.cross_moved),
        }

        self.items["vertical"].linked_hover = self.items["horizontal"]
        self.items["horizontal"].linked_hover = self.items["vertical"]

        for name, item in self.items.items():
            if name in ["rect"]:
                item.setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                item.setBrush(QtGui.QColor(0, 0, 0, 0))

            item.setParentItem(self)
            item.hide()
            item.setZValue(2)

        self.center = np.zeros(2)
        self.fov = 0
        self.limit_rectangle = None

        super().hide()
        if self.view is not None:
            self.view.graphics_area.addItem(self)

        self.wizard_step = 0

        self.wizard_finished_callback = None

        self.min_fov = 0.0001  # 1Angstrom
        self.N = 512  # required number of pixels
        self.N_max = 2**14

        self.fov_items = {}
        self.fov_max = 40
        self.center_edge_limit = 1

        self.fovs = np.array([])
        self.rectangles = np.array([])
        #
        # get_reference_fovs(self, preferred_fov, fov_max=None)

    def clear_fovs(self, keys):
        for key in keys:
            self.fov_items[key].setParentItem(None)
            # self.fov_items[key].prepareGeometryChange()
            self.view.scene().removeItem(self.fov_items[key])
            del self.fov_items[key]

    def update_fovs(self, fovs, rectangles):
        to_clear = []
        for key in self.fov_items:
            if key not in fovs:
                to_clear.append(key)
        self.clear_fovs(to_clear)

        for i, fov2 in enumerate(fovs):
            if fov2 not in self.fov_items:
                self.fov_items[fov2] = QtWidgets.QGraphicsRectItem(0, 0, 0, 0)
                self.fov_items[fov2].setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                self.fov_items[fov2].setBrush(QtGui.QColor(0, 0, 0, 0))

                self.fov_items[fov2].setParentItem(self)
                self.fov_items[fov2].setZValue(1)
            self.fov_items[fov2].show()

            self.fov_items[fov2].setRect(
                rectangles[i][0],
                rectangles[i][1],
                rectangles[i][2] - rectangles[i][0],
                rectangles[i][3] - rectangles[i][1],
            )

    def rectangle_active(self):
        return self.wizard_step is None and self.isVisible()

    def reset_wizard(self):
        for item in self.items.values():
            item.hide()
            item.interactivity_enabled = False
        for item in self.fov_items.values():
            item.hide()

        self.wizard_step = 0
        self.limit_rectangle = None

    def use_wizard(self, x, y, click=False):
        if self.wizard_step == 0:
            if click:
                self.set_rectangle(np.array([x, y]), 0)

                for name, item in self.items.items():
                    item.show()
                self.wizard_step += 1
                return
            else:
                self.set_rectangle(np.array([x, y]), 0)
                for name, item in self.items.items():
                    item.show()
                self.items["vertical"].show()
                self.items["horizontal"].show()

                return
        elif self.wizard_step == 1:
            r = max(abs(x - self.center[0]), abs(y - self.center[1]))
            r = max(r, self.min_fov)

            self.set_rectangle(self.center, 2 * r)

            if click:
                self.wizard_step = None
                for item in self.items.values():
                    item.interactivity_enabled = True
                if self.wizard_finished_callback is not None:
                    self.wizard_finished_callback()

    def set_rectangle(self, center, fov, limit_rectangle=None):
        if limit_rectangle is None:
            if self.limit_rectangle is None:
                self.limit_rectangle = [-self.fov_max / 2, -self.fov_max / 2, self.fov_max / 2, self.fov_max / 2]
            limit_rectangle = self.limit_rectangle

        # limit center:
        center = np.clip(
            center, -self.fov_max / 2 + self.center_edge_limit, +self.fov_max / 2 - self.center_edge_limit
        )  # limit by fov_max
        center = np.clip(center, limit_rectangle[:2], limit_rectangle[2:])  # limit by outer rectangle

        self.center = center

        # limit fov
        fov = max(self.min_fov, fov)  # limit by min fov to prevent zer fovs
        fov = max(fov, 2 * np.max(np.abs(self.center)) * self.N / (self.N_max - self.N))  # limit by pixels

        if fov > self.fov_max:
            fov = self.fov_max
            self.fovs = np.array([fov])
        else:
            base = 1.5
            log_fov = np.log(fov) / np.log(base)
            log_fov_limit = np.log(self.fov_max) / np.log(base)
            num = max(1, int(np.round(log_fov_limit - log_fov)))
            self.fovs = np.logspace(log_fov, log_fov_limit, num=num, base=base)[::-1]

        self.fov = fov

        self.rectangles = []
        for fov2 in self.fovs:
            vec = -self.fov_max / 2 - self.center + fov / 2
            alpha = (fov2 - fov) / (self.fov_max - fov)

            rectangle = [
                self.center[0] - fov / 2 + alpha * vec[0],
                self.center[1] - fov / 2 + alpha * vec[1],
                self.center[0] - fov / 2 + alpha * vec[0] + fov2,
                self.center[1] - fov / 2 + alpha * vec[1] + fov2,
            ]

            rectangle[:2] = np.clip(rectangle[:2], limit_rectangle[:2], limit_rectangle[2:])
            rectangle[2:] = np.clip(rectangle[2:], limit_rectangle[:2], limit_rectangle[2:])
            self.rectangles.append(rectangle)

        self.rectangles, indices = np.unique(self.rectangles, axis=0, return_index=True)
        self.fovs = self.fovs[indices]

        self.rectangles = np.array(self.rectangles)
        self.update_fovs(self.fovs, self.rectangles)

        rmin = self.rectangles[-1]
        rmax = self.rectangles[0]

        self.items["left"].setLine(rmin[0], rmin[1], rmin[0], rmin[3])
        self.items["right"].setLine(rmin[2], rmin[1], rmin[2], rmin[3])

        self.items["top"].setLine(rmin[0], rmin[1], rmin[2], rmin[1])
        self.items["bottom"].setLine(rmin[0], rmin[3], rmin[2], rmin[3])

        self.items["left_limit"].setLine(rmax[0], rmax[1], rmax[0], rmax[3])
        self.items["right_limit"].setLine(rmax[2], rmax[1], rmax[2], rmax[3])

        self.items["top_limit"].setLine(rmax[0], rmax[1], rmax[2], rmax[1])
        self.items["bottom_limit"].setLine(rmax[0], rmax[3], rmax[2], rmax[3])

        self.items["vertical"].setLine(
            center[0] - self.fov_max / 2 * (1 + self.cross_overlap),
            center[1],
            center[0] + self.fov_max / 2 * (1 + self.cross_overlap),
            center[1],
        )
        self.items["horizontal"].setLine(
            center[0],
            center[1] - self.fov_max / 2 * (1 + self.cross_overlap),
            center[0],
            center[1] + self.fov_max / 2 * (1 + self.cross_overlap),
        )

        # self.get_fovs_rectangles()  # DEBUG TEST TODO remove

    def get_fovs_rectangles(self, N=512):
        fovs_total = 2 * np.max(np.abs(self.rectangles), axis=1)
        total_pixels = np.round(N * fovs_total / self.fovs).astype("int")

        rectangles = np.round(
            (
                (self.rectangles + fovs_total.reshape(-1, 1) / 2)
                / fovs_total.reshape(-1, 1)
                * total_pixels.reshape(-1, 1)
            )
        ).astype("int")

        rectangles[:, 2:] -= rectangles[:, :2]
        # fix over 512 due rounding:
        rectangles[:, 2:] = np.minimum(self.N, rectangles[:, 2:])

        # just for debug:
        if np.any(rectangles[:, :2] + rectangles[:, 2:] > total_pixels.reshape(-1, 1)) or np.any(
            rectangles.reshape(-1, 1) < 0
        ):
            print("indices problem")
            print(total_pixels)
            print(rectangles)
        if np.any(total_pixels > 2**14):
            print("too big amount of pixels", total_pixels)

        rectangles2 = rectangles * 1
        # flip from xy to ij:
        rectangles2[:, 0] = rectangles[:, 1]
        rectangles2[:, 1] = rectangles[:, 0]
        rectangles2[:, 2] = rectangles[:, 3]
        rectangles2[:, 3] = rectangles[:, 2]

        return (
            fovs_total,
            total_pixels,
            rectangles2,
            self.fovs,
            0.5 * (self.rectangles[:, :2] + self.rectangles[:, 2:]),
            self.rectangles,
        )

    def get_center(self):
        return self.center

    def cross_moved(self, x, y):
        self.set_rectangle(np.array([x, y]), self.fov)

    def edge_moved(self, x, y):
        r = max(abs(self.center[0] - x), abs(self.center[1] - y))
        self.set_rectangle(self.center, 2 * r)

    def left_limit_moved(self, x, y):
        limit_rectangle = self.limit_rectangle
        limit_rectangle[0] = max(x, -self.fov_max / 2)
        self.set_rectangle(self.center, self.fov, limit_rectangle=limit_rectangle)

    def right_limit_moved(self, x, y):
        limit_rectangle = self.limit_rectangle
        limit_rectangle[2] = min(x, self.fov_max / 2)
        self.set_rectangle(self.center, self.fov, limit_rectangle=limit_rectangle)

    def top_limit_moved(self, x, y):
        limit_rectangle = self.limit_rectangle
        limit_rectangle[1] = max(y, -self.fov_max / 2)
        self.set_rectangle(self.center, self.fov, limit_rectangle=limit_rectangle)

    def bottom_limit_moved(self, x, y):
        limit_rectangle = self.limit_rectangle
        limit_rectangle[3] = min(y, self.fov_max / 2)
        self.set_rectangle(self.center, self.fov, limit_rectangle=limit_rectangle)

    def update(self, *args, **kwargs):
        if self.view is not None:
            sr = self.view.sceneRect()
            scale = self.view.graphics_area.scale()
            for name in [
                "top",
                "bottom",
                "right",
                "left",
                "top_limit",
                "bottom_limit",
                "right_limit",
                "left_limit",
                "vertical",
                "horizontal",
            ]:
                self.items[name].set_scale(1.0 / scale)

    def view_mouse_pressed(self, event, focused_item=None):
        if self.is_active:
            if (self.wizard_step is not None and self.wizard_step == 0) or not self.has_item(focused_item):
                self.reset_wizard()
                area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
                self.use_wizard(area_pos[0], area_pos[1], click=True)

    def view_mouse_moved(self, event, focused_item=None):
        if self.is_active:
            if self.wizard_step is not None:
                area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
                self.use_wizard(
                    area_pos[0],
                    area_pos[1],
                )

    def view_mouse_released(self, event, focused_item=None):
        if self.is_active:
            if (self.wizard_step is not None and self.wizard_step == 1) or not self.has_item(focused_item):
                area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
                self.use_wizard(
                    area_pos[0],
                    area_pos[1],
                    click=True,
                )
