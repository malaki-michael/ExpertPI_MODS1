import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.tools import base
from expert_pi.gui.tools.graphic_items import DragLine, DragPoint, ItemGroup


class DistanceTool(base.Tool, ItemGroup):
    def __init__(self, view, diffraction=False):
        super().__init__(view)
        self.view = view
        self.diffraction = diffraction

        self.default_size_factor = 0.2  # profile size factor during wizard
        self.size = 0  # profile size um

        self.items = {
            "profile_rect": QtWidgets.QGraphicsRectItem(0, 0, self.size, self.size),
            "profile_edge1": DragLine(0, 0, 0, 0, self.size_changed),
            "profile_edge2": DragLine(0, 0, 0, 0, self.size_changed),
            "profile": QtWidgets.QGraphicsPathItem(),
            "line": QtWidgets.QGraphicsLineItem(0, 0, 0, 0),
            "start": DragPoint(self.start_point_moved),
            "end": DragPoint(self.end_point_moved),
            "text": QtWidgets.QGraphicsTextItem(""),
        }

        for name, item in self.items.items():
            if name == "profile_rect":
                item.setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                item.setBrush(QtGui.QColor(0, 0, 255, 150))
            elif name in {"profile_edge1", "profile_edge2", "start", "end", "text"}:
                pass
            else:
                item.setPen(QtGui.QPen(QtGui.QColor(200, 200, 255, 255), 0))
            item.setParentItem(self)
            item.hide()

        self.hide()
        self.view.graphics_area.addItem(self)

        self.wizard_step = 0

    def show(self):
        super().show()
        self.is_active = True
        self.update()

    def hide(self):
        self.is_active = False
        super().hide()

    def reset_wizard(self):
        for item in self.items.values():
            item.hide()
        self.wizard_step = 0

    def view_mouse_pressed(self, event, focused_item=None):
        if not self.has_item(focused_item):
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.wizard_step = 0
            self.use_wizard(area_pos[0], area_pos[1], click=True)

    def view_mouse_moved(self, event, focused_item=None):
        if self.wizard_step == 1:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.use_wizard(
                area_pos[0],
                area_pos[1],
            )

    def view_mouse_released(self, event, focused_item=None):
        area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
        self.use_wizard(area_pos[0], area_pos[1], click=True)

    def use_wizard(self, x, y, click=False):
        if self.wizard_step == 0:
            if click:
                self.items["start"].setPos(x, y)
                self.items["end"].setPos(x, y)
                self.update_profile_rect()
                for item in self.items.values():
                    item.show()
                self.wizard_step += 1
                return
            else:
                self.items["start"].setPos(x, y)
                self.items["start"].show()
                return
        elif self.wizard_step == 1:
            self.items["end"].setPos(x, y)
            self.update_profile_rect()
            if click:
                self.wizard_step = 2

    def update_profile_rect(self, size=None):
        start = self.items["start"].pos()
        start = np.array([start.x(), start.y()])
        end = self.items["end"].pos()
        end = np.array([end.x(), end.y()])
        v = end - start
        v_abs = np.sqrt(v[0] ** 2 + v[1] ** 2)
        if size is None:
            size = self.default_size_factor * v_abs
            self.size = size
        if v_abs == 0:
            v_t = np.array([0, 0])
        else:
            v_t = np.array([-v[1], v[0]]) / v_abs * size / 2
        self.items["line"].setLine(start[0], start[1], end[0], end[1])
        self.items["profile_rect"].setRect(start[0] + v_t[0], start[1] + v_t[1], v_abs, -size)
        self.items["profile_rect"].setTransformOriginPoint(start[0] + v_t[0], start[1] + v_t[1])
        self.items["profile_rect"].setRotation(np.arctan2(v[1], v[0]) / np.pi * 180)

        self.items["profile_edge1"].setLine(start[0] + v_t[0], start[1] + v_t[1], end[0] + v_t[0], end[1] + v_t[1])
        self.items["profile_edge2"].setLine(start[0] - v_t[0], start[1] - v_t[1], end[0] - v_t[0], end[1] - v_t[1])

        text = self.items["text"]
        r = text.boundingRect()
        text.setPos(
            start[0] + v[0] / 2 - r.center().x() * text.scale(), start[1] + v[1] / 2 - r.center().y() * 2 * text.scale()
        )

        if self.diffraction:
            label = f"{v_abs:6.2f} mrad"
        elif v_abs < 1:
            label = f"{v_abs * 1000:6.2f} nm"
        else:
            label = f"{v_abs:6.2f} um"

        text.setHtml(f"<span style='color:#aaaaff;font-size:16px'>{label}</span>")

        y, _, _ = self.calculate_profile()
        self.plot_profile(y)

    def start_point_moved(self, x, y):
        self.items["start"].setPos(x, y)
        self.update_profile_rect(size=self.size)

    def end_point_moved(self, x, y):
        self.items["end"].setPos(x, y)
        self.update_profile_rect(size=self.size)

    def size_changed(self, x, y):
        a_pos = [x, y]

        start = self.items["start"].pos()
        start = np.array([start.x(), start.y()])
        end = self.items["end"].pos()
        end = np.array([end.x(), end.y()])

        v = end - start
        v_abs = np.sqrt(v[0] ** 2 + v[1] ** 2)
        v_t = np.array([-v[1], v[0]]) / v_abs

        M = np.array([[v[0], v_t[0]], [v[1], v_t[1]]])
        xy = np.array([[a_pos[0] - start[0]], [a_pos[1] - start[1]]])

        result = np.linalg.solve(M, xy).flatten()

        self.size = np.abs(result[1]) * 2
        self.update_profile_rect(self.size)

    def update(self, *args, **kwargs):
        # sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        self.items["start"].setScale(1.0 / scale)
        self.items["end"].setScale(1.0 / scale)
        self.items["text"].setScale(1.0 / scale)

        self.items["profile_edge1"].set_scale(1.0 / scale)
        self.items["profile_edge2"].set_scale(1.0 / scale)

    def calculate_profile(self):
        line1 = self.items["profile_edge1"].line()
        line2 = self.items["profile_edge2"].line()

        points = np.array([
            [line1.x1(), line1.y1()],
            [line2.x1(), line2.y1()],
            [line2.x2(), line2.y2()],
            [line1.x2(), line1.y2()],
        ])

        n = self.view.main_image_item.image.shape[0]
        fov = self.view.main_image_item.fov

        points_px = (points + fov / 2) / fov * n

        rect = cv2.minAreaRect(points_px.astype("float32"))
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(self.view.main_image_item.image, M, (width, height))

        angle = np.arctan2(points[-1, 1] - points[0, 1], points[-1, 0] - points[0, 0])

        if angle < -np.pi / 2:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        elif angle > np.pi / 2:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle <= 0:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return np.sum(warped, axis=0), warped, angle

    def plot_profile(self, y):
        x = np.linspace(0 + 0.5 / len(y), 1 - 0.5 / len(y), num=len(y))
        y_norm = y - np.min(y)
        if np.max(y_norm) != 0:
            y_norm /= np.max(y_norm)

        line0 = self.items["line"].line()
        line1 = self.items["profile_edge1"].line()
        line2 = self.items["profile_edge2"].line()

        s = np.array([line2.x1(), line2.y1()])
        xd = np.array([line0.x2() - line0.x1(), line0.y2() - line0.y1()])
        yd = np.array([line1.x1() - line2.x1(), line1.y1() - line2.y1()])

        points_plot = s.reshape(2, 1) + x * xd.reshape(2, 1) + y_norm * yd.reshape(2, 1)

        points = []
        for p in points_plot.T:
            points.append(QtCore.QPointF(p[0], p[1]))

        polygon = QtGui.QPolygonF(points)
        path = QtGui.QPainterPath()
        path.addPolygon(polygon)
        self.items["profile"].setPath(path)
