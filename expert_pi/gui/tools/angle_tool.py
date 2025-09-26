import numpy as np
from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base
from expert_pi.gui.tools.graphic_items import DragLine, DragPoint, ItemGroup


class AngleTool(base.Tool, ItemGroup):
    def __init__(self, view):
        super().__init__(view)
        self.items = {
            "arc": QtWidgets.QGraphicsEllipseItem(),
            "start_line": DragLine(0, 0, 0, 0, lambda x: x),
            "end_line": DragLine(0, 0, 0, 0, lambda x: x),
            "center": DragPoint(self.center_point_moved),
            "start": DragPoint(self.start_point_moved),
            "end": DragPoint(self.end_point_moved),
            "text": QtWidgets.QGraphicsTextItem(""),
        }

        for name, item in self.items.items():
            if name == "arc":
                item.setPen(QtGui.QPen(QtGui.QColor(100, 100, 255, 100), 0))
                item.setBrush(QtGui.QColor(0, 0, 255, 150))
            elif name in {"center", "start", "end", "text", "start_line", "end_line"}:
                pass
            else:
                item.setPen(QtGui.QPen(QtGui.QColor(200, 200, 255, 255), 0))
            item.setParentItem(self)
            item.hide()

        self.hide()
        self.view.graphics_area.addItem(self)
        self.wizard_step = None

    def show(self):
        super().show()
        self.is_active = True
        self.update()

    def hide(self):
        self.is_active = False
        self.reset_wizard()
        super().hide()

    def reset_wizard(self):
        for item in self.items.values():
            item.hide()
        self.wizard_step = None

    def view_mouse_pressed(self, event, focused_item=None):
        print(self.wizard_step, self.has_item(focused_item))
        if self.has_item(focused_item) or self.wizard_step is None:
            if self.wizard_step is None:
                self.wizard_step = 1
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.use_wizard(area_pos[0], area_pos[1], click=True)

    def view_mouse_moved(self, event, focused_item=None):
        if self.wizard_step == 1:
            area_pos = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.use_wizard(
                area_pos[0],
                area_pos[1],
            )

    def limit_end_position(self, x, y):
        cp = self.items["center"].pos()
        sp = self.items["start"].pos()
        r = np.sqrt((sp.x() - cp.x()) ** 2 + (sp.y() - cp.y()) ** 2)

        v = np.array([x - cp.x(), y - cp.y()])
        v_abs = np.sqrt(v[0] ** 2 + v[1] ** 2)
        if v_abs > 0:
            v = v / v_abs * r
        return cp.x() + v[0], cp.y() + v[1]

    def use_wizard(self, x, y, click=False):
        if click:
            if self.wizard_step == 0:
                self.items["center"].setPos(x, y)
                self.items["start"].setPos(x, y)
                self.items["end"].setPos(x, y)

                self.items["start"].show()
                self.items["start_line"].show()
                self.update_arc()
                self.wizard_step = 1
                return

            elif self.wizard_step == 1:
                self.items["start"].setPos(x, y)
                self.items["end"].setPos(x, y)

                for item in self.items.values():
                    item.show()
                self.update_arc()

                self.wizard_step = 2
                return

            elif self.wizard_step == 2:
                x2, y2 = self.limit_end_position(x, y)
                self.items["end"].setPos(x2, y2)
                self.update_arc()

                self.wizard_step = None

        elif self.wizard_step == 0:
            self.items["center"].setPos(x, y)
            self.items["center"].show()

        elif self.wizard_step == 1:
            self.items["start"].setPos(x, y)
            self.items["end"].setPos(x, y)
            self.update_arc()

        elif self.wizard_step == 2:
            x2, y2 = self.limit_end_position(x, y)
            self.items["end"].setPos(x2, y2)
            self.update_arc()

    def update_arc(self):
        self.items["start_line"].setLine(
            self.items["center"].pos().x(),
            self.items["center"].pos().y(),
            self.items["start"].pos().x(),
            self.items["start"].pos().y(),
        )
        self.items["end_line"].setLine(
            self.items["center"].pos().x(),
            self.items["center"].pos().y(),
            self.items["end"].pos().x(),
            self.items["end"].pos().y(),
        )

        cp = self.items["center"].pos()
        sp = self.items["start"].pos()
        ep = self.items["end"].pos()
        sv = np.array([sp.x() - cp.x(), sp.y() - cp.y()])
        ev = np.array([ep.x() - cp.x(), ep.y() - cp.y()])

        if sv[0] == 0 and sv[1] == 0:
            sp_angle = 0
        else:
            sp_angle = np.arctan2(sv[1], sv[0])
        if ev[0] == 0 and ev[1] == 0:
            ep_angle = 0
        else:
            ep_angle = np.arctan2(ev[1], ev[0])
        span = ep_angle - sp_angle

        if span > np.pi:
            span = -2 * np.pi + span
        elif span < -np.pi:
            span = 2 * np.pi + span

        r = np.sqrt((sp.x() - cp.x()) ** 2 + (sp.y() - cp.y()) ** 2)
        self.items["arc"].setRect(cp.x() - r, cp.y() - r, 2 * r, 2 * r)
        self.items["arc"].setStartAngle(-sp_angle / np.pi * 180 * 16)
        self.items["arc"].setSpanAngle(-span / np.pi * 180 * 16)  # 1/16 of degrees

        text = self.items["text"]
        r = text.boundingRect()

        pos = [(cp.x() + sp.x() + ep.x()) / 3, (cp.y() + sp.y() + ep.y()) / 3]
        text.setPos(pos[0] - r.center().x() * text.scale(), pos[1] - r.center().y() * 2 * text.scale())
        text.setHtml(f"<span style='color:#aaaaff;font-size:16px'>{np.abs(span / np.pi * 180):6.2f}°</span>")

    def start_point_moved(self, x, y):
        self.items["start"].setPos(x, y)
        x2, y2 = self.limit_end_position(self.items["end"].pos().x(), self.items["end"].pos().y())
        self.items["end"].setPos(x2, y2)
        self.update_arc()

    def end_point_moved(self, x, y):
        x2, y2 = self.limit_end_position(x, y)
        self.items["end"].setPos(x2, y2)
        self.update_arc()

    def center_point_moved(self, x, y):
        self.items["center"].setPos(x, y)
        x2, y2 = self.limit_end_position(self.items["end"].pos().x(), self.items["end"].pos().y())
        self.items["end"].setPos(x2, y2)
        self.update_arc()

    def update(self, *args, **kwargs):
        # sr = self.view.sceneRect()
        scale = self.view.graphics_area.scale()
        self.items["center"].setScale(1.0 / scale)
        self.items["start"].setScale(1.0 / scale)
        self.items["end"].setScale(1.0 / scale)
        self.items["text"].setScale(1.0 / scale)
