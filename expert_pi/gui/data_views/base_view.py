# ruff: noqa: N802
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.data_views.image_item import ImageItem
from expert_pi.gui.tools.base import Tool


class GraphicsArea(QtWidgets.QGraphicsItemGroup):
    def __init__(self):
        """GraphicsArea.

        graphics area works in real units (either um for stem image or mrads for camera view)
        it acts as a wrapper for zooming/panning of the children in the scene
        """
        super().__init__()
        self.items = set([])

    def addItem(self, item):
        self.items.add(item)
        item.setParentItem(self)
        return item

    def removeItem(self, item):
        item.setParentItem(None)
        self.items.remove(item)

    def clearItems(self):
        for item in self.items:
            item.setParentItem(None)
        self.items = set([])


class GraphicsView(QtWidgets.QGraphicsView):
    update_signal = QtCore.Signal()  # used by threads
    request_update_signal = QtCore.Signal()  # call to request emitters push image to pipelines

    wheelEventSignal = QtCore.Signal(object)
    mousePressEventSignal = QtCore.Signal(object)
    mouseMoveEventSignal = QtCore.Signal(object)
    mouseReleaseEventSignal = QtCore.Signal(object)
    leaveEventSignal = QtCore.Signal(object)
    mouseDoubleClickEventSignal = QtCore.Signal(object)

    save_image_signal = QtCore.Signal()
    save_scene_signal = QtCore.Signal()

    # double_clicked_signal = QtCore.Signal(object)
    # clicked_signal = QtCore.Signal(object)

    # ctrl_wheel_signal = QtCore.Signal(object)  # focus
    # mouse_dragging_signal = QtCore.Signal(object)  # illumination/stigmation (ctrl)

    def __init__(self):
        super().__init__()
        self.max_N = 2048
        scene = QtWidgets.QGraphicsScene(0, 0, self.max_N, self.max_N)
        self.setStyleSheet("background-color:black")
        self.setScene(scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self.graphics_area = GraphicsArea()  # this is used for inner interaction
        self.scene().addItem(self.graphics_area)

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)

        # use from thread
        self.update_signal.connect(self.redraw, type=QtCore.Qt.ConnectionType.BlockingQueuedConnection)

        self.image_items: dict[str, ImageItem] = {}
        self.tools: dict[str, Tool] = {}

        self.context_menu = QtWidgets.QMenu(self)
        self.context_menu.addAction("save raw image", self.save_image_signal.emit)
        self.context_menu.addAction("save scene", self.save_scene_signal.emit)
        self.customContextMenuRequested.connect(self.on_context_menu)

        self.hovered_item = None
        self.dragged_item = None
        self.dragging = False

    def add_image_item(self, name, image_item):
        if name in self.image_items:
            raise Exception(name + "already in image items")
        self.image_items[name] = image_item
        self.graphics_area.addItem(image_item)
        image_item.show()
        image_item.update_transforms()

    def update_hover_item(self, e, leave=False):
        # TODO for some reason hoverEnter hoverLeave on graphics item does not work properly, use this instead
        if leave:
            if self.hovered_item is not None:
                if hasattr(self.hovered_item, "hover_leave"):
                    self.hovered_item.hover_leave()
            self.hovered_item = None
        else:
            scene_pos = self.mapToScene(e.pos())

            new_hovered_item = self.scene().itemAt(scene_pos, QtGui.QTransform())
            if self.hovered_item is not None:
                if new_hovered_item is not self.hovered_item:
                    if hasattr(self.hovered_item, "hover_leave"):
                        self.hovered_item.hover_leave()
            self.hovered_item = new_hovered_item
            if hasattr(self.hovered_item, "hover"):
                self.hovered_item.hover()

        return self.hovered_item is not None

    def update_dragged_item(self, e, leave=False, enter=False):
        if leave:
            self.dragged_item = None
            return False
        if enter:
            scene_pos = self.mapToScene(e.pos())
            self.dragged_item = self.scene().itemAt(scene_pos, QtGui.QTransform())
            if hasattr(self.dragged_item, "set_start_drag_position"):
                a_pos = self.map_to_area([e.pos().x(), e.pos().y()])
                self.dragged_item.set_start_drag_position(a_pos[0], a_pos[1])
                self.dragged_item.drag(a_pos[0], a_pos[1])
                return True
        if self.dragged_item is not None:
            if hasattr(self.dragged_item, "drag"):
                a_pos = self.map_to_area([e.pos().x(), e.pos().y()])
                self.dragged_item.drag(a_pos[0], a_pos[1])
                return True

        return False

    def update_tools(self):
        for tool in self.tools.values():
            if tool.is_active:
                tool.update()

    def redraw(self):
        for image_item in self.image_items.values():
            image_item.update_image()

    def get_fov(self):
        sr = self.sceneRect()
        size = min(sr.width(), sr.height())
        scale = self.graphics_area.scale()
        if scale != 0:
            return size / scale
        else:
            return np.inf

    def set_fov(self, fov, center_position=None):
        if center_position is None:
            center_position = self.get_center_position()
        self.setSceneRect(self.visibleRegion().boundingRect())
        sr = self.sceneRect()
        size = min(sr.width(), sr.height())

        self.graphics_area.setScale(size / fov)
        self.set_center_position(center_position)

    def get_area_rectangle(self):
        scale = self.graphics_area.scale()
        rectangle = [
            -self.graphics_area.pos().x() / scale,
            -self.graphics_area.pos().y() / scale,
            self.sceneRect().width() / scale,
            self.sceneRect().height() / scale,
        ]
        return rectangle

    def map_from_area(self, position):
        """From real units [i,j] to pixels on screen."""
        scale = self.graphics_area.scale()
        return [self.graphics_area.pos().x() + position[0] * scale, self.graphics_area.pos().y() + position[1] * scale]

    def map_to_area(self, position):
        """From on screen to real units [i,j]."""
        scale = self.graphics_area.scale()
        return [
            (position[0] - self.graphics_area.pos().x()) / scale,
            (position[1] - self.graphics_area.pos().y()) / scale,
        ]

    def get_center_position(self):
        sr = self.sceneRect()
        scale = self.graphics_area.scale()

        return [
            (self.graphics_area.pos().x() - sr.width() / 2) / scale,
            (self.graphics_area.pos().y() - sr.height() / 2) / scale,
        ]

    def set_center_position(self, position):
        """Position in real units [x,y]."""
        scale = self.graphics_area.scale()
        sr = self.sceneRect()
        self.graphics_area.setPos(position[0] * scale + sr.width() / 2, position[1] * scale + sr.height() / 2)

    def resizeEvent(self, event):
        self.set_fov(self.get_fov())
        for name, tool in self.tools.items():
            if tool.is_active:
                tool.update()

    def wheelEvent(self, event):
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().x() / 2880 + event.angleDelta().y() / 2880  # in degrees
            factor = 2 ** (delta * 12)
            center_on = event.position()
            last_origin = self.graphics_area.pos()
            new_origin = center_on + factor * (last_origin - center_on)
            self.graphics_area.setPos(new_origin)
            self.graphics_area.setScale(self.graphics_area.scale() * factor)
            self.update_tools()

        self.wheelEventSignal.emit(event)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        focused_item = self.scene().itemAt(scene_pos, QtGui.QTransform())

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.update_dragged_item(event, enter=True)
            for name, tool in self.tools.items():
                if tool.is_active:
                    tool.view_mouse_pressed(event, focused_item)

        self.mousePressEventSignal.emit(event)
        #print(event.pos())
        
    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        focused_item = self.scene().itemAt(scene_pos, QtGui.QTransform())

        for tool in self.tools.values():
            if tool.is_active:
                tool.view_mouse_moved(event, focused_item)
        self.update_dragged_item(event)
        self.update_hover_item(event)
        self.mouseMoveEventSignal.emit(event)

    def mouseReleaseEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        focused_item = self.scene().itemAt(scene_pos, QtGui.QTransform())

        for tool in self.tools.values():
            if tool.is_active:
                tool.view_mouse_released(event, focused_item)

        self.update_hover_item(event, leave=True)
        self.update_dragged_item(event, leave=True)

        self.mouseReleaseEventSignal.emit(event)

    def leaveEvent(self, event):
        for tool in self.tools.values():
            if tool.is_active:
                tool.view_mouse_leaved(event)

        self.update_hover_item(event, leave=True)
        self.update_dragged_item(event, leave=True)

        self.leaveEventSignal.emit(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        self.mouseDoubleClickEventSignal.emit(event)

    def on_context_menu(self, point):
        self.context_menu.exec_(self.mapToGlobal(point))
