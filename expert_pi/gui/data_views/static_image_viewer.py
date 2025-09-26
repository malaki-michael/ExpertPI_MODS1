import threading

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class StaticImageViewer(QtWidgets.QGraphicsView):
    name: str

    _update_signal = QtCore.Signal(object, object)

    save_image_signal = QtCore.Signal()
    point_selection_signal = QtCore.Signal(object)

    def __init__(self):
        super().__init__()

        self.setScene(QtWidgets.QGraphicsScene())
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        self.image = (np.random.rand(20, 20) * 255).astype(np.uint8)
        image2 = QtGui.QImage(self.image, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap(image2)
        self.pix_map = self.scene().addPixmap(pix)

        self._update_signal.connect(self._update_image, type=QtCore.Qt.ConnectionType.BlockingQueuedConnection)

        self.aspect = (1, 1)  # if None it will be automatically expanding to parent
        self.zoom = 1.0

        self.last_position = QtCore.QPointF(0, 0)
        self.last_dimensions = self.image.shape[:2]
        self.panning = False

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

        # create context menu
        self.pop_menu = QtWidgets.QMenu(self)
        self.pop_menu.addAction("Save", self.save_image_signal.emit)

        self.tools = {}
        self.graphics_area = self.scene()
        self.graphics_area.scale = lambda: self.zoom

        self.hovered_item = None
        self.dragged_item = None
        self.dragging = False

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

    def extract_format(self, image: np.ndarray):
        if image.ndim == 2:
            if image.dtype == np.uint8:
                return QtGui.QImage.Format.Format_Grayscale8
            else:
                return QtGui.QImage.Format.Format_Grayscale16
        elif image.shape[2] == 4:
            return QtGui.QImage.Format.Format_RGBA8888
        else:
            return QtGui.QImage.Format.Format_RGB888

    def set_image(self, image_8b: np.ndarray, rectangle=None):
        format_ = self.extract_format(image_8b)
        self.image = image_8b
        self.format = format_
        pix_map = QtGui.QPixmap(
            QtGui.QImage(image_8b, image_8b.shape[1], image_8b.shape[0], image_8b.strides[0], format_)
        )
        dimensions = [self.image.shape[1], self.image.shape[0]]

        if threading.current_thread() is threading.main_thread():
            self._update_image(pix_map, dimensions)
        else:
            self._update_signal.emit(pix_map, dimensions)

    def _update_image(self, pix_map, dimensions):
        self.pix_map.setPixmap(pix_map)

        if self.last_dimensions[0] != dimensions[0] or self.last_dimensions[1] != dimensions[1]:
            self.last_dimensions = dimensions
            self.apply_zoom()
            self.scene().setSceneRect(0, 0, self.last_dimensions[0], self.last_dimensions[1])

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x() / 2880 + ev.angleDelta().y() / 2880  # in degrees
        self.zoom = max(1, self.zoom * 2 ** (delta * 5))
        self.apply_zoom(center_on=ev.position())

    def apply_zoom(self, zoom=None, center_on=None):
        self.setSceneRect(
            0, 0, self.last_dimensions[0], self.last_dimensions[1]
        )  # to update at centering even when there is a same scaling
        if zoom is not None:
            self.zoom = zoom
        v_box = self.visibleRegion().boundingRect()
        i_box = [self.image.shape[1], self.image.shape[0]]

        if v_box.width() == 0:
            fact_w = i_box[0] / 1.0
        else:
            fact_w = i_box[0] / 1.0 / v_box.width()
        if v_box.height() == 0:
            fact_h = i_box[1] / 1.0
        else:
            fact_h = i_box[1] / 1.0 / v_box.height()

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()

        if center_on is not None:
            if h_bar.pageStep() == 0:
                h_bar.setPageStep(v_box.width())
            if v_bar.pageStep() == 0:
                v_bar.setPageStep(v_box.height())
            factor_on_image = [
                (center_on.x() + h_bar.value()) / (h_bar.maximum() + h_bar.pageStep()),
                (center_on.y() + v_bar.value()) / (v_bar.maximum() + v_bar.pageStep()),
            ]

        if self.aspect is not None:
            fact = np.array([max(fact_w, fact_h) * self.aspect[0], max(fact_w, fact_h) * self.aspect[1]])
        else:
            fact = np.array([fact_w, fact_h])

        self.setTransform(QtGui.QTransform.fromScale(self.zoom / fact[0], self.zoom / fact[1]))

        if center_on is not None:
            h_bar.setValue((h_bar.maximum() + h_bar.pageStep()) * factor_on_image[0] - center_on.x())
            v_bar.setValue((v_bar.maximum() + v_bar.pageStep()) * factor_on_image[1] - center_on.y())

        self.update_tools()

    def resizeEvent(self, event):
        self.apply_zoom()

    def show(self):
        super().show()
        self.apply_zoom()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.panning = True
            self.last_position = event.position()
            for name, tool in self.tools.items():
                if tool.is_active:
                    tool.view_mouse_pressed(event, None)

        # self.mousePressEventSignal.emit(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.panning:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            delta = event.position() - self.last_position

            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self.last_position = event.position()
            return

        self.update_hover_item(event)
        if event.buttons() & QtCore.Qt.LeftButton:
            self.update_dragged_item(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.panning:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            delta = event.position() - self.last_position

            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self.last_position = event.position()
        self.panning = False

    def leaveEvent(self, e):
        self.update_hover_item(None, leave=True)
        self.update_dragged_item(None, leave=True)

    def on_context_menu(self, point):
        # show context menu
        self.pop_menu.exec_(self.mapToGlobal(point))

    def map_to_area(self, position):
        res = self.mapToScene(position[0], position[1])
        return np.array([res.x(), res.y()])
