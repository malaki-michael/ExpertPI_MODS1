# ruff: noqa: N802
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.config import NavigationConfig
from expert_pi.gui.data_views import base_view, navigation_map
from expert_pi.gui.elements import buttons
from expert_pi.gui.style import images_dir
from expert_pi.gui.tools import cross, size_bar, stage_tracker


class NavigationControl(QtWidgets.QWidget):
    def __init__(self, central_widget):
        super().__init__()
        self.central_widget = central_widget
        self.setParent(self.central_widget)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.setStyleSheet("background-color:rgba(0,0,0,0)")

        self.reload_button = buttons.IconButton(
            images_dir + "tools_icons/navigation/reload.svg",
            icon_hover=images_dir + "tools_icons/navigation/reload_hover.svg",
        )

        self.start_button = buttons.IconSwitchableButton(
            images_dir + "tools_icons/navigation/start.svg",
            images_dir + "tools_icons/navigation/start_selected.svg",
            icon_selected_hover=images_dir + "tools_icons/navigation/start_selected_hover.svg",
            icon_hover=images_dir + "tools_icons/navigation/start_hover.svg",
        )
        self.stop_button = buttons.IconSwitchableButton(
            images_dir + "tools_icons/navigation/pause.svg",
            images_dir + "tools_icons/navigation/pause_selected.svg",
            icon_selected_hover=images_dir + "tools_icons/navigation/pause_selected_hover.svg",
            icon_hover=images_dir + "tools_icons/navigation/pause_hover.svg",
        )

        self.layout().addWidget(self.reload_button)
        self.layout().addWidget(self.start_button)
        self.layout().addWidget(self.stop_button)

        size = 40

        for button in [self.reload_button, self.start_button, self.stop_button]:
            button.setFixedSize(size, size)
            button.setIconSize(QtCore.QSize(size, size))

        self.start_button.set_selected(False)
        self.stop_button.set_selected(True)

        self.start_button.linked_button = self.stop_button
        self.stop_button.linked_button = self.start_button

        self.setGeometry(20, 20, size, size * 3)


class NavigationView(base_view.GraphicsView):
    view_rectangle_changed = QtCore.Signal(object)

    update_tiles_signal = QtCore.Signal(object, object, object)
    clean_tiles_signal = QtCore.Signal(int, object, object)

    def __init__(self):
        super().__init__()

        self.tools["stage_tracker"] = stage_tracker.StageTracker(self)
        self.tools["stage_tracker"].setZValue(100)

        self.tools["size_bar"] = size_bar.SizeBar(self, color=QtGui.QColor(102, 102, 221))
        self.tools["size_bar"].show()

        self.tools["cross"] = cross.Cross(self)
        self.tools["cross"].setZValue(99)
        self.tools["cross"].show()
        self.tools["cross"].lines[0].setLine(-4000 / 2, 0, 4000 / 2, 0)
        self.tools["cross"].lines[1].setLine(0, -4000 / 2, 0, 4000 / 2)

        self.panning = False
        self.start_position = QtCore.QPoint(0, 0)

        self.set_fov(100, [0, 0])

        self.tools["stage_tracker"].show()
        self.tools["stage_tracker"].update()

        self.navigation_control = NavigationControl(self)
        self.navigation_map = navigation_map.NavigationMap(self)

        self.update_tiles_signal.connect(self.navigation_map.update_tiles)
        self.clean_tiles_signal.connect(self.navigation_map.clean_tiles)

        self.panning = False
        self.start_position = None

    def resizeEvent(self, event):
        self.view_rectangle_changed.emit(self.get_area_rectangle())
        return super().resizeEvent(event)

    def wheelEvent(self, event):
        result = super().wheelEvent(event)
        self.view_rectangle_changed.emit(self.get_area_rectangle())
        return result

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        result = super().mousePressEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.panning = True
            self.start_position = event.pos()
            return result

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        result = super().mouseMoveEvent(event)
        if self.panning:
            last_origin = self.graphics_area.pos()
            self.graphics_area.setPos(last_origin + event.pos() - self.start_position)
            self.start_position = event.pos()
            self.view_rectangle_changed.emit(self.get_area_rectangle())
            self.tools["stage_tracker"].update()

        return result

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        result = super().mouseReleaseEvent(event)
        if self.panning:
            last_origin = self.graphics_area.pos()
            self.graphics_area.setPos(last_origin + event.pos() - self.start_position)
            self.start_position = event.pos()
            self.view_rectangle_changed.emit(self.get_area_rectangle())
            self.tools["stage_tracker"].update()

        self.panning = False
        return result

    def leaveEvent(self, event):
        result = super().leaveEvent(event)
        self.panning = False
        return result

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        result = super().mouseDoubleClickEvent(event)
        last_origin = self.graphics_area.pos()
        center = QtCore.QPointF(self.geometry().width() / 2, self.geometry().height() / 2)
        self.graphics_area.setPos(last_origin - event.pos() + center)

        self.view_rectangle_changed.emit(self.get_area_rectangle())
        self.tools["stage_tracker"].update()

        return result

    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     self.update_children()

    # def wheelEvent(self, event):
    #     delta = event.angleDelta().x() / 2880 + event.angleDelta().y() / 2880  # in degrees
    #     factor = 2 ** (delta * 12)
    #     center_on = event.position()
    #     last_origin = self.graphics_area.pos()

    #     sr = self.sceneRect()
    #     size = min(sr.width(), sr.height())
    #     scale = self.graphics_area.scale()
    #     if scale != 0:
    #         fov = size / scale
    #         min_fov = settings.max_size / 2**settings.max_zoom
    #         if fov / factor > settings.max_size:
    #             factor = fov / settings.max_size
    #         elif fov / factor < min_fov:
    #             factor = fov / min_fov

    #     new_origin = center_on + factor * (last_origin - center_on)
    #     self.graphics_area.setPos(new_origin)
    #     self.graphics_area.setScale(self.graphics_area.scale() * factor)

    #     self.update_children()

    # def mousePressEvent(self, event):
    #     if event.button() == QtCore.Qt.MouseButton.LeftButton:
    #         self.panning = True
    #         self.start_position = event.pos()

    # def mouseMoveEvent(self, event):
    #     if self.panning:
    #         last_origin = self.graphics_area.pos()
    #         self.graphics_area.setPos(last_origin + event.pos() - self.start_position)
    #         self.start_position = event.pos()

    # def mouseReleaseEvent(self, event):
    #     if self.panning:
    #         last_origin = self.graphics_area.pos()
    #         self.graphics_area.setPos(last_origin + self.start_position - event.pos())
    #         self.start_position = event.pos()
    #         self.panning = False
    #         self.update_children()

    # def mouseDoubleClickEvent(self, event):
    #     last_origin = self.graphics_area.pos()
    #     center = QtCore.QPointF(self.geometry().width() / 2, self.geometry().height() / 2)
    #     print(event.pos(), last_origin, center)
    #     self.graphics_area.setPos(last_origin - event.pos() + center)

    #     self.update_children()

    # def update_children(self):
    #     if self.stage_marker.isVisible():
    #         self.stage_marker.update()
    #     self.size_bar.update()

    #     self.children_updated_signal.emit()

    # def position_changed(self, x, y):
    #     self.stage_marker.update()
    #     self.stage_marker.setPos(x, -y)  # y is in pixel units
    #     self.update()
