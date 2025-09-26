import os
import shutil

import numpy as np
from PySide6 import QtCore, QtWidgets

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.app.modules import acquisition_controller, adjustments_controller
from expert_pi.app.modules.navigation import cache, runner
from expert_pi.app.states import preserved_state_saver
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.config import get_config
from expert_pi.gui.main_window import MainWindow
from expert_pi.stream_clients import CacheClient


class NavigationController(QtCore.QObject):
    # update_info_signal = QtCore.Signal(float, float, float)
    # position_changed_signal = QtCore.Signal(float, float)

    stage_position_signal = QtCore.Signal(float, float)

    def __init__(self, window: MainWindow, states: StatesHolder, cache_client: CacheClient) -> None:
        super().__init__()
        self.window = window
        self.states = states
        self._navigation = window.navigation
        self._navigation_view = window.navigation_view

        self.filename = os.path.abspath(get_config().data.navigation_cache)

        self.cache = cache.NavigationCache(get_config())
        self.initialized = False

        self.runner = runner.NavigationRunner(
            self.cache,
            self.update_tiles,
            self.update_info,
            get_config().navigation,
            cache_client,
            self.update_stage_info,
        )

        self.stage_position_signal.connect(self._navigation.tools["stage_tracker"].setPos)

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self.window = window
        self._navigation = window.navigation

        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._navigation_view.show_signal: self.show,
            self._navigation_view.hide_signal: self.hide,
            self._navigation.navigation_control.reload_button.clicked: self.reload_clicked,
            self._navigation.navigation_control.start_button.clicked: self.start_button_clicked,
            self._navigation.navigation_control.stop_button.clicked: self.stop_button_clicked,
            self._navigation.view_rectangle_changed: self.view_rectangle_changed,
            self.window.tool_bar.menu.clear_navigation_action.triggered: lambda: self.cache.initialize(
                self.filename, mode="w"
            ),
            self.window.tool_bar.file_saving.save_signal: self.save_file,
            self.window.tool_bar.file_saving.new_signal: self.new_file,
            self.window.tool_bar.file_saving.opened_change_signal: self.close_open_file,
        }

        return signals

    def view_rectangle_changed(self, rect, force=False):
        if not self.runner.running:
            return
        rect = np.array(rect)
        xy0 = rect[:2]
        xy2 = rect[:2] + rect[2:]

        xy0[1] *= -1  # flip to y
        xy2[1] *= -1

        xy0, xy2, z, ij0, ij2 = self.cache.get_tiles_from_rect(xy0, xy2, self._navigation.sceneRect().width())
        self.runner.view_rectangle_changed(xy0, xy2, z, ij0, ij2, force=force)

    def show(self):
        self.states.interaction_mode = StatesHolder.InteractionMode.navigation
        if not self.initialized:
            self.cache.initialize(self.filename)
            self.initialized = True

        last_electronic, last_stage, _rotation, transform2x2 = scan_helper.get_scanning_shift_and_transform()
        actual_xy = last_stage + transform2x2 @ last_electronic  # sample coordinates
        self._navigation.set_center_position([-actual_xy[0], actual_xy[1]])
        self._navigation.set_fov(self.window.image_view.get_fov())

        self.runner.start()
        rect0 = self._navigation.get_area_rectangle()
        self.view_rectangle_changed(rect0, force=True)

        xy_s = grpc_client.stage.read_x_y_z_a_b()
        xy_stage = np.array([xy_s["x"], xy_s["y"]]) * 1e6  # to um
        self.update_stage_info(xy_stage)

        self.window.tool_bar.file_saving.show(self.filename)

        self.window.tool_bar.file_saving.status_signal.emit(self.filename, True, False)

    def hide(self):
        # move stage to central position
        if self.states.interaction_mode == StatesHolder.InteractionMode.navigation:
            rect = self._navigation.get_area_rectangle()
            rect = np.array(rect)
            pos = rect[:2] + rect[2:] / 2

            self.runner.final_stage_position = np.array([pos[0], -pos[1]])
            self.runner.stop()

            n = int(self.window.scanning.size_combo.currentText().split(" ")[0])

            img, fov = self.get_central_image(n)

            self.window.image_view.main_image_item.set_fov(fov)
            self.window.image_view.set_fov(fov)
            self.window.image_view.normalizer.set_image(img)

            adjustments_controller.synchronize(self.window, self.states)
            new_fov = min(fov, self.window.scanning.fov_spin.maximum())
            self.window.scanning.fov_spin.set_read_sent_value(new_fov)

            acquisition_controller.fov_changed(self.window, self.states, new_fov)

    def reload_clicked(self):
        rect0 = self._navigation.get_area_rectangle()
        rect = np.array(rect0) * 1
        xy0 = rect[:2]
        xy2 = rect[:2] + rect[2:]

        xy0[1] *= -1  # flip to y
        xy2[1] *= -1
        xy0, xy2, z, ij0, ij2 = self.cache.get_tiles_from_rect(xy0, xy2, self._navigation.sceneRect().width())
        self.cache.reset(z, ij0, ij2)
        self.view_rectangle_changed(rect0, force=True)

    def start_button_clicked(self):
        if self._navigation.navigation_control.start_button.selected:
            self.runner.enable_acquisition = True
            rect = self._navigation.get_area_rectangle()
            self.view_rectangle_changed(rect, force=True)

    def stop_button_clicked(self):
        if self._navigation.navigation_control.stop_button.selected:
            self.runner.enable_acquisition = False

    def update_tiles(self, tids, images, selection_range=None):
        self._navigation.update_tiles_signal.emit(tids, images, selection_range)
        # print("update tiles",tids,time.perf_counter()-start)

    def update_info(self, remaining, x, y):
        # print("update info",remaining,x,y)
        pass

    def update_stage_info(self, stage_position):
        self.stage_position_signal.emit(stage_position[0], -stage_position[1])  # flip to scanning coordinates

    def get_central_image(self, px_size, channel="BF"):
        rect0 = self._navigation.get_area_rectangle()
        rect = np.array(rect0) * 1
        xy0 = rect[:2]
        xy2 = rect[:2] + rect[2:]

        xy0[1] *= -1  # flip to y
        xy2[1] *= -1

        center = [(xy0[0] + xy2[0]) / 2, (xy0[1] + xy2[1]) / 2]
        size = min(np.abs(xy2[0] - xy0[0]), np.abs(xy2[1] - xy0[1]))

        xy0_new = [center[0] - size / 2, center[1] + size / 2]
        xy2_new = [center[0] + size / 2, center[1] - size / 2]

        img = self.cache.get_image(xy0_new, xy2_new, px_size, px_size, channel)
        return img, size

    def save_file(self, filename, close):
        if self.states.interaction_mode == StatesHolder.InteractionMode.navigation:
            if filename != self.cache.f.filename:
                self.cache.f.close()
                shutil.copyfile(self.cache.f.filename, filename)
                self.cache.initialize(filename, "r+")
            if close:
                self.cache.f.close()

    def new_file(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.navigation:
            last_folder = preserved_state_saver.actual_state["last_save_directory"]
            name, *_ = QtWidgets.QFileDialog.getSaveFileName(
                self.window, "Save File", last_folder + "untitled.h5", "stem 4D h5 file (*.h5)"
            )
            if not name:
                raise Exception("no name selected")

            preserved_state_saver.actual_state["last_save_directory"] = os.path.dirname(name) + "/"
            preserved_state_saver.save(get_config().data.data_folder)

            self.filename = name
            self.cache.initialize(self.filename, "w")
            self.window.tool_bar.file_saving.status_signal.emit(self.filename, True, True)

    def close_open_file(self, value):
        if self.states.interaction_mode == StatesHolder.InteractionMode.navigation:
            if not value:
                self.save_file(self.filename, True)
            elif not bool(self.cache.f):
                self.cache.initialize(self.filename)
