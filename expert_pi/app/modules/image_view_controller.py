import threading
import time

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

from expert_pi.app import data_saver, scan_helper
from expert_pi.app.modules import acquisition_controller, adjustments_controller
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.config import get_config
from expert_pi.gui import main_window
from expert_pi.gui.elements import file_dialog
from expert_pi.gui.main_window import MainWindow
from expert_pi.stream_processors import normalizer


class ImageViewController:
    def __init__(self, window: main_window.MainWindow, states: StatesHolder):
        self.window = window
        self.states = states
        self.normalizer = normalizer.Normalizer(self.update_view, self.update_histogram)
        self.view = window.image_view
        self.view.normalizer = self.normalizer  # insert
        self.histogram = window.image_histogram

        self._signals = self._create_signals()
        self.connect_signals(window)

        self.last_single_point_emit = time.perf_counter()
        self.min_time_remit = 1 / 10

    def connect_signals(self, window: MainWindow):
        self._signals = self._create_signals()
        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self.window.stem_tools.visualisation.clicked: self.enable_histogram,
            self.window.image_histogram.histogram_changed: self.histogram_ranges_changed,
            self.window.stem_tools.image_type.clicked: self.set_channel,
            self.view.point_selection_signal: self.point_selection,
            # mouse interaction
            self.view.mouseDoubleClickEventSignal: self._mouse_double_click,
            self.view.wheelEventSignal: self._wheel_event,
            self.view.mousePressEventSignal: self._mouse_press,
            self.view.mouseMoveEventSignal: self._mouse_move,
            self.view.mouseReleaseEventSignal: self._mouse_release,
            self.view.leaveEventSignal: self._leave,
            #
            self.view.save_image_signal: self.save_image,
            self.view.save_scene_signal: self.save_scene,
        }
        return signals

    def set_channel(self, channel):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
            alpha, beta, amplify = self.histogram.set_channel(channel + "_4DSTEM")
        elif self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            alpha, beta, amplify = self.histogram.set_channel(channel)
        else:
            return
        self.normalizer.set_alpha_beta(alpha, beta, amplify)

    def update_view(self, image_8b, frame_index, scan_id):
        self.view.main_image_item.set_image(image_8b)
        self.view.tools["status_info"].update_frame_index(frame_index, scan_id)

        if threading.current_thread() is threading.main_thread():
            self.view.redraw()
        else:
            self.view.update_signal.emit()

        # set info

    def update_histogram(self, histogram_data, frame_index, scan_id):
        self.histogram.recalculate_polygons(histogram_data)

        if self.histogram.auto_range_enabled[self.histogram.channel]:
            mask = histogram_data > 0
            alpha, beta = self.histogram.set_range([
                np.argmax(mask) / len(mask),
                (len(mask) - np.argmax(mask[::-1])) / len(mask),
            ])
            self.normalizer.set_alpha_beta(alpha, beta, self.histogram.amplify[self.histogram.channel])

        if threading.current_thread() is threading.main_thread():
            self.histogram.redraw()
        else:
            self.histogram.update_signal.emit()

    def histogram_ranges_changed(self, channel, alpha, beta):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d
            and channel == self.states.image_view_channel + "_4DSTEM"
        ) or (
            self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and channel == self.states.image_view_channel
        ):
            self.normalizer.set_alpha_beta(alpha, beta, self.histogram.amplify[self.histogram.channel])

    def enable_histogram(self, name):
        if "histogram" in self.window.stem_tools.visualisation.selected:
            self.normalizer.histogram_enable = True
        else:
            self.normalizer.histogram_enable = False

    def _wheel_event(self, event: QtGui.QWheelEvent):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.setModifiers(QtCore.Qt.KeyboardModifier.NoModifier)  # remove it otherwise step will be 10x
            self.window.stem_adjustments.focus.wheelEvent(event, force=True)

    def _mouse_double_click(self, event: QtGui.QMouseEvent):
        if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            xy = self.view.map_to_area([event.pos().x(), event.pos().y()])
            self.relative_move(xy)

    def _mouse_press(self, event: QtGui.QMouseEvent):
        pass

    def _mouse_move(self, event: QtGui.QMouseEvent):
        pass

    def _mouse_release(self, event: QtGui.QMouseEvent):
        pass

    def _leave(self, event: QtCore.QEvent):
        pass

    def point_selection(self, dxy: list[float]):
        if not self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            return

        if "stop" in self.window.camera.control_buttons.selected:
            return

        if "start" in self.window.camera.control_buttons.selected:
            frames = 0
        else:
            frames = 1

        fov = self.view.main_image_item.fov
        n = self.view.main_image_item.image.shape[0]
        ij = [int(dxy[1] / fov * n + 0.5) + n // 2, int(dxy[0] / fov * n + 0.5) + n // 2]

        now = time.perf_counter()
        if now - self.last_single_point_emit > self.min_time_remit:
            acquisition_controller.start_4dstem(self.window, self.states, frames=frames, rectangle=[ij[0], ij[1], 1, 1])
            self.last_single_point_emit = now

    def relative_move(self, dxy: list[float]):
        if self.states.image_view_channel == "4DSTEM":
            return
        fov0 = self.view.main_image_item.fov * 1.0
        fov = self.view.get_fov()

        n = self.view.main_image_item.image.shape[0]
        m02 = 0.5 * (1 - fov0 / fov) * n - dxy[0] / fov * n
        m12 = 0.5 * (1 - fov0 / fov) * n - dxy[1] / fov * n
        M = np.float32([[fov0 / fov, 0, m02], [0, fov0 / fov, m12]])

        modified = cv2.warpAffine(self.view.main_image_item.image, M, self.view.main_image_item.image.shape[:2])

        dxy[1] *= -1  # due to scan vs stage coordinate system
        xy_electronic, xy_stage, _rotation, transform2x2 = self.view.update_transforms()

        dxy_rotated = np.dot(
            np.array([
                [np.cos(self.view.rotation), np.sin(self.view.rotation)],
                [-np.sin(self.view.rotation), np.cos(self.view.rotation)],
            ]),
            dxy,
        )

        # if self.view.xyz_tracking.isVisible():
        #     dxy_rotated -= np.array(
        #             [self._view.xyz_tracking.aligned_image.shift[0], -self._view.xyz_tracking.aligned_image.shift[1]]
        #     )

        _new_stage, new_electronic, future, _use_stage = scan_helper.set_combined_shift(
            dxy_rotated,
            xy_electronic,
            xy_stage,
            transform2x2,
            mode=self.window.stem_adjustments.stem_shift_type.selected,
            electronic_limit=1,
            stage_nowait=not self.window.stem_adjustments.stage_backlash.isChecked(),
        )
        self.window.stem_adjustments.shift_x.set_read_sent_value(new_electronic[0] * 1e3)
        self.window.stem_adjustments.shift_y.set_read_sent_value(new_electronic[1] * 1e3)

        if future is not None:

            def callback(x):
                self.window.stem_adjustments.stage_x.setProperty(
                    "error", bool(x.exception())
                )  # most likely stage colission
                self.window.stem_adjustments.stage_y.setProperty("error", bool(x.exception()))
                self.window.stem_adjustments.stage_x.update_style()
                self.window.stem_adjustments.stage_y.update_style()

            future.add_done_callback(callback)

        fov = min(fov, self.states.acquisition.max_fov)

        self.window.scanning.fov_spin.set_read_sent_value(fov)

        self.normalizer.image_8b = modified

        self.view.main_image_item.set_image(modified)
        self.view.main_image_item.update_image()
        self.view.main_image_item.set_fov(fov)

        self.view.set_center_position([0, 0])

        acquisition_controller.fov_changed(self.window, self.states, fov)

        # self.view.xy_electronic = new_electronic
        # self.view.xy_stage = new_stage

        # if len(self._view.background_images) > 0:
        #     self._view.repaint_background(new_electronic, new_stage, rotation, transform2x2)
        # if self._view.stage_ranges.isVisible():
        #     self._view.stage_ranges.update(new_stage, self._view.rotation, self._view.transform2x2)

        adjustments_controller.synchronize(self.window, self.states)

    def save_image(self):
        config = get_config()
        filename = file_dialog.image_file_save_dialog(self.window, config.data.data_folder)
        if filename is None:
            return

        if ".tif" in filename:
            image = self.normalizer.raw_image
        else:
            image = self.normalizer.image_8b

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if ".tif" in filename:
            data_saver.save_tiff_image(filename, image)
        elif ".png" in filename:
            data_saver.save_png_image(filename, image)

        if config.data.save_metadata:
            metadata = data_saver.get_microscope_metadata(config.connection)

            channels = [self.states.image_view_channel]
            if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
                pixel_time = self.window.scanning.pixel_time_spin.value()
                detector = "STEM"
            elif self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
                pixel_time = self.window.stem_4d.pixel_time_spin.value() * 1e-3
                detector = "virtual STEM"

            metadata["detectors"][detector] = data_saver.get_stem_metadata(
                self.view.main_image_item.fov, pixel_time, channels
            )
            data_saver.save_metadata(filename, metadata)

    def save_scene(self):
        name = file_dialog.image_file_save_dialog(self.window, get_config().data.data_folder)
        if name is not None:
            pix_map = self.view.grab(self.view.sceneRect().toRect())
            pix_map.save(name)
