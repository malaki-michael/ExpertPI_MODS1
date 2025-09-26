import threading

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

from expert_pi import grpc_client
from expert_pi.app import data_saver
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.config import get_config
from expert_pi.gui import main_window
from expert_pi.gui.elements import file_dialog
from expert_pi.gui.main_window import MainWindow
from expert_pi.stream_processors import normalizer


class DiffractionViewController:
    def __init__(self, window: main_window.MainWindow, states: StatesHolder):
        self.window = window
        self.states = states
        self.normalizer = normalizer.Normalizer(self.update_view, self.update_histogram)
        self.view = window.diffraction_view
        self.histogram = window.diffraction_histogram

        self.dragging = False
        self.drag_initial_position = None

        self.lock = threading.Lock()

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self._signals = self._create_signals()
        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self.window.diffraction_tools.visualisation.clicked: self.enable_histogram,
            self.histogram.histogram_changed: self.histogram_ranges_changed,
            self.window.diffraction_tools.image_type.clicked: self.set_channel,
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

        if channel == "camera":
            self.view.main_image_item.set_rotation(
                grpc_client.projection.get_camera_to_stage_rotation() + grpc_client.scanning.get_rotation()
            )
        elif channel == "fft":
            self.view.main_image_item.set_rotation(0)

    def update_view(self, image_8b, frame_index, scan_id):
        self.view.main_image_item.set_image(image_8b)

        self.view.tools["status_info"].update_frame_index(frame_index, scan_id)

        with self.lock:
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
            and channel == self.states.diffraction_view_channel + "_4DSTEM"
        ) or (
            self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and channel == self.states.diffraction_view_channel
        ):
            self.normalizer.set_alpha_beta(alpha, beta, self.histogram.amplify[self.histogram.channel])

    def enable_histogram(self, name):
        if "histogram" in self.window.diffraction_tools.visualisation.selected:
            self.normalizer.histogram_enable = True
        else:
            self.normalizer.histogram_enable = False

    def _wheel_event(self, event: QtGui.QWheelEvent):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.setModifiers(QtCore.Qt.KeyboardModifier.NoModifier)  # remove it otherwise step will be 10x
            self.window.stem_adjustments.focus.wheelEvent(event, force=True)

    def _mouse_double_click(self, event: QtGui.QMouseEvent):
        # press release double release sequence will be triggered

        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and not self.window.diffraction_tools.selectors.selected
        ):
            xy = self.view.map_to_area([event.pos().x(), event.pos().y()])
            if self.window.diffraction.stem_tilt_type.selected == "electronic":
                self.change_projection_tilt(xy)
            elif self.window.diffraction.stem_tilt_type.selected == "mechanic":
                self.change_mechanical_tilt(xy)

    def _mouse_press(self, event: QtGui.QMouseEvent):
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and not self.window.diffraction_tools.selectors.selected
        ):
            self.dragging = True
            self.drag_initial_position = self.view.map_to_area([event.pos().x(), event.pos().y()])
            if self.window.diffraction.stem_tilt_type.selected == "electronic":
                if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                    self.change_illumination_stigmation(None)
                else:
                    self.change_illumination_tilt(None)

    def _mouse_move(self, event: QtGui.QMouseEvent):
        if (
            self.dragging
            and self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and not self.window.diffraction_tools.selectors.selected
        ):
            xy = self.view.map_to_area([event.pos().x(), event.pos().y()])
            if self.window.diffraction.stem_tilt_type.selected == "electronic":
                if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                    self.change_illumination_stigmation(xy)
                else:
                    self.change_illumination_tilt(xy)

    def _mouse_release(self, event: QtGui.QMouseEvent):
        if self.dragging:
            self.dragging = False

    def _leave(self, event: QtCore.QEvent):
        if self.dragging:
            self.dragging = False

    def change_illumination_stigmation(self, xy):
        if xy is None:
            self.initial_stigmation = grpc_client.illumination.get_stigmator()
        else:
            dxy = [xy[0] - self.drag_initial_position[0], xy[1] - self.drag_initial_position[1]]
            single_step = min(
                self.window.stem_adjustments.stigmator_normal.singleStep(),
                self.window.stem_adjustments.stigmator_skew.singleStep(),
            )
            factor = single_step * 1
            new_values = (
                self.initial_stigmation["x"] + dxy[0] * factor * 1e-3,
                self.initial_stigmation["y"] - dxy[1] * factor * 1e-3,
            )
            grpc_client.illumination.set_stigmator(*new_values)

            self.window.stem_adjustments.stigmator_normal.set_read_sent_value(new_values[0] * 1000)
            self.window.stem_adjustments.stigmator_skew.set_read_sent_value(new_values[1] * 1000)

    def change_illumination_tilt(self, xy):
        if xy is None:
            self.initial_tilt = grpc_client.illumination.get_tilt(grpc_client.illumination.DeflectorType.Scan)
        else:
            dxy = [xy[0] - self.drag_initial_position[0], xy[1] - self.drag_initial_position[1]]
            single_step = min(
                self.window.stem_adjustments.stigmator_normal.singleStep(),
                self.window.stem_adjustments.stigmator_skew.singleStep(),
            )
            factor = single_step * 1
            new_values = (
                self.initial_tilt["x"] + dxy[0] * factor * 1e-3,
                self.initial_tilt["y"] - dxy[1] * factor * 1e-3,
            )
            grpc_client.illumination.set_tilt(
                {
                    "x": new_values[0],
                    "y": new_values[1],
                },
                grpc_client.illumination.DeflectorType.Scan,
            )
            if self.window.diffraction.tilt_type.currentText() == "illumination":
                self.window.diffraction.tilt_x.set_read_sent_value(new_values[0] * 1000)
                self.window.diffraction.tilt_y.set_read_sent_value(new_values[1] * 1000)

    def change_projection_tilt(self, dxy):
        xy0 = grpc_client.projection.get_tilt(grpc_client.illumination.DeflectorType.Scan)
        grpc_client.projection.set_tilt(
            {"x": xy0["x"] - dxy[0] * 1e-3, "y": xy0["y"] + dxy[1] * 1e-3},
            grpc_client.projection.DeflectorType.Scan,
        )

    def change_mechanical_tilt(self, dxy):
        xy0 = grpc_client.stage.get_tilt()
        beta_enabled = grpc_client.stage.is_beta_enabled()

        if not beta_enabled:
            grpc_client.stage.set_alpha.future(
                xy0["alpha"] + dxy[1] * 1e-3, nowait=not self.window.diffraction.stage_backlash.isChecked()
            )
            self.window.diffraction.stage_alpha.set_read_sent_value((xy0["alpha"] + dxy[1] * 1e-3) / np.pi * 180)
        else:
            grpc_client.stage.set_tilt.future(
                xy0["alpha"] + dxy[1] * 1e-3,
                xy0["beta"] - dxy[0] * 1e-3,
                nowait=not self.window.diffraction.stage_backlash.isChecked(),
            )

            self.window.diffraction.stage_alpha.set_read_sent_value((xy0["alpha"] + dxy[1] * 1e-3) / np.pi * 180)
            self.window.diffraction.stage_beta.set_read_sent_value((xy0["beta"] - dxy[1] * 1e-3) / np.pi * 180)

    def save_image(self):
        filename = file_dialog.image_file_save_dialog(self.window, get_config().data.data_folder)

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
        if get_config().data.save_metadata:
            metadata = data_saver.get_microscope_metadata(get_config().connection)

            if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
                pixel_time = self.window.scanning.pixel_time_spin.value()
                camera_exposure = self.window.camera.exposure.value()

            elif self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
                pixel_time = self.window.stem_4d.pixel_time_spin.value() * 1e-3
                camera_exposure = self.window.stem_4d.pixel_time_spin.value()

            if self.states.diffraction_view_channel == "fft":
                channels = [self.states.image_view_channel]
                metadata["detectors"]["FFT"] = data_saver.get_stem_metadata(
                    self.view.main_image_item.fov, pixel_time, channels
                )
            else:
                metadata["detectors"]["Camera"] = data_saver.get_camera_metadata(
                    camera_exposure,
                    precession=self.window.precession.enabled.diffraction.selected,  # exposure ms
                    precession_angle=self.window.precession.precession_angle.value(),  # angle mrad
                    precession_frequency=self.window.precession.precession_frequency.value(),  # frequency kHz)
                )

            data_saver.save_metadata(filename, metadata)

    def save_scene(self):
        name = file_dialog.image_file_save_dialog(self.window, get_config().data.data_folder)
        if name is not None:
            pix_map = self.view.grab(self.view.sceneRect().toRect())
            pix_map.save(name)
