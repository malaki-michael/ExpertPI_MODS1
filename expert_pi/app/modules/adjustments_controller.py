import numpy as np

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.app.modules import acquisition_controller
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.automations import autofocus
from expert_pi.grpc_client import stem_detector as sd
from expert_pi.gui import console_threads
from expert_pi.gui.main_window import MainWindow
from expert_pi.gui.style import coloring
from expert_pi.measurements import shift_measurements
from expert_pi.stream_clients import CacheClient


class AdjustmentsController:
    def __init__(self, window: MainWindow, states: StatesHolder, emitters, cache_client: CacheClient) -> None:
        self._window = window
        self.emitters = emitters
        self._cache_client = cache_client
        self._states = states
        self._adjustments = window.stem_adjustments
        self._diffraction = window.diffraction

        self._signals = self._create_signals()
        self.connect_signals(window)

        self.tilt_wobbler_coloring_contrast = [
            1.7,
            0.6,
        ]  # first number is amplification of differences, second contrast of averaged image

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._adjustments = window.stem_adjustments
        self._diffraction = window.diffraction

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._adjustments.tilt_wobbler.clicked: self.start_tilt_wobbling,
            self._adjustments.z_set_eucentric.clicked: self.z_set_eucentric_clicked,
            self._adjustments.focus_auto.clicked: self.autofocus_clicked,
            self._adjustments.stigmators_auto.clicked: self.start_auto_stigmation,
            self._adjustments.stage_z.set_signal: lambda x: grpc_client.stage.set_z.future(
                x * 1e-6, not self._adjustments.stage_backlash.isChecked(), True, True
            ),
            self._adjustments.stage_x.set_signal: lambda x: grpc_client.stage.set_x_y.future(
                x * 1e-6,
                self._adjustments.stage_y.sent_value * 1e-6,
                not self._adjustments.stage_backlash.isChecked(),
                True,
                True,
            ),
            self._adjustments.stage_y.set_signal: lambda x: grpc_client.stage.set_x_y.future(
                self._adjustments.stage_x.sent_value * 1e-6,
                x * 1e-6,
                not self._adjustments.stage_backlash.isChecked(),
                True,
                True,
            ),
            self._adjustments.mechanic_stop.clicked: grpc_client.stage.stop,
            self._adjustments.focus.set_signal: lambda x: grpc_client.illumination.set_condenser_defocus.future(
                x * 1e-6, grpc_client.illumination.CondenserFocusType.C3
            ),  # TODO allow objective focus
            self._adjustments.shift_x.set_signal: lambda x: grpc_client.illumination.set_shift(
                {"x": x * 1e-9, "y": self._adjustments.shift_y.sent_value * 1e-9},
                grpc_client.illumination.DeflectorType.Scan,
            ),  # TODO
            self._adjustments.shift_y.set_signal: lambda y: grpc_client.illumination.set_shift(
                {"x": self._adjustments.shift_x.sent_value * 1e-9, "y": y * 1e-9},
                grpc_client.illumination.DeflectorType.Scan,
            ),  # TODO
            self._adjustments.stigmator_normal.set_signal: lambda x: grpc_client.illumination.set_stigmator(
                x * 1e-3, self._adjustments.stigmator_skew.sent_value * 1e-3
            ),
            self._adjustments.stigmator_skew.set_signal: lambda y: grpc_client.illumination.set_stigmator(
                self._adjustments.stigmator_normal.sent_value * 1e-3, y * 1e-3
            ),
            # diffraction
            self._diffraction.tilt_type.currentIndexChanged: lambda x: tilt_type_changed(self._window, x),
            self._diffraction.tilt_x.set_signal: lambda x: self.set_tilt(x, self._diffraction.tilt_y.sent_value),
            self._diffraction.tilt_y.set_signal: lambda y: self.set_tilt(self._diffraction.tilt_x.sent_value, y),
            self._diffraction.stage_alpha.set_signal: lambda x: grpc_client.stage.set_alpha.future(
                x / 180 * np.pi, not self._diffraction.stage_backlash.isChecked(), True
            ),
            self._diffraction.stage_beta.set_signal: lambda x: grpc_client.stage.set_beta.future(
                x / 180 * np.pi, not self._diffraction.stage_backlash.isChecked(), True
            ),
        }

        return signals

    def z_set_eucentric_clicked(self):
        calibration = grpc_client.stage.get_calibration()
        calibration["z"] -= self._adjustments.stage_z.value() / 1e6
        grpc_client.stage.set_calibration(
            calibration["x"],
            calibration["y"],
            calibration["z"],
            calibration["alpha"],
            calibration["beta"],
            calibration["gamma"],
        )
        self._adjustments.stage_z.set_read_sent_value(0)

    def set_tilt(self, x, y):
        if self._diffraction.tilt_type.currentText() == "illumination":
            grpc_client.illumination.set_tilt(
                {"x": x * 1e-3, "y": y * 1e-3}, grpc_client.illumination.DeflectorType.Scan
            )
        elif self._diffraction.tilt_type.currentText() == "projection":
            grpc_client.projection.set_tilt({"x": x * 1e-3, "y": y * 1e-3}, grpc_client.projection.DeflectorType.Scan)

    def autofocus_clicked(self):
        if self._adjustments.focus_auto.selected and (
            self._states.acquisition.automation_thread is None
            or not self._states.acquisition.automation_thread.is_alive()
        ):
            defocus_init = self._window.scanning.fov_spin.value() * 1e-6
            pixel_time = self._window.scanning.pixel_time_spin.value() * 1e-6
            n = int(self._window.scanning.size_combo.currentText().split(" ")[0])
            if self._window.image_view.tools["rectangle_selector"].is_active:
                rectangle = self._window.image_view.tools["rectangle_selector"].get_scan_rectangle()
            else:
                rectangle = None

            def set_callback(value):
                self._adjustments.focus.setProperty("busy", True)
                self._adjustments.focus.update_style()
                self._adjustments.focus.set_read_sent_signal.emit(value * 1e6)

            def target():
                try:
                    autofocus.focus_by_C3(
                        self._cache_client,
                        defocus_init=defocus_init,
                        kernel="sobel",
                        blur="median",
                        n=n,
                        rectangle=rectangle,
                        pixel_time=pixel_time,
                        max_iterations=5,
                        steps=7,
                        output=False,
                        stop_callback=lambda: not self._adjustments.focus_auto.selected,
                        set_callback=set_callback,
                        is_precession_enabled=self._window.precession.enabled.image.selected,
                    )
                except autofocus.StopException:
                    pass

                self._adjustments.focus_auto.update_selected_signal.emit(False)
                self._adjustments.focus.setProperty("busy", False)
                self._adjustments.focus.update_style()

            self._cache_client.connect()
            self._states.acquisition.automation_thread = console_threads.start_threaded_function(target)

    def start_tilt_wobbling(self):
        if self._adjustments.tilt_wobbler.selected and (
            self._states.acquisition.automation_thread is None
            or not self._states.acquisition.automation_thread.is_alive()
        ):
            if self._adjustments.tilt_wobbler_direction.currentText() == "X":
                tilt_factors = np.array([[0, 0], [1, 0]])
            else:
                tilt_factors = np.array([[0, 0], [0, 1]])

            fov = self._window.scanning.fov_spin.value()

            N = int(self._window.scanning.size_combo.currentText().split(" ")[0])

            grpc_client.scanning.set_precession_angle(self._adjustments.tilt_wobbler_angle.value() * 1e-3)
            grpc_client.scanning.set_precession_frequency(0)
            init_precession_heights = grpc_client.scanning.get_precession_height_correction()
            grpc_client.scanning.set_precession_height_correction([0, 0, 0, 0])

            if self._window.image_view.tools["rectangle_selector"].is_active:
                rectangle = self._window.image_view.tools["rectangle_selector"].get_scan_rectangle()
                total_pixels = rectangle[2] * rectangle[3]
                shape = (rectangle[2], rectangle[3])
            else:
                rectangle = np.array([0, 0, N, N])
                total_pixels = N**2
                shape = (N, N)

            # max_repeats = 2**12

            image = self._window.image_view.main_image_item.image * 1
            if len(image.shape) > 2:
                image = np.mean(image, dtype="uint8", axis=2)
            base_image = np.dstack([image, image, image])

            def target():
                while self._adjustments.tilt_wobbler.selected:
                    # do not use continuous acquisition, otherwise there might be big delay between cache and data
                    scan_id = scan_helper.start_multi_series(
                        pixel_time=self._window.scanning.pixel_time_spin.value() * 1e-6,
                        rectangle=rectangle,
                        total_size=N,
                        frames=1,
                        detectors=[sd.DetectorType.BF, sd.DetectorType.HAADF],
                        tilt_factors=tilt_factors,
                    )
                    try:
                        _header, data = self._cache_client.get_item(scan_id, total_pixels * len(tilt_factors))
                        imgs = data["stemData"]["BF"].reshape(len(tilt_factors), shape[0], shape[1])

                        shift = shift_measurements.get_offset_of_pictures(
                            imgs[0], imgs[1], np.array(fov * rectangle[2:4] / N)
                        )

                        directional_shift = np.sum(shift * tilt_factors[1, :] / np.sum(tilt_factors[1, :] ** 2))
                        z = directional_shift / (self._adjustments.tilt_wobbler_angle.value() * 1e-3)  # in um

                        self._adjustments.tilt_wobbler_set.set_value = z
                        self._adjustments.tilt_wobbler_set.update_text_signal.emit(f"{z:+6.3f} um")

                        image_8b = coloring.get_colored_differences(
                            imgs[0],
                            imgs[1],
                            self.tilt_wobbler_coloring_contrast[0],
                            self.tilt_wobbler_coloring_contrast[1],
                        )

                        if rectangle is None:
                            self._window.image_view.main_image_item.set_image(image_8b)
                        else:
                            base_image[
                                rectangle[0] : rectangle[0] + rectangle[2],
                                rectangle[1] : rectangle[1] + rectangle[3],
                                :,
                            ] = image_8b
                            self._window.image_view.main_image_item.set_image(base_image, rectangle)

                        self._window.image_view.update_signal.emit()
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        self._adjustments.tilt_wobbler.update_selected_signal.emit(False)
                        break
                self._adjustments.tilt_wobbler_set.update_enabled_signal.emit(False)

                grpc_client.scanning.stop_scanning()
                grpc_client.scanning.set_precession_height_correction(init_precession_heights)
                self.emitters[self._states.image_view_channel + "_accumulator"].start()
                acquisition_controller.rescan_stem(self._window, self._states)

            self.emitters[self._states.image_view_channel + "_accumulator"].stop()

            self._cache_client.connect()

            self._adjustments.tilt_wobbler_set.setEnabled(True)

            self._states.acquisition.automation_thread = console_threads.start_threaded_function(target)

    def start_auto_stigmation(self):
        if self._adjustments.stigmators_auto.selected and (
            self._states.acquisition.automation_thread is None
            or not self._states.acquisition.automation_thread.is_alive()
        ):
            fov = self._window.scanning.fov_spin.value()
            pixel_time = self._window.scanning.pixel_time_spin.value() * 1e-6
            n = int(self._window.scanning.size_combo.currentText().split(" ")[0])
            if self._window.image_view.tools["rectangle_selector"].is_active:
                rectangle = self._window.image_view.tools["rectangle_selector"].get_scan_rectangle()
            else:
                rectangle = None

            def set_callback(value):
                for i, item in enumerate([self._adjustments.stigmator_normal, self._adjustments.stigmator_skew]):
                    item.setProperty("busy", True)
                    item.update_style()
                    item.set_read_sent_signal.emit(value[i])

            def target():
                try:
                    autofocus.auto_stigmation(
                        self._cache_client,
                        init_step=min(50.0, fov * 10),
                        kernel="sobel",
                        blur="median",
                        n=n,
                        rectangle=rectangle,
                        pixel_time=pixel_time,
                        stop_callback=lambda: not self._adjustments.stigmators_auto.selected,
                        set_callback=set_callback,
                        is_precession_enabled=self._window.precession.enabled.image.selected,
                    )
                except autofocus.StopException:
                    pass

                self._adjustments.stigmators_auto.update_selected_signal.emit(False)

                for i, item in enumerate([self._adjustments.stigmator_normal, self._adjustments.stigmator_skew]):
                    item.setProperty("busy", False)
                    item.update_style()

            self._cache_client.connect()
            self._states.acquisition.automation_thread = console_threads.start_threaded_function(target)


def synchronize(window: MainWindow, states: StatesHolder):
    xy = grpc_client.stage.get_x_y()
    window.stem_adjustments.stage_x.set_read_sent_value(xy["x"] * 1e6)
    window.stem_adjustments.stage_y.set_read_sent_value(xy["y"] * 1e6)

    z = grpc_client.stage.get_z()
    window.stem_adjustments.stage_z.set_read_sent_value(z * 1e6)

    exy = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)  # TODO
    window.stem_adjustments.shift_x.set_read_sent_value(exy["x"] * 1e9)
    window.stem_adjustments.shift_y.set_read_sent_value(exy["y"] * 1e9)

    e_z = grpc_client.illumination.get_condenser_defocus(grpc_client.illumination.CondenserFocusType.C3)  # TODO
    window.stem_adjustments.focus.set_read_sent_value(e_z * 1e6)

    q_ns = grpc_client.illumination.get_stigmator()
    window.stem_adjustments.stigmator_normal.set_read_sent_value(q_ns["x"] * 1e3)
    window.stem_adjustments.stigmator_skew.set_read_sent_value(q_ns["y"] * 1e3)

    # diffrection
    tilts = grpc_client.stage.get_tilt()
    window.diffraction.stage_alpha.set_read_sent_value(tilts["alpha"] / np.pi * 180)
    window.diffraction.stage_beta.set_read_sent_value(tilts["beta"] / np.pi * 180)

    window.diffraction.stage_beta.setEnabled(grpc_client.stage.is_beta_enabled())

    tilt_type_changed(window, None)


def tilt_type_changed(window: MainWindow, index):
    if window.diffraction.tilt_type.currentText() == "illumination":
        exy = grpc_client.illumination.get_tilt(grpc_client.illumination.DeflectorType.Scan)
        window.diffraction.tilt_x.set_read_sent_value(exy["x"] * 1e3)
        window.diffraction.tilt_y.set_read_sent_value(exy["y"] * 1e3)
    elif window.diffraction.tilt_type.currentText() == "projection":
        exy = grpc_client.projection.get_tilt(grpc_client.illumination.DeflectorType.Scan)
        window.diffraction.tilt_x.set_read_sent_value(exy["x"] * 1e3)
        window.diffraction.tilt_y.set_read_sent_value(exy["y"] * 1e3)
