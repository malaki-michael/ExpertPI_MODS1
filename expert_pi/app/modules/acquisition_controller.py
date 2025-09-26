from collections.abc import Callable

import numpy as np

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.automations import precession
from expert_pi.grpc_client import stem_detector as sd
from expert_pi.gui import console_threads
from expert_pi.gui.main_window import MainWindow
from expert_pi.gui.toolbars.precession import M2x2_names
from expert_pi.stream_clients import CacheClient


class AcquisitionController:
    def __init__(self, window: MainWindow, states: StatesHolder, emitters, cache_client: CacheClient) -> None:
        self._window: MainWindow = window
        self._states = states
        self.emitters = emitters
        self._cache_client = cache_client

        self._scanning = window.scanning
        self._camera = window.camera
        self._precession = window.precession
        self._stem_4d = window.stem_4d

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._scanning = window.scanning
        self._camera = window.camera
        self._precession = window.precession
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._scanning.fov_spin.set_signal: lambda x: fov_changed(
                self._window, self._states, x, update_image_view_fov=True
            ),
            self._scanning.pixel_time_spin.set_signal: lambda _: (
                update_max_fov(self._window, self._states),
                rescan_stem(self._window, self._states),
            ),
            self._scanning.rotation_spin.set_signal: lambda x: (
                grpc_client.scanning.set_rotation(x / 180 * np.pi),
                rescan_stem(self._window, self._states),
                self._window.image_view.update_transforms(),
            ),
            self._scanning.size_combo.currentIndexChanged: lambda x: (
                update_max_fov(self._window, self._states),
                rescan_stem(self._window, self._states),
            ),
            self._scanning.control_buttons.clicked: lambda option: (
                stop_stem(self._window, self._states) if option == "stop" else rescan_stem(self._window, self._states)
            ),
            self._scanning.blanker_button.clicked: lambda value: grpc_client.scanning.set_blanker_mode(
                grpc_client.scanning.BeamBlankType[value]
            ),
            self._scanning.off_axis_butt.clicked: self.off_axis_butt_clicked,
            self._camera.exposure.set_signal: lambda x: rescan_camera(self._window, self._states),
            self._camera.fov.set_signal: self.set_camera_angle,
            self._camera.control_buttons.clicked: lambda option: (
                stop_camera(self._window, self._states)
                if option == "stop"
                else rescan_camera(self._window, self._states)
            ),
            self._camera.roi_buttons.clicked: self.camera_roi_clicked,
            self._window.image_view.selection_changed_signal: self.update_rect_diffraction,
            # self._window.image_view.selector_pos_changed: lambda position:(), #TODO
            # self._window.image_view.rescan_request: lambda :rescan_stem(self._window,self._state),
            # mask detector click -4dstem offline
            # ronchigram click
            # autodefocus
            # autostigmation
            # xyz tracking tool
            self._precession.precession_clicked_signal: self.precession_clicked,
            self._precession.precession_angle.set_signal: lambda x: grpc_client.scanning.set_precession_angle(
                x * 1e-3, skip_deprecession=not self._precession.deprecession_enabled.isChecked()
            ),
            self._precession.precession_frequency.set_signal: lambda x: grpc_client.scanning.set_precession_frequency(
                x * 1e3
            ),
            self._precession.deprecession_enabled.stateChanged: lambda value: grpc_client.scanning.set_precession_angle(
                self._precession.precession_angle.value() * 1e-3, skip_deprecession=not value
            ),
            self._precession.auto_precession.clicked: self.auto_precession_clicked,
            self._precession.auto_deprecession.clicked: self.auto_deprecession_clicked,
        }

        for name in M2x2_names:
            signal = self._precession.heights[name].set_signal
            signals[signal] = self.set_precession_height_correction
            signal = self._precession.deprecession_corrections[name].set_signal
            signals[signal] = self.set_deprecession_tilt_correction

        return signals

    # def rotate_vector(self, xy, inverse=False):
    #     gamma = self._scanning.rotation_spin.value() / 180 * np.pi
    #     if inverse:
    #         gamma = -gamma
    #     R = np.array([[np.cos(gamma), np.sin(gamma)], [np.sin(-gamma), np.cos(gamma)]])
    #     return R @ xy

    def set_camera_angle(self, value):
        grpc_client.projection.set_max_camera_angle(grpc_client.projection.DetectorType.Camera, value / 2 * 1e-3, True)

        camera_roi = grpc_client.scanning.get_camera_roi()
        fov_factor = 1
        if camera_roi["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_256.name:
            fov_factor = 2
        elif camera_roi["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_128.name:
            fov_factor = 4

        self._window.diffraction_view.main_image_item.set_fov(value / fov_factor)

        if value > 0:
            self._window.diffraction_view.set_fov(value)
            self._window.diffraction_view.update_tools()

        update_detector_angles(self._window)

        if self._camera.use_rotation.selected:
            self._window.diffraction_view.main_image_item.set_rotation(
                grpc_client.projection.get_camera_to_stage_rotation() + grpc_client.scanning.get_rotation()
            )

        rescan_camera(self._window, self._states)

    def camera_roi_clicked(self, option):
        fov_factor = 1
        if option == "full":
            grpc_client.scanning.set_camera_roi(grpc_client.scanning.RoiMode.Disabled, use16bit=True)
        elif option == "roi256":
            grpc_client.scanning.set_camera_roi(grpc_client.scanning.RoiMode.Lines_256, use16bit=True)
            fov_factor = 2
        elif option == "roi128":
            grpc_client.scanning.set_camera_roi(grpc_client.scanning.RoiMode.Lines_128, use16bit=True)
            fov_factor = 4

        self._window.diffraction_view.main_image_item.set_fov(
            grpc_client.projection.get_max_camera_angle() * 2 * 1e3 / fov_factor
        )
        rescan_camera(self._window, self._states)

    def off_axis_butt_clicked(self, value):
        grpc_client.projection.set_is_off_axis_stem_enabled(value)
        if not self._window.image_view.tools["point_selector"].is_active:
            rescan_stem(self._window, self._states, ignore_off_axis=True)

    def precession_clicked(self):
        if self._window.image_view.tools["point_selector"].isVisible():
            pos = self._window.image_view.tools["point_selector"].pos()
            self._window.image_view.point_selection_signal.emit([pos.x(), pos.y()])
        else:
            rescan_stem(self._window, self._states)

    def auto_precession_clicked(self):
        if self._precession.auto_precession.selected and (
            self._states.acquisition.automation_thread is None
            or not self._states.acquisition.automation_thread.is_alive()
        ):
            fov = self._window.scanning.fov_spin.value() * self._precession.precession_auto_factor.value()
            n = int(self._window.scanning.size_combo.currentText().split(" ")[0])
            if self._window.image_view.tools["rectangle_selector"].is_active:
                rectangle = self._window.image_view.tools["rectangle_selector"].get_scan_rectangle()
            else:
                rectangle = None

            def set_callback(value):
                for i, name in enumerate(M2x2_names):
                    self._precession.heights[name].setProperty("busy", True)
                    self._precession.heights[name].update_style()
                    self._precession.heights[name].set_read_sent_signal.emit(value[i] * 1000)

            def target():
                try:
                    precession.focus_pivot_points(
                        self._cache_client,
                        init_step_diagonal=fov,
                        init_step_non_diagonal=fov / 5,
                        kernel=self._precession.pick_precession_kernel(
                            self._precession.precession_method.currentText()
                        ),
                        blur="median",
                        n=n,
                        rectangle=rectangle,
                        pixel_time=1 / (self._precession.precession_frequency.value() * 1e3),
                        stop_callback=lambda: not self._precession.auto_precession.selected,
                        set_callback=set_callback,
                    )
                except precession.StopError:
                    pass

                self._precession.auto_precession.update_selected_signal.emit(False)

                for i, name in enumerate(M2x2_names):
                    self._precession.heights[name].setProperty("busy", False)
                    self._precession.heights[name].update_style()

            self._states.acquisition.automation_thread = console_threads.start_threaded_function(target)

    def auto_deprecession_clicked(self):
        if self._precession.auto_deprecession.selected and (
            self._states.acquisition.automation_thread is None
            or not self._states.acquisition.automation_thread.is_alive()
        ):
            angle = self._precession.precession_angle.value() * self._precession.deprecession_auto_factor.value()

            if self._window.image_view.tools["point_selector"].is_active:
                pos = self._window.image_view.tools["point_selector"].pos()
                fov = self._window.image_view.main_image_item.fov
                N = self._window.image_view.main_image_item.image.shape[0]
                ij = [int(pos.y() / fov * N + 0.5) + N // 2, int(pos.x() / fov * N + 0.5) + N // 2]

            else:
                N = 1024
                ij = [512, 512]

            def set_callback(value):
                for i, name in enumerate(M2x2_names):
                    self._precession.deprecession_corrections[name].setProperty("busy", True)
                    self._precession.deprecession_corrections[name].update_style()
                    self._precession.deprecession_corrections[name].set_read_sent_signal.emit(value[i])

            def target():
                try:
                    precession.optimize_deprecession(
                        self._cache_client,
                        init_step_diagonal=angle / 2,
                        init_step_non_diagonal=angle / 10,
                        kernel=self._precession.pick_deprecession_kernel(
                            self._precession.deprecession_method.currentText()
                        ),
                        blur="median",
                        n=N,
                        ij=ij,
                        camera_time=self._window.camera.exposure.value() * 1e-3,
                        stop_callback=lambda: not self._precession.auto_deprecession.selected,
                        set_callback=set_callback,
                    )

                except precession.StopError:
                    pass

                self._precession.auto_deprecession.update_selected_signal.emit(False)

                for i, name in enumerate(M2x2_names):
                    self._precession.deprecession_corrections[name].setProperty("busy", False)
                    self._precession.deprecession_corrections[name].update_style()

            self._states.acquisition.automation_thread = console_threads.start_threaded_function(target)

    def set_precession_height_correction(self):
        M = []
        for name in M2x2_names:
            M.append(self._precession.heights[name].value() * 1e-9)
        grpc_client.scanning.set_precession_height_correction(M)

    def set_deprecession_tilt_correction(self):
        M = []
        for name in M2x2_names:
            M.append(self._precession.deprecession_corrections[name].value() * 1e-3)
        grpc_client.scanning.set_deprecession_tilt_correction(M)

    def update_rect_diffraction(self):
        if self._states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
            return

        for detector in ["BF", "HAADF"]:
            # do not process the data untill inner rectangle is set:
            self.emitters[detector + "_accumulator"].minimal_scan_id = (
                self.emitters[detector + "_accumulator"]._scan_index + 1
            )

        rescan_stem(self._window, self._states)


def synchronize(window: MainWindow, states: StatesHolder):
    update_max_fov(window, states)  # must be before updating fov
    window.scanning.fov_spin.set_read_sent_value(grpc_client.scanning.get_field_width() * 1e6)
    window.scanning.blanker_button.set_state(grpc_client.scanning.get_blanker_mode().name)
    window.scanning.off_axis_butt.set_selected(grpc_client.projection.get_is_off_axis_stem_enabled())
    window.scanning.rotation_spin.set_read_sent_value(grpc_client.scanning.get_rotation() * 180 / np.pi)
    # window.image_view.update_transforms()

    # sync camera
    camera_roi = grpc_client.scanning.get_camera_roi()
    fov_factor = 1
    if camera_roi["roi_mode"].name == grpc_client.scanning.RoiMode.Disabled.name:
        window.camera.roi_buttons.option_clicked("full", emit=False)
    elif camera_roi["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_256.name:
        window.camera.roi_buttons.option_clicked("roi256", emit=False)
        fov_factor = 2
    elif camera_roi["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_128.name:
        window.camera.roi_buttons.option_clicked("roi128", emit=False)
        fov_factor = 4

    try:
        # these will give error if not any optical mode selected:
        window.camera.fov.set_read_sent_value(grpc_client.projection.get_max_camera_angle() * 2 * 1e3)
        window.diffraction_view.set_fov(window.camera.fov.value())
        window.diffraction_view.main_image_item.set_fov(window.camera.fov.value() / fov_factor)

        if window.camera.use_rotation.selected:
            if states.diffraction_view_channel == "camera":
                window.diffraction_view.main_image_item.set_rotation(
                    grpc_client.projection.get_camera_to_stage_rotation() + grpc_client.scanning.get_rotation()
                )
    except:
        pass
    update_detector_angles(window)

    update_max_fov(window, states)

    if grpc_client.scanning.camera_is_acquiring():
        if window.camera.control_buttons.selected not in ["1x", "start"]:
            window.camera.control_buttons.buttons["stop"].setProperty("selected", False)
            window.camera.control_buttons.buttons["1x"].setProperty("selected", True)
            window.camera.control_buttons.selected = "1x"
    window.camera.control_buttons.setStyleSheet(window.camera.control_buttons.styleSheet())

    # precession
    angle = grpc_client.scanning.get_precession_angle()
    window.precession.precession_angle.set_read_sent_value(angle * 1e3)

    try:
        height = grpc_client.scanning.get_precession_height_correction()
        for i, name in enumerate(M2x2_names):
            window.precession.heights[name].set_read_sent_value(height[i] * 1e9)

        correction = grpc_client.scanning.get_deprecession_tilt_correction()
        for i, name in enumerate(M2x2_names):
            window.precession.deprecession_corrections[name].set_read_sent_value(correction[i] * 1e3)
    except:
        pass


def fov_changed(window: MainWindow, states: StatesHolder, value, update_image_view_fov=False):
    window.scanning.fov_spin.update_read_value(grpc_client.scanning.set_field_width(value * 1e-6) * 1e6)
    if update_image_view_fov and window.scanning.fov_spin.value() > 0:
        window.image_view.set_fov(window.scanning.fov_spin.value())
        window.image_view.update_tools()
    rescan_stem(window, states)


def update_detector_angles(window: MainWindow):
    try:
        # TODO maybe use future to speed up?
        angles = grpc_client.projection.get_max_detector_angles()

        window.camera.HAADF_angles.setText(
            f"HAADF: {angles['haadf']['start']*1000:5.1f}-{angles['haadf']['end']*1000:5.1f}mrad"
        )
        window.camera.BF_angles.setText(f"BF: {angles['bf']['end']*1000:5.1f}mrad")

    except:
        window.camera.HAADF_angles.setText("HAADF: ?-?mrad")
        window.camera.BF_angles.setText("BF: ?-?mrad")


def update_max_fov(window: MainWindow, states: StatesHolder):
    result = grpc_client.scanning.get_field_width_range(
        window.scanning.pixel_time_spin.value() * 1e-6, int(window.scanning.size_combo.currentText().split(" ")[0])
    )
    states.acquisition.max_fov = result["end"] * 1e6
    window.scanning.fov_spin.setMaximum(states.acquisition.max_fov)


def rescan_stem(window: MainWindow, states: StatesHolder, ignore_off_axis=False):
    if window.scanning.control_buttons.selected != "stop":
        if window.scanning.control_buttons.selected == "start":
            frames = 0
        else:
            frames = 1
        start_stem(window, states, frames, ignore_off_axis=ignore_off_axis)


def start_stem(window: MainWindow, states: StatesHolder, frames, ignore_off_axis=False, **kwargs):
    scan_widget = window.scanning

    stop_automation(window, states)
    if states.acquisition.max_fov is None:
        update_max_fov(window, states)
    window.image_view.main_image_item.set_fov(scan_widget.fov_spin.value())

    N = int(scan_widget.size_combo.currentText().split(" ")[0])
    if frames == 0:
        scan_widget.blanker_button.set_state("BeamOn")
    else:
        scan_widget.blanker_button.set_state("BeamAcq")

    detectors = [sd.DetectorType.BF, sd.DetectorType.HAADF] + [
        sd.DetectorType[edx] for edx in window.xray.get_selected_detectors()
    ]

    # off axis detector change
    if not ignore_off_axis:
        if not window.detectors.BF_insert.selected and not window.detectors.HAADF_insert.selected:
            if not grpc_client.projection.get_is_off_axis_stem_enabled():
                grpc_client.projection.set_is_off_axis_stem_enabled(True)
                scan_widget.off_axis_butt.set_selected(True)
        elif grpc_client.projection.get_is_off_axis_stem_enabled():
            grpc_client.projection.set_is_off_axis_stem_enabled(False)
            scan_widget.off_axis_butt.set_selected(False)

    rectangle = None

    if window.image_view.tools["rectangle_selector"].is_active:
        rectangle = window.image_view.tools["rectangle_selector"].get_scan_rectangle()
        window.image_view.normalizer.rectangle = rectangle
    else:
        window.image_view.normalizer.rectangle = None

    pixel_time = scan_widget.pixel_time_spin.value()
    if window.precession.enabled.image.selected:
        grpc_client.scanning.set_precession_angle(
            window.precession.precession_angle.value() * 1e-3,
            skip_deprecession=not window.precession.deprecession_enabled.isChecked(),
        )  # we need to specifically call this because of xyz tracking
        grpc_client.scanning.set_precession_frequency(window.precession.precession_frequency.value() * 1000)
        pixel_time = max(
            scan_widget.pixel_time_spin.value(),
            window.precession.get_minimal_pixel_time() * window.precession.cycles_in_stem.value(),
        )

    scan_id = scan_helper.start_rectangle_scan(
        pixel_time=pixel_time * 1e-6,
        rectangle=rectangle,
        total_size=N,
        frames=frames,
        detectors=detectors,
        is_precession_enabled=window.precession.enabled.image.selected,
    )
    return scan_id, N, rectangle, pixel_time


def rescan_camera(window: MainWindow, states: StatesHolder):
    if window.camera.control_buttons.selected != "stop":
        if window.camera.control_buttons.selected == "start":
            frames = 0
        else:
            frames = 1
        start_camera(window, states, frames)


def start_camera(window: MainWindow, states: StatesHolder, frames, ignore_off_axis=False, **kwargs):
    stop_automation(window, states)

    if not ignore_off_axis and grpc_client.projection.get_is_off_axis_stem_enabled():
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
        window.scanning.off_axis_butt.set_selected(False)

    if window.camera.roi_buttons.selected != "full":  # for full mode the bit range will be selected automatically
        if window.camera.roi_buttons.selected == "roi256":
            roi_mode = grpc_client.scanning.RoiMode.Lines_256
        elif window.camera.roi_buttons.selected == "roi128":
            roi_mode = grpc_client.scanning.RoiMode.Lines_128

        if window.camera.exposure.value() < 1000 / 2250:  # auto switching bit range
            grpc_client.scanning.set_camera_roi(roi_mode, use16bit=False)
        else:
            grpc_client.scanning.set_camera_roi(roi_mode, use16bit=True)

    grpc_client.scanning.start_camera(
        window.camera.exposure.value() * 1e-3, 1e3 / window.camera.exposure.value(), frames
    )
    window.scanning.blanker_button.set_state("BeamOn")


def start_4dstem(
    window: MainWindow, states: StatesHolder, frames, auto_retract=True, ignore_off_axis=False, rectangle=None, **kwargs
):
    # this is used also for single point selector
    stop_automation(window, states)

    scan_widget = window.scanning

    window.image_view.main_image_item.set_fov(scan_widget.fov_spin.value())

    if frames == 0:
        window.scanning.blanker_button.set_state("BeamOn")
    else:
        window.scanning.blanker_button.set_state("BeamAcq")

    if auto_retract:
        futures = {}
        for name in ["BF", "HAADF"]:
            if window.detectors.__dict__[name + "_insert"].selected:
                print(
                    "retracting",
                    name,
                )
                futures[name] = sd.set_is_inserted.future(sd.DetectorType[name], False)
                window.detectors.__dict__[name + "_insert"].set_busy(True)
                futures[name].add_done_callback(
                    (lambda name_: (lambda x: window.detectors.update_detectors_state(name_, x.result())))(name)
                )

        # wait for retraction
        # TODO this will not show the progress on UI:
        for future in futures.values():
            future.result()

    detectors = [sd.DetectorType.Camera] + [sd.DetectorType[edx] for edx in window.xray.get_selected_detectors()]

    # off axis detector change
    if not ignore_off_axis:
        if grpc_client.projection.get_is_off_axis_stem_enabled():
            grpc_client.projection.set_is_off_axis_stem_enabled(False)
            scan_widget.off_axis_butt.set_selected(False)

    if (
        states.interaction_mode == StatesHolder.InteractionMode.survey
        and window.image_view.tools["point_selector"].is_active
    ):
        N = int(scan_widget.size_combo.currentText().split(" ")[0])
        pixel_time = window.camera.exposure.value() * 1e-3  # to seconds

    elif window.image_view.tools["rectangle_selector"].is_active:
        raise Exception("rectangle selector of 4DSTEM not yet supported")

    else:
        # window.image_view.inner_rectangle = None # TODO
        N = int(window.stem_4d.size_combo.currentText().split(" ")[0])
        pixel_time = window.stem_4d.pixel_time_spin.value() * 1e-3  # to seconds

    if window.camera.roi_buttons.selected != "full":  # for full mode the bit range will be selected automatically
        if window.camera.roi_buttons.selected == "roi256":
            roi_mode = grpc_client.scanning.RoiMode.Lines_256
        elif window.camera.roi_buttons.selected == "roi128":
            roi_mode = grpc_client.scanning.RoiMode.Lines_128

        if pixel_time < 1 / 2250:  # auto switching bit range
            grpc_client.scanning.set_camera_roi(roi_mode, use16bit=False)
        else:
            grpc_client.scanning.set_camera_roi(roi_mode, use16bit=True)

    if window.precession.enabled.diffraction.selected:
        grpc_client.scanning.set_precession_angle(
            window.precession.precession_angle.value() * 1e-3,
            skip_deprecession=not window.precession.deprecession_enabled.isChecked(),
        )  # we need to specifically call this because of xyz tracking
        grpc_client.scanning.set_precession_frequency(window.precession.precession_frequency.value() * 1000)
        pixel_time = max(
            pixel_time,
            window.precession.get_minimal_pixel_time() * 1e-6,
        )
    scan_id = scan_helper.start_rectangle_scan(
        pixel_time=pixel_time,
        rectangle=rectangle,
        total_size=N,
        frames=frames,
        detectors=detectors,
        camera_exposure=0,  # automatic
        is_precession_enabled=window.precession.enabled.diffraction.selected,
    )

    return scan_id, N, rectangle, pixel_time * 1e6


def stop_stem(window: MainWindow, states: StatesHolder, **kwargs):
    grpc_client.scanning.stop_scanning()
    window.scanning.blanker_button.set_state("BeamOff")
    states.acquisition.stem_running = False


def stop_camera(window: MainWindow, states: StatesHolder, **kwargs):
    grpc_client.scanning.stop_camera()
    window.scanning.blanker_button.set_state("BeamOff")


def start_automation(window: MainWindow, states: StatesHolder, target: Callable, **kwargs):
    stop_automation(window, states)
    states.acquisition.automation_thread = console_threads.start_threaded_function(target)


def stop_automation(window: MainWindow, states: StatesHolder, **kwargs):
    if states.acquisition.automation_thread is not None and states.acquisition.automation_thread.is_alive():
        states.acquisition.automation_thread.try_stop()
