import numpy as np
from PySide6 import QtCore

from expert_pi import grpc_client
from expert_pi.app import functions
from expert_pi.app.modules import acquisition_controller
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.grpc_client import stem_detector as sd
from expert_pi.gui.main_window import MainWindow

optical_mode_names = {
    grpc_client.microscope.OpticalMode.Off.name: "Off",
    grpc_client.microscope.OpticalMode.ConvergentBeam.name: "STEM",
    grpc_client.microscope.OpticalMode.LowMagnification.name: "LM",
    grpc_client.microscope.OpticalMode.ParallelBeam.name: "Parallel",
}


class MicroscopeController(QtCore.QObject):
    synchronize_after_mode_change_signal = QtCore.Signal(bool)

    def __init__(self, window: MainWindow, states: StatesHolder) -> None:
        super().__init__()
        self._window = window
        self._states = states

        self._signals = self._create_signals()
        self.connect_signals(window)
        self.synchronize_after_mode_change_signal.connect(self.synchronize_after_mode_change)

    def connect_signals(self, window: MainWindow):
        self._window = window
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

        for name in optical_mode_names.values():
            self._window.optics.mode.addItem(name)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._window.optics.mode.activated: self.mode_changed,
            self._window.optics.current.set_signal: lambda x: self.set_illumination(),
            self._window.optics.angle.set_signal: lambda x: self.set_illumination(),
            self._window.optics.diameter.set_signal: lambda x: self.set_illumination(),
            self._window.optics.state_button.clicked: self.state_change_button_clicked,
            self._window.detectors.BF_scale.set_signal: lambda x: grpc_client.stem_detector.set_scale(
                grpc_client.stem_detector.DetectorType.BF, x
            ),
            self._window.detectors.HAADF_scale.set_signal: lambda x: grpc_client.stem_detector.set_scale(
                grpc_client.stem_detector.DetectorType.HAADF, x
            ),
            self._window.detectors.BF_insert.clicked: self.BF_insert_click,
            self._window.detectors.HAADF_insert.clicked: self.HAADF_insert_click,
            self.synchronize_after_mode_change_signal: self.synchronize_after_mode_change,
        }

        return signals

    def synchronize_after_mode_change(self, apply_adjustments):
        if apply_adjustments:
            if self._states.optical_mode in self._states.adjustments_per_mode:
                p = self._states.adjustments_per_mode[self._states.optical_mode]
                print("setting ad", self._states.optical_mode, p)
                if "C3_defocus" in p:
                    grpc_client.illumination.set_condenser_defocus(
                        p["C3_defocus"], grpc_client.illumination.CondenserFocusType.C3
                    )
                if "stigmator" in p:
                    grpc_client.illumination.set_stigmator(p["stigmator"]["x"], p["stigmator"]["y"])
                if "tilt" in p:
                    grpc_client.illumination.set_tilt(p["tilt"], grpc_client.illumination.DeflectorType.Scan)

        self.synchronize()
        # self.main_window.synchronize_modules(["detectors", "scanning", "precession", "diffraction", "camera"])

    def synchronize(self):
        self._window.optics.energy.setText(f"{int(self._states.energy_value / 1000)} keV")
        self._window.diffraction_view.tools["status_info"].wave_length = functions.get_wavelength(
            self._states.energy_value
        )  # TODO might be better to introduce synchronize of diffraction controller?

        microscope_state = grpc_client.microscope.get_state().name
        self._window.optics.state_button.setText(microscope_state)
        self._states.last_microscope_state["substate"] = microscope_state

        self._states.optical_mode = grpc_client.microscope.get_optical_mode().name
        if self._states.optical_mode == grpc_client.microscope.OpticalMode.ConvergentBeam.name:
            self._window.optics.set_buttons_state(True, True, False)
            self._window.optics.set_buttons_tool_tip("Illumination half angle", "Theoretical d50 spot size")
            self._window.optics.set_read_sent_values(
                grpc_client.illumination.get_current() * 1e9,
                grpc_client.illumination.get_convergence_half_angle() * 1e3,
                grpc_client.illumination.get_beam_diameter() * 1e9,
            )
            self._window.optics.diameter.units = "nm"

        elif self._states.optical_mode == grpc_client.microscope.OpticalMode.ParallelBeam.name:
            self._window.optics.set_buttons_state(True, False, True)
            self._window.optics.set_buttons_tool_tip("Theoretical incidence angle", "Beam diameter")
            self._window.optics.set_read_sent_values(
                grpc_client.illumination.get_current() * 1e9,
                grpc_client.illumination.get_convergence_half_angle() * 1e3,
                grpc_client.illumination.get_beam_diameter() * 1e6,
            )
            self._window.optics.diameter.units = "um"

        elif self._states.optical_mode == grpc_client.microscope.OpticalMode.LowMagnification.name:
            self._window.optics.set_buttons_state(True, False, False)
            self._window.optics.set_buttons_tool_tip("Estimated half angle", "Estimated spot size")
            self._window.optics.set_read_sent_values(
                grpc_client.illumination.get_current() * 1e9,
                grpc_client.illumination.get_convergence_half_angle() * 1e3,
                np.nan,
            )
            self._window.optics.diameter.units = "nm"
        else:
            self._window.optics.set_read_sent_values(np.nan, np.nan, np.nan)
            self._window.optics.set_buttons_state(False, False, False)

        self._window.optics.mode.setCurrentText(optical_mode_names[self._states.optical_mode])
        self._window.optics.mode.setProperty("busy", False)
        self._window.optics.mode.setStyleSheet(self._window.optics.mode.styleSheet())

        self._window.detectors.BF_insert.set_selected(sd.get_is_inserted(sd.DetectorType.BF))
        self._window.detectors.HAADF_insert.set_selected(sd.get_is_inserted(sd.DetectorType.HAADF))

        acquisition_controller.synchronize(self._window, self._states)

    def state_change_button_clicked(self):
        if self._window.optics.state_button.property("busy"):
            return

        substate = self._states.last_microscope_state["substate"]
        if substate in {"Acquiring", "Ready"}:
            if substate == "Acquiring":
                acquisition_controller.start_scanning(self._window, self._states, "stop")
            # TODO wait
            self.future = grpc_client.microscope.set_state.future(
                grpc_client.microscope.MicroscopeState.Standby
            )  # if we don't store it to variable sometimes it does not propagate

        elif substate == "Standby":
            self.future = grpc_client.microscope.set_state.future(grpc_client.microscope.MicroscopeState.Ready)

    def set_illumination(self):
        if self._states.optical_mode == grpc_client.microscope.OpticalMode.ParallelBeam.name:
            future = grpc_client.illumination.set_parallel_illumination_values.future(
                self._window.optics.current.sent_value * 1e-9,
                self._window.optics.diameter.sent_value * 1e-6,
                self._window.optics.keep_adjustment.isChecked(),
            )

        else:
            future = grpc_client.illumination.set_illumination_values.future(
                self._window.optics.current.sent_value * 1e-9,
                self._window.optics.angle.sent_value * 1e-3,
                self._window.optics.keep_adjustment.isChecked(),
            )

        for item in [self._window.optics.current, self._window.optics.angle, self._window.optics.diameter]:
            item.setProperty("error", False)
            item.setProperty("busy", True)
            item.update_style()

        def done_callback(result):
            for item in [self._window.optics.current, self._window.optics.angle, self._window.optics.diameter]:
                if not result.code().name == "OK":
                    item.setProperty("error", True)

                item.setProperty("busy", False)
                item.update_style()

            if self._states.optical_mode == grpc_client.microscope.OpticalMode.ParallelBeam.name:
                self._window.optics.angle.set_read_sent_signal.emit(
                    grpc_client.illumination.get_convergence_half_angle() * 1e3
                )
            elif self._states.optical_mode == grpc_client.microscope.OpticalMode.ConvergentBeam.name:
                self._window.optics.diameter.set_read_sent_signal.emit(
                    grpc_client.illumination.get_beam_diameter() * 1e9
                )  # to nm

        future.add_done_callback(done_callback)

    def mode_changed(self, index):
        mode = grpc_client.microscope.OpticalMode[list(optical_mode_names.keys())[index]]
        previous_mode = self._states.optical_mode

        if (
            mode.name == grpc_client.microscope.OpticalMode.ParallelBeam.name
            and previous_mode == grpc_client.microscope.OpticalMode.ConvergentBeam.name
        ):
            pars = self._states.adjustments_per_mode[mode.name]
            future = grpc_client.illumination.set_parallel_illumination_values.future(
                pars["illumination"][0] * 1e-9,
                pars["illumination"][1] * 1e-6,
                keep_adjustments=self._window.optics.keep_adjustment.isChecked(),
            )

            # remember last values:
            self._states.adjustments_per_mode[previous_mode] = {
                "illumination": [self._window.optics.current.value(), self._window.optics.angle.value()],  # nA, mrad
                "C3_defocus": grpc_client.illumination.get_condenser_defocus(
                    grpc_client.illumination.CondenserFocusType.C3
                ),
                "stigmator": grpc_client.illumination.get_stigmator(),
                "tilt": grpc_client.illumination.get_tilt(grpc_client.illumination.DeflectorType.Scan),
            }

            def done_callback(result):
                if result.code().name == "OK":
                    apply_adjustments = self._window.optics.keep_adjustment.isChecked()
                    self.synchronize_after_mode_change_signal.emit(apply_adjustments)

        elif (
            mode.name == grpc_client.microscope.OpticalMode.ConvergentBeam.name
            and previous_mode == grpc_client.microscope.OpticalMode.ParallelBeam.name
        ):
            pars = self._states.adjustments_per_mode[mode.name]
            future = grpc_client.illumination.set_illumination_values.future(
                pars["illumination"][0] * 1e-9,
                pars["illumination"][1] * 1e-3,
                keep_adjustments=self._window.optics.keep_adjustment.isChecked(),
            )

            # remember last values:
            self._states.adjustments_per_mode[previous_mode] = {
                "illumination": [self._window.optics.current.value(), self._window.optics.diameter.value()],  # nA, um
                "C3_defocus": grpc_client.illumination.get_condenser_defocus(
                    grpc_client.illumination.CondenserFocusType.C3
                ),
                "stigmator": grpc_client.illumination.get_stigmator(),
                "tilt": grpc_client.illumination.get_tilt(grpc_client.illumination.DeflectorType.Scan),
            }

            def done_callback(result):
                if result.code().name == "OK":
                    apply_adjustments = self._window.optics.keep_adjustment.isChecked()
                    self.synchronize_after_mode_change_signal.emit(apply_adjustments)

        else:
            future = grpc_client.microscope.set_optical_mode.future(mode)

            def done_callback(result):
                if result.code().name == "OK":
                    self.synchronize_after_mode_change_signal.emit(False)

        self._window.optics.mode.setProperty("busy", True)
        self._window.optics.mode.setStyleSheet(self._window.optics.mode.styleSheet())
        self._window.optics.current.setEnabled(False)
        self._window.optics.angle.setEnabled(False)
        self._window.optics.diameter.setEnabled(False)

        future.add_done_callback(done_callback)

    def BF_insert_click(self, value):
        future = sd.set_is_inserted.future(sd.DetectorType.BF, value)
        future.add_done_callback(lambda x: self._window.detectors.update_detectors_state("BF", x.result()))
        self._window.detectors.BF_insert.set_busy(True)

    def HAADF_insert_click(self, value):
        future = sd.set_is_inserted.future(sd.DetectorType.HAADF, value)
        future.add_done_callback(lambda x: self._window.detectors.update_detectors_state("HAADF", x.result()))
        self._window.detectors.HAADF_insert.set_busy(True)
