from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.app.modules import xyz_tracking_controller
    from expert_pi.app.states.states_holder import StatesHolder
    from expert_pi.gui.main_window import MainWindow

import numpy as np
from PySide6 import QtCore

from expert_pi.automations import xyz_tracking
from expert_pi.automations.xyz_tracking import nodes, objects
from expert_pi.stream_processors import process_thread


class DebugConnector(QtCore.QObject):
    set_input_signal = QtCore.Signal(object)
    set_output_signal = QtCore.Signal(object, object)
    set_error_signal = QtCore.Signal()

    def __init__(self, input_process, output_process, error_process=lambda: None):
        super().__init__()
        self.set_input_signal.connect(input_process)
        self.set_output_signal.connect(output_process)
        self.set_error_signal.connect(error_process)


class TiltAxesController(QtCore.QObject):
    def __init__(
        self,
        window: MainWindow,
        states: StatesHolder,
        xyz_tracking_controller: xyz_tracking_controller.XYZTrackingController,
    ):
        super().__init__()
        self.window = window
        self.states = states
        self.xyz_tracking_controller = xyz_tracking_controller

        self.tilt_tuner = self.window.diffraction.tilt_axis_tuner

        self.tilt_tuner.start_button.clicked.connect(
            lambda: self.start() if self.tilt_tuner.start_button.selected else self.stop()
        )
        self.tilt_tuner.clear_button.clicked.connect(self.clear)

        self.tilt_tuner.filter_value.valueChanged.connect(lambda: self.tilt_tuner.update_plots_signal.emit())
        self.tilt_tuner.filter_type.currentIndexChanged.connect(lambda: self.tilt_tuner.update_plots_signal.emit())

        self.thread = None
        self.running = False

        self.use_beta = False

        self.diagnostic_data = []
        self.saving_node = None

    def init_saving(self):
        if self.saving_node is None:
            node_manager = xyz_tracking.manager
            register_node = xyz_tracking.manager.get_node_by_name("registration")
            self.saving_node = nodes.Node(self.save_diagnostic, name="saving_tilt_axes")
            register_node.output_nodes.append(self.saving_node)  # Watchout for reloading of tomography code
            node_manager.nodes.append(self.saving_node)

    def save_diagnostic(self, input: objects.NodeImage):
        if self.running:
            self.diagnostic_data.append(input)

            alpha = input.stage_ab[0] / np.pi * 180
            beta = input.stage_ab[1] / np.pi * 180
            xy = (
                input.stage_xy
                - np.dot(input.transform2x2, input.offset_xy)
                + np.dot(input.transform2x2, input.shift_electronic - input.reference_object.shift_electronic)
                - input.reference_object.stage_xy
            )
            z = input.stage_z - input.offset_z - input.reference_object.stage_z

            if xyz_tracking.stage_shifting._use_beta_instead:
                self.tilt_tuner.angle_plot_lines["base"].addData(beta, 0)
                self.tilt_tuner.angle_plot_lines["x"].addData(beta, xy[0])
                self.tilt_tuner.angle_plot_lines["y"].addData(beta, xy[1])
                self.tilt_tuner.angle_plot_lines["z"].addData(beta, z)

                self.tilt_tuner.plane_plot_lines["item"].addData(xy[0], z)

            else:
                self.tilt_tuner.angle_plot_lines["base"].addData(alpha, 0)
                self.tilt_tuner.angle_plot_lines["x"].addData(alpha, xy[0])
                self.tilt_tuner.angle_plot_lines["y"].addData(alpha, xy[1])
                self.tilt_tuner.angle_plot_lines["z"].addData(alpha, z)

                self.tilt_tuner.plane_plot_lines["item"].addData(xy[1], z)

            self.tilt_tuner.update_plots_signal.emit()

    def start(self):
        self.init_saving()
        if self.thread is not None and self.thread.is_alive():
            self.thread.stop()

        self.running = True
        xyz_tracking.settings.alpha_speed = self.tilt_tuner.speed.value()
        if self.tilt_tuner.angle_type.currentText() == "beta":
            xyz_tracking.stage_shifting._use_beta_instead = True
            self.tilt_tuner.angle_plot.setLabel("bottom", "beta (deg)")
            self.tilt_tuner.plane_plot.setLabel("bottom", "x (um)")
        else:
            xyz_tracking.stage_shifting._use_beta_instead = False
            self.tilt_tuner.angle_plot.setLabel("bottom", "alpha (deg)")
            self.tilt_tuner.plane_plot.setLabel("bottom", "y (um)")

        self.xyz_tracking_controller.start_tracking(True)
        self.thread = process_thread.ProcessingThread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.stop()
        xyz_tracking.stage_shifting._use_beta_instead = False
        self.xyz_tracking_controller.goto_reference()

    def clear(self):
        self.tilt_tuner.clear_data()

    def is_running_callback(self, alpha, measured_i, total_measured):
        return self.tilt_tuner.start_button.selected, False

    def finish_callback(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.stop()
        xyz_tracking.stage_shifting._use_beta_instead = False
        self.tilt_tuner.start_button.update_selected_signal.emit(False)

    def _run(self):
        alpha_positions = np.array([self.tilt_tuner.min_angle.value(), self.tilt_tuner.max_angle.value()])

        xyz_tracking.alpha_scheduler.measure_positions(
            alpha_positions,
            is_running_callback=self.is_running_callback,
            finish_callback=self.finish_callback,
            correction_model=None,
            skip_measure=True,
        )
