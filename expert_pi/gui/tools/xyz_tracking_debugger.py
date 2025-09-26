import copy

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.automations import xyz_tracking
from expert_pi.automations.xyz_tracking import objects
from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.measurements import shift_measurements


class InfoTable(QtWidgets.QWidget):
    def __init__(self, num_labels):
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.labels = []
        for l in range(num_labels):
            self.labels.append(QtWidgets.QLabel(""))
            self.layout().addWidget(self.labels[-1])

    def set_data(self, data):
        for i, d in enumerate(data):
            self.labels[i].setText(d)


class NodeViewer(QtWidgets.QWidget):
    set_input_signal = QtCore.Signal(object)
    set_output_signal = QtCore.Signal(object, object)
    set_error_signal = QtCore.Signal()

    def __init__(self, name, input_num, input_to_texts, output_num, output_to_texts):
        super().__init__()
        self.input = None
        self.output = None
        self.statistic = None

        self.input_to_texts = input_to_texts
        self.output_to_texts = output_to_texts

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.name = QtWidgets.QLabel(name)
        self.setStyleSheet("border-right:1px solid #3f3e3e;")
        self.name.setStyleSheet(
            "padding:3px 6px;font-weight:bold;color:silver;border-bottom:1px solid #3f3e3e;border-top:1px solid #3f3e3e;"
        )
        self.name.setAlignment(QtGui.Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.name)
        self.speed = QtWidgets.QLabel("-")
        self.layout().addWidget(self.speed)

        self.input_table = InfoTable(input_num)
        self.layout().addWidget(self.input_table)
        self.output_table = InfoTable(output_num)
        self.layout().addWidget(self.output_table)

        self.set_input_signal.connect(self.set_input)
        self.set_output_signal.connect(self.set_output)
        self.set_error_signal.connect(self.set_error)

    def set_input(self, input: objects.NodeImage):
        self.input = input
        self.setStyleSheet("background-color: #363548;")
        self.input_table.set_data(self.input_to_texts(input))

    def set_output(self, statistic: objects.Diagnostic, output: objects.NodeImage):
        self.output = output
        self.statistic = statistic

        percent = 100
        if statistic.loop_time:
            percent = 100 * statistic.job_time / statistic.loop_time

        self.speed.setText(f"{output.id:5} {statistic.job_time * 1000:4.0f}ms {percent:5.1f}%")
        self.output_table.set_data(self.output_to_texts(statistic, output))
        self.setStyleSheet("background-color: #302F2F;")

    def set_error(self):
        self.setStyleSheet("background-color: red;")


class LivePlotItem(pg.PlotCurveItem):
    def __init__(self, max_history_time=120, pen=None):
        super().__init__(pen=pen)
        self.max_history_time = max_history_time
        self.ts = []
        self.xs = []
        self.t0 = 0

    def addData(self, t, x):
        self.ts.append(t)
        self.xs.append(x)
        while t - self.ts[0] > self.max_history_time:
            self.ts.pop(0)
            self.xs.pop(0)
        self.setData(np.array(self.ts) - self.t0, self.xs)


class XYZTrackingDebugger(QtWidgets.QWidget):
    def __init__(self, main_window=None):
        super().__init__()

        self.main_window = main_window
        self.image_view = None
        if main_window is not None:
            self.image_view = main_window.image_view
        self.setWindowTitle("XYZ tracking debugger")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, on=True)

        self.setStyleSheet(open("expert_pi/gui/style/style.qss").read())
        self.setLayout(QtWidgets.QGridLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        # self.setGeometry(2560, 285, 900, 400)

        self.elements = {}

        self.axes = ["x", "y", "z"]
        axes_colors = {
            "x": (255, 0, 0),
            "y": (0, 0, 255),
            "z": (0, 255, 0),
        }
        self.plot_types = ["offset", "E", "I", "D", "target", "regulated", "stage", "electronic", "test_error"]

        self.global_lines = ["base", "motion_type"]

        self.plot_control = QtWidgets.QWidget()
        self.plot_control.setLayout(QtWidgets.QHBoxLayout())
        self.plot_control.layout().setContentsMargins(0, 0, 0, 0)
        self.plot_control.layout().setSpacing(20)

        self.global_selector = buttons.ToolbarMultiButton(
            {x: (x, None) for x in self.global_lines},
            multi_select=True,
            no_select=True,
            default_option=copy.deepcopy(self.global_lines),
        )
        self.plot_control.layout().addWidget(self.global_selector)
        self.global_selector.clicked.connect(self.update_plot_visibility)
        self.global_selector.setStyleSheet("padding:5px")

        self.axes_selector = buttons.ToolbarMultiButton(
            {x: (x, None) for x in self.axes},
            multi_select=True,
            no_select=True,
            default_option=copy.deepcopy(self.axes),
        )
        self.plot_control.layout().addWidget(self.axes_selector)
        self.axes_selector.clicked.connect(self.update_plot_visibility)
        self.axes_selector.setStyleSheet("padding:5px")

        self.type_selector = buttons.ToolbarMultiButton(
            {x: (x, None) for x in self.plot_types}, multi_select=True, no_select=True, default_option=["offset"]
        )
        self.plot_control.layout().addWidget(self.type_selector)
        self.type_selector.setStyleSheet("padding:5px")
        self.type_selector.clicked.connect(self.update_plot_visibility)
        self.plot_control.layout().addStretch()

        self.layout().addWidget(self.plot_control)

        self.plot_widget = pg.PlotWidget()

        self.layout().addWidget(self.plot_widget)

        self.plot_widget.setLabel("left", "xyz offset (um)")

        self.plot_lines = {C: LivePlotItem(pen=pg.mkPen(color=(255, 255, 255))) for C in self.global_lines}

        for name in self.axes:
            for type in self.plot_types:
                self.plot_lines[name + "_" + type] = LivePlotItem(pen=pg.mkPen(color=axes_colors[name]))

        for item in self.plot_lines.values():
            self.plot_widget.addItem(item)

        self.settings = QtWidgets.QWidget()
        self.settings.setLayout(QtWidgets.QHBoxLayout())
        self.settings.layout().setContentsMargins(0, 0, 0, 0)
        self.settings.layout().setSpacing(0)
        self.layout().addWidget(self.settings)

        # available_methods = [shift_measurements.Method.Templates,  # faster then Patches
        #                      shift_measurements.Method.PatchesPass1,  # quit slow
        #                      shift_measurements.Method.PatchesPass2,  # quit slow
        #                      shift_measurements.Method.CrossCorr,  # fast but inaccurate
        #                      shift_measurements.Method.ECCMaximization,  # relatively speed and good accuracy
        #                      shift_measurements.Method.Orb,  # can be useful for repeated patterns
        #                      shift_measurements.Method.OpticalFlow,  # not tested

        available_methods = [
            shift_measurements.Method.TEMRegistration,  # A.I. image registration method
            shift_measurements.Method.TEMRegistrationMedium,
            shift_measurements.Method.TEMRegistrationTiny,
        ]

        self.model = combo_box.ToolbarComboBox()
        for method in available_methods:
            self.model.addItem(method.name)
        self.model.setCurrentIndex(len(available_methods) - 2)
        self.model.currentIndexChanged.connect(self.update_settings)
        self.settings.layout().addWidget(self.model)

        self.settings.layout().addWidget(QtWidgets.QLabel("reference N:"))
        self.reference_N = combo_box.ToolbarComboBox()
        for N in [64, 128, 256, 512, 1024]:
            self.reference_N.addItem(str(N))
        self.reference_N.setCurrentIndex(int(np.log2(xyz_tracking.settings.reference_N) - 6))
        self.reference_N.currentIndexChanged.connect(self.update_settings)
        self.settings.layout().addWidget(self.reference_N)
        self.reference_N.currentIndexChanged.connect(lambda x: (self.update_max_fov(), self.rescan()))

        self.settings.layout().addWidget(QtWidgets.QLabel("tracking N:"))
        self.tracking_N = combo_box.ToolbarComboBox()
        for N in [64, 128, 256, 512, 1024]:
            self.tracking_N.addItem(str(N))
        self.tracking_N.setCurrentIndex(int(np.log2(xyz_tracking.settings.tracking_N) - 6))
        self.tracking_N.currentIndexChanged.connect(self.update_settings)
        self.settings.layout().addWidget(self.tracking_N)

        self.settings.layout().addWidget(QtWidgets.QLabel("pixel_time:"))
        self.pixel_time = spin_box.SpinBoxWithUnits(
            xyz_tracking.settings.pixel_time, [40, 10_000], 100, "ns", decimals=0, send_immediately=True
        )
        self.pixel_time = spin_box.SpinBoxWithUnits(100, [40, 10_000], 100, "ns", decimals=0, send_immediately=True)
        self.pixel_time.set_signal.connect(self.update_settings)
        self.settings.layout().addWidget(self.pixel_time)

        self.info = QtWidgets.QWidget()

        self.info.setLayout(QtWidgets.QHBoxLayout())
        self.info.layout().setContentsMargins(0, 0, 0, 0)
        self.info.layout().setSpacing(0)
        self.layout().addWidget(self.info)

        self.node_viewers = {
            "acquisition": NodeViewer(
                "ACQUISITION", 1, self.acquisition_input_processor, 5, self.acquisition_output_processor
            ),
            "registration": NodeViewer(
                "REGISTRATION", 0, self.registration_input_processor, 4, self.registration_output_processor
            ),
            "regulation": NodeViewer(
                "REGULATION", 0, self.regulation_input_processor, 4, self.regulation_output_processor
            ),
            "stage_shift": NodeViewer("STAGE", 1, self.stage_input_processor, 5, self.stage_output_processor),
        }

        node_names = np.array([d.name for d in xyz_tracking.manager.nodes])
        for name, item in self.node_viewers.items():
            self.info.layout().addWidget(item)
            i = np.argwhere(node_names == name).flat[0]
            if item not in xyz_tracking.manager.nodes[i].debug_trackers:
                xyz_tracking.manager.nodes[i].debug_trackers.append(item)

        self.update_plot_visibility()

    def reset_time(self, start_time):
        for n, i in self.plot_lines.items():
            i.t0 = start_time

    def update_plot_visibility(self):
        for item in self.global_lines:
            if item in self.global_selector.selected:
                self.plot_lines[item].show()
            else:
                self.plot_lines[item].hide()

        for name in self.axes:
            for type in self.plot_types:
                full_name = name + "_" + type
                if full_name in self.plot_lines:
                    if name in self.axes_selector.selected and type in self.type_selector.selected:
                        self.plot_lines[full_name].show()
                    else:
                        self.plot_lines[full_name].hide()

    def set_max_history_time(self, max_history_time):
        for item in self.plot_lines.values():
            item.max_history_time = max_history_time

    def acquisition_input_processor(self, input: objects.NodeImage):
        return []

    def acquisition_output_processor(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        # if xyz_tracking.settings.enable_regulation:
        #     self.main_window.stem_adjustments.shift_x.setReadSentSignal.emit(output.shift_electronic[0] * 1e3)
        #     self.main_window.stem_adjustments.shift_y.setReadSentSignal.emit(output.shift_electronic[1] * 1e3)

        #     self.main_window.stem_adjustments.stage_x.setReadSentSignal.emit(output.stage_xy[0])
        #     self.main_window.stem_adjustments.stage_y.setReadSentSignal.emit(output.stage_xy[1])
        #     self.main_window.stem_adjustments.stage_z.setReadSentSignal.emit(output.stage_z)

        self.plot_lines["x_stage"].addData(statistics.timestamp, output.stage_xy[0])
        self.plot_lines["y_stage"].addData(statistics.timestamp, output.stage_xy[1])
        self.plot_lines["z_stage"].addData(statistics.timestamp, output.stage_z)

        self.plot_lines["x_electronic"].addData(statistics.timestamp, output.shift_electronic[0])
        self.plot_lines["y_electronic"].addData(statistics.timestamp, output.shift_electronic[1])

        return [
            f"{output.image.shape[0]:3} x {output.image.shape[1]:3} ({output.scale})x",
            f"fov {output.fov:6.3f}",
            f"x: {output.stage_xy[0]:6.3f} | {output.shift_electronic[0]:6.3f} um",
            f"y: {output.stage_xy[1]:6.3f} | {output.shift_electronic[1]:6.3f} um",
            f"z: {output.stage_z:6.3f}  um",
        ]

    def registration_input_processor(self, input: objects.NodeImage):
        return []

    def registration_output_processor(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        self.plot_lines["base"].addData(statistics.timestamp, 0)
        self.plot_lines["x_offset"].addData(statistics.timestamp, output.offset_xy[0])
        self.plot_lines["y_offset"].addData(statistics.timestamp, output.offset_xy[1])

        if output.offset_z is not None:
            self.plot_lines["z_offset"].addData(statistics.timestamp, output.offset_z)

            z_text = f"{output.offset_z:6.3f} um ({output.confidence_z * 100:4.1f}% , {output.offset_z_xy_shift[0]:6.3f}|{output.offset_z_xy_shift[1]:6.3f} um)  "
        else:
            z_text = ""

        return [
            f"{output.offset_xy[0]:6.3f} um ({output.confidence_xy[0] * 100:4.1f}% {output.std_x:4.3f})",
            f"{output.offset_xy[1]:6.3f} um ({output.confidence_xy[1] * 100:4.1f}% {output.std_y:4.3f})",
            z_text,
        ]

    def regulation_input_processor(self, input: objects.NodeImage):
        return []

    def regulation_output_processor(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        results = ["E:", "I:", "D", "output:"]
        self.plot_lines["x_target"].addData(statistics.timestamp, output.target_xy_offset[0])
        self.plot_lines["y_target"].addData(statistics.timestamp, output.target_xy_offset[1])
        self.plot_lines["z_target"].addData(statistics.timestamp, output.target_z_offset)

        for i in range(len(output.errors)):
            self.plot_lines[f"{self.axes[i]}_E"].addData(statistics.timestamp, output.errors[i])
            self.plot_lines[f"{self.axes[i]}_I"].addData(statistics.timestamp, output.errors_integral[i])
            self.plot_lines[f"{self.axes[i]}_D"].addData(statistics.timestamp, output.errors_difference[i])
            self.plot_lines[f"{self.axes[i]}_regulated"].addData(statistics.timestamp, output.regulated_xyz_shift[i])

            results[0] += f"{output.errors[i]:6.2f}"
            results[1] += f"{output.errors_integral[i]:6.2f}"
            results[2] += f"{output.errors_difference[i]:6.2f}"
            results[3] += f"{output.regulated_xyz_shift[i]:6.2f}"

        return results

    def stage_input_processor(self, input: objects.NodeImage):
        return [f"total error: {input.total_error:6.3f} um"]

    def stage_output_processor(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        self.plot_lines[f"motion_type"].addData(statistics.timestamp, output.motion_type.value)
        for i in range(3):
            self.plot_lines[f"{self.axes[i]}_test_error"].addData(statistics.timestamp, output.errors[i])

        return [
            f"{output.motion_type.name}",
            f"x correction: {output.xyz_correction[0]:6.3f} um",
            f"y correction:{output.xyz_correction[1]:6.3f} um",
            f"z correction:{output.xyz_correction[2]:6.3f} um",
            f"next fov:{output.next_fov:6.3f} um",
        ]

    def stage_output_processor(self, statistics, output):
        return []

    def update_settings(self, *args):
        xyz_tracking.settings.model = self.model.currentText()
        xyz_tracking.settings.reference_N = int(self.reference_N.currentText())
        xyz_tracking.settings.tracking_N = int(self.tracking_N.currentText())
        xyz_tracking.settings.pixel_time = self.pixel_time.value()
