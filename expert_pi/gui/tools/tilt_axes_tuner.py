import numpy as np
import pyqtgraph as pg
import scipy.signal
from PySide6 import QtCore, QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box


class LivePlotItem(pg.PlotCurveItem):
    def __init__(self, max_history_time=30, pen=None, name=None):
        super().__init__(pen=pen, name=name)
        self.xs = []
        self.ys = []

    def addData(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        # self.setData(self.xs, self.ys)

    def clear(self):
        self.xs = []
        self.ys = []


class TiltAxisTuner(QtWidgets.QWidget):
    update_plots_signal = QtCore.Signal()  # incoming

    def __init__(self, main_window=None):
        super().__init__()

        self.main_window = main_window
        self.image_view = None
        self.setWindowTitle("Tilt axes tuner")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, on=True)

        self.setStyleSheet(open("expert_pi/gui/style/style.qss").read())
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.resize(1000, 600)
        # self.setGeometry(2560, 285, 900, 400)

        self.plots = QtWidgets.QWidget()
        self.plots.setLayout(QtWidgets.QHBoxLayout())
        self.plots.layout().setContentsMargins(0, 0, 0, 0)
        self.plots.layout().setSpacing(0)

        self.controls = QtWidgets.QWidget()
        self.controls.setFixedHeight(20)
        self.controls.setLayout(QtWidgets.QHBoxLayout())
        self.controls.layout().setContentsMargins(0, 0, 0, 0)
        self.controls.layout().setSpacing(0)

        self.layout().addWidget(self.plots)
        self.layout().addWidget(self.controls)

        self.angle_plot = pg.PlotWidget()
        self.angle_plot.setLabel("bottom", "alpha (deg)")
        self.angle_plot.setLabel("left", "xyz error (um)")

        self.angle_plot_lines = {
            "base": LivePlotItem(pen=pg.mkPen(color=(255, 255, 255))),
            "x": LivePlotItem(pen=pg.mkPen(color=(255, 0, 0)), name="x"),
            "y": LivePlotItem(pen=pg.mkPen(color=(0, 0, 255)), name="y"),
            "z": LivePlotItem(pen=pg.mkPen(color=(0, 255, 0)), name="z"),
            "xp": LivePlotItem(pen=pg.mkPen(color=(255, 100, 0), style=QtCore.Qt.PenStyle.DashLine)),
            "yp": LivePlotItem(pen=pg.mkPen(color=(100, 0, 255), style=QtCore.Qt.PenStyle.DashLine)),
            "zp": LivePlotItem(pen=pg.mkPen(color=(0, 255, 100), style=QtCore.Qt.PenStyle.DashLine)),
        }
        for key, item in self.angle_plot_lines.items():
            self.angle_plot.addItem(item)

        self.legend = pg.LegendItem((80, 60), offset=(70, 20))
        self.legend.setParentItem(self.angle_plot.graphicsItem())

        self.legend.addItem(self.angle_plot_lines["x"], "x")
        self.legend.addItem(self.angle_plot_lines["y"], "y")
        self.legend.addItem(self.angle_plot_lines["z"], "z")

        self.plane_plot = pg.PlotWidget()

        self.plane_plot.setLabel("bottom", "y (um)")
        self.plane_plot.setLabel("left", "z (um)")

        self.plane_plot_lines = {
            "item": LivePlotItem(pen=pg.mkPen(color=(255, 255, 255))),
        }

        for key, item in self.plane_plot_lines.items():
            self.plane_plot.addItem(item)

        self.plots.layout().addWidget(self.angle_plot)
        self.plots.layout().addWidget(self.plane_plot)

        self.angle_type = combo_box.ToolbarComboBox()
        self.angle_type.addItem("alpha")
        self.angle_type.addItem("beta")
        self.angle_type.setCurrentIndex(0)

        self.min_angle = spin_box.SpinBoxWithUnits(-10, [-90, 90], 1, "deg", decimals=2, send_immediately=True)
        self.max_angle = spin_box.SpinBoxWithUnits(10, [-90, 90], 1, "deg", decimals=2, send_immediately=True)
        self.speed = spin_box.SpinBoxWithUnits(1, [0, 45], 0.5, "deg/s", decimals=2, send_immediately=True)

        self.start_button = buttons.ToolbarPushButton("start", selectable=True)
        self.clear_button = buttons.ToolbarPushButton("clear")

        self.filter_type = combo_box.ToolbarComboBox()
        self.filter_type.addItem("median")
        self.filter_type.addItem("mean")
        self.filter_type.setCurrentIndex(0)

        self.filter_value = spin_box.SpinBoxWithUnits(0, [0, 31], 1, "x", decimals=0, send_immediately=True)
        self.filter_value.setToolTip("filter (valuex2+1)")

        self.controls.layout().addWidget(self.angle_type)
        self.controls.layout().addWidget(self.min_angle)
        self.controls.layout().addWidget(self.max_angle)
        self.controls.layout().addWidget(self.speed)
        self.controls.layout().addWidget(self.start_button)
        self.controls.layout().addWidget(self.clear_button)
        self.controls.layout().addWidget(self.filter_type)
        self.controls.layout().addWidget(self.filter_value)

        self.update_plots_signal.connect(self.update_plots)

    def update_plots(self):
        filter_value = int(self.filter_value.value() * 2 + 1)
        for lines in [self.plane_plot_lines, self.angle_plot_lines]:
            for line in lines.values():
                if filter_value > 1 and len(line.ys) > filter_value:
                    if self.filter_type.currentText() == "median":
                        ys = scipy.signal.medfilt(line.ys, filter_value)
                    else:
                        ys = np.convolve(line.ys, np.ones(filter_value) / filter_value, mode="same")
                else:
                    ys = line.ys
                line.setData(line.xs, ys)

    def clear_data(self):
        for line in self.plane_plot_lines.values():
            line.clear()

        for line in self.angle_plot_lines.values():
            line.clear()

        self.update_plots()
