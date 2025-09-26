import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from sklearn.linear_model import LinearRegression


class LivePlotItem(pg.PlotCurveItem):
    def __init__(self, max_history_time=30, pen=None):
        super().__init__(pen=pen)
        self.xs = []
        self.ys = []

    def addData(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        # self.setData(self.xs, self.ys)

    def clear(self):
        self.xs = []
        self.ys = []


def seconds_to_time(seconds):
    minutes = seconds // 60
    seconds2 = seconds % 60
    return f"{int(minutes)}:{int(seconds2):02d}"


class XYZPlot(pg.PlotWidget):
    name: str

    update_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()

        self.setLabel("bottom", "alpha (deg)")
        self.setLabel("left", "xyz error (um)")

        self.plot_lines = {
            "base": LivePlotItem(pen=pg.mkPen(color=(255, 255, 255))),
            "x": LivePlotItem(pen=pg.mkPen(color=(255, 0, 0))),
            "y": LivePlotItem(pen=pg.mkPen(color=(0, 0, 255))),
            "z": LivePlotItem(pen=pg.mkPen(color=(0, 255, 0))),
            "xp": LivePlotItem(pen=pg.mkPen(color=(255, 100, 0), style=QtCore.Qt.PenStyle.DashLine)),
            "yp": LivePlotItem(pen=pg.mkPen(color=(100, 0, 255), style=QtCore.Qt.PenStyle.DashLine)),
            "zp": LivePlotItem(pen=pg.mkPen(color=(0, 255, 100), style=QtCore.Qt.PenStyle.DashLine)),
        }

        self.correction_model = None

        for item in self.plot_lines.values():
            self.addItem(item)

        self.update_signal.connect(self.redraw)

    def clear_data(self, keep_model=False):
        for name, item in self.plot_lines.items():
            if keep_model and name in ["xp", "yp", "zp"]:
                continue
            item.clear()
            item.setData(item.xs, item.ys)

    def create_bases(self, alpha, max_N=6):
        bases = []
        for i in range(1, max_N):
            bases.append(alpha**i)

        # bases.append(np.sign(np.ediff1d(alpha, to_begin=0)))

        return np.array(bases)

    def fit_data(self, custom_bases=None):
        alpha = np.array(self.plot_lines["y"].xs)
        x = np.array(self.plot_lines["x"].ys)
        y = np.array(self.plot_lines["y"].ys)
        z = np.array(self.plot_lines["z"].ys)

        if custom_bases is None:
            X = self.create_bases(alpha)
        else:
            X = custom_bases(alpha)

        self.correction_model = LinearRegression().fit(X.T, np.array([x, y, z]).T)

        alpha_plot = np.linspace(min(alpha), max(alpha), num=1024)

        if custom_bases is None:
            Xp = self.create_bases(alpha_plot)
        else:
            Xp = custom_bases(alpha_plot)

        xyzp = self.correction_model.predict(Xp.T)
        for i, name in enumerate(["xp", "yp", "zp"]):
            self.plot_lines[name].xs = alpha_plot
            self.plot_lines[name].ys = xyzp[:, i]
            self.plot_lines[name].setData(alpha_plot / np.pi * 180, xyzp[:, i])

    def redraw(self):
        for name, line in self.plot_lines.items():
            line.setData(np.array(line.xs) / np.pi * 180, line.ys)

    def precalculate_correction_model(self, alphas):
        """alpha in degs"""
        alphas_rad = np.array(alphas) / 180 * np.pi
        corrections = np.zeros((len(alphas), 3))
        if self.correction_model is not None:
            corrections = self.correction_model.predict(self.create_bases(alphas_rad).T)
        return corrections
