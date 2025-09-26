import pyqtgraph as pg


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
