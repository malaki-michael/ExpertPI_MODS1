import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from expert_pi.gui.tools.periodic_table import ElementItem


class SpectrumView(pg.PlotWidget):
    update_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.name = "spectrum_view"

        self.histogram = self.plot(
            [0, 30000],
            [0],
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=(18, 18, 60),
            pen=pg.mkPen(color=(80, 80, 120)),
        )

        self.histogram.setZValue(10)
        self.lines = []
        self.elements = {}

        self.setYRange(0, 1.5)
        self.setXRange(0, 10000)
        self.setLabel("bottom", "energy", units="V")

        # x = np.linspace(0, 30000, num=100)
        # y = (np.sin(x/1000)**2)*np.sin(x/8000)**2
        #
        # self.set_histogram(x, y[:-1])

        self.data = [[], []]

        self.update_signal.connect(self.update_histogram)

        self.lock = threading.Lock()

    def update_data(self, energy, histogram):
        # function to be used from different thread
        with self.lock:
            self.data = [energy, histogram]

        self.update_signal.emit()

    def update_histogram(self):
        with self.lock:
            energy, histogram = self.data
        max_hist = np.max(histogram)
        if max_hist > 0:
            histogram = histogram / max_hist
        self.histogram.setData(energy, histogram)

    def update_lines(self, element_item: ElementItem):
        element_name = element_item.name

        if element_name in self.elements:
            if not element_item.hover and not element_item.active:
                lines = self.elements.pop(element_name)
                for line_item in lines:
                    self.removeItem(line_item)
            elif element_item.color != "":
                for line in self.elements[element_name]:
                    if hasattr(line, "setColor"):
                        line.setColor(element_item.color)
                    else:
                        line.setPen(pg.mkPen(color=element_item.color))

        elif element_item.hover or (element_item.selected and element_item.active):
            lines = []
            if element_item.color == "":
                color = "#ffffff"
            else:
                color = element_item.color
            for line_name, energy in element_item.lines.items():
                line_item = self.plot([energy, energy], [0, 1], pen=pg.mkPen(color=color))
                text = f"{element_name} {line_name}: {int(energy)}"
                text_item = pg.TextItem(text=text, anchor=(0, 0.5), angle=45, color=color)
                self.addItem(text_item)
                text_item.setPos(energy, 1)
                lines.append(text_item)
                lines.append(line_item)
            self.elements[element_name] = lines
