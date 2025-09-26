import time

import numpy as np
from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base


class StatusInfo(base.Tool):
    def __init__(self, view, is_diffraction=False):
        super().__init__(view)
        self.is_diffraction = is_diffraction

        self.outline = QtWidgets.QGraphicsSimpleTextItem()

        self.fill = QtWidgets.QGraphicsSimpleTextItem()

        self.xy = [0, 0]
        self.xy_real = [0, 0]  # um
        self.fps = 0
        self.fps_average_factor = 0.8
        self.frame_id = 0
        self.scan_id = 0
        self.variance = 0
        self.wave_length = None  # need to be setup on microscope synchronization

        self.update_view_tm = time.perf_counter()

        self.outline.setFont(QtGui.QFont("Courier", 9))
        self.fill.setFont(QtGui.QFont("Courier", 9))
        self.fill.setBrush(QtGui.QColor(255, 255, 255))
        self.outline.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 5))
        self.outline.setBrush(QtGui.QColor(0, 0, 0))

        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(5)
        self.outline.setGraphicsEffect(blur)

        self.hide()
        self.view.scene().addItem(self.outline)
        self.view.scene().addItem(self.fill)

    def show(self):
        super().show()
        self.fill.show()
        self.outline.show()
        self.is_active = True
        self.update()

    def hide(self):
        self.is_active = False
        super().hide()
        self.fill.hide()
        self.outline.hide()

    def update_frame_index(self, frame_index, scan_id):
        self.scan_id = scan_id
        self.frame_id = frame_index
        now = time.perf_counter()

        dt = now - self.update_view_tm
        if dt > 0:
            fps = 1 / dt
            self.fps = self.fps_average_factor * self.fps + (1 - self.fps_average_factor) * fps
        self.update_view_tm = now

    def generate_text(self):
        if self.is_diffraction:
            distance = np.sqrt(self.xy_real[0] ** 2 + self.xy_real[1] ** 2)
            if distance > 0:
                real_distance = self.wave_length / (distance * 1e-3) * 1e9  # to nm
            else:
                real_distance = 0
            text = f"({distance:6.2f} mrad {real_distance:6.3f} nm "

        else:
            if np.any(np.abs(self.xy_real) < 1):
                real = [f"{self.xy_real[0] * 1000:6.2f}", f"{self.xy_real[1] * 1000:6.2f}"]
                unit = "nm"
            else:
                real = [f"{self.xy_real[0]:6.2f}", f"{self.xy_real[1]:6.2f}"]
                unit = "um"

            text = f"({real[0]},{real[1]}) {unit} "
        text += f"FPS:{self.fps:.1f} frame:{self.frame_id:6}"
        self.fill.setText(text)
        self.outline.setText(text)

    def setPos(self, x, y):
        self.fill.setPos(x, y)
        self.outline.setPos(x, y)

    def update(self, *args, **kwargs):
        self.setPos(10, self.view.size().height() - 20)
        self.generate_text()

    def view_mouse_moved(self, e, focused_item=None):
        self.xy = [e.pos().x(), e.pos().y()]
        self.xy_real = self.view.map_to_area(self.xy)
        self.generate_text()
