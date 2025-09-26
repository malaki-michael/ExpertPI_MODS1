from threading import Lock

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class ColoredSlider(QtWidgets.QSlider):
    update_signal = QtCore.Signal()

    def __init__(self, on_resize=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ranges_lock = Lock()
        self.ranges_colors = {(0, 10): "#555555"}
        # example:
        # self.ranges_colors = {(0, 4): "#009900",
        #                       (4, 5): "#cc0000",
        #                       (5, 6): "#cc8800",
        #                       (6, 10): "#555555"}

        self.groove_height = 8  # px
        self.ticks_color = "#c0c0c0"
        self.indicator_position = 4.3  # None if not used
        self.indicator_color = "#6666dd"

        self.update_signal.connect(self.update)

        self.on_resize = on_resize

        self.tick_positions = {}

    def set_ranges_colors(self, ranges_colors):
        with self.ranges_lock:
            self.ranges_colors = ranges_colors

    def paintEvent(self, ev):
        style = QtWidgets.QApplication.style()

        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)

        handle_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt, QtWidgets.QStyle.SubControl.SC_SliderHandle
        )
        painter = QtWidgets.QStylePainter(self)

        indicator_position = self.indicator_position  # move to variable due to possible changes from different thread
        if indicator_position is not None:
            x = np.round(
                indicator_position / self.maximum() * (self.width() - handle_rect.width()) + handle_rect.width() / 2
            )
            painter.setPen(QtGui.QColor(self.indicator_color))
            painter.drawLine(x, self.rect().top(), x, self.rect().bottom() - self.rect().top())

        opt.subControls = QtWidgets.QStyle.SubControl.SC_SliderHandle
        # print(opt)
        style.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_Slider, opt, painter, self)

        size = 8
        dw = self.width() - handle_rect.width()
        if dw != 0:
            factor = self.maximum() / dw
        else:
            factor = self.maximum()
        handle_start = handle_rect.left()
        handle_end = handle_rect.left() + handle_rect.width()

        with self.ranges_lock:
            for crange, color in self.ranges_colors.items():
                if crange[0] == 0:
                    start = 0
                else:
                    start = int(crange[0] / factor + handle_rect.width() / 2)
                if crange[1] == self.maximum():
                    end = self.width()
                else:
                    end = int(crange[1] / factor + handle_rect.width() / 2)

                painter.setPen(QtGui.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(color)))
                if handle_start <= start and handle_end >= end:
                    pass
                elif handle_start > start and handle_end < end:
                    rect = QtCore.QRect(
                        start, int(self.height() / 3 - self.groove_height / 2), handle_start - start, self.groove_height
                    )
                    painter.drawRect(rect)
                    rect = QtCore.QRect(
                        handle_end,
                        int(self.height() / 3 - self.groove_height / 2),
                        end - handle_end,
                        self.groove_height,
                    )
                    painter.drawRect(rect)
                elif handle_start < end <= handle_end:
                    rect = QtCore.QRect(
                        start, int(self.height() / 3 - self.groove_height / 2), handle_start - start, self.groove_height
                    )
                    painter.drawRect(rect)
                elif handle_end > start >= handle_start:
                    rect = QtCore.QRect(
                        handle_end,
                        int(self.height() / 3 - self.groove_height / 2),
                        end - handle_end,
                        self.groove_height,
                    )
                    painter.drawRect(rect)
                else:
                    rect = QtCore.QRect(
                        start, int(self.height() / 3 - self.groove_height / 2), end - start, self.groove_height
                    )
                    painter.drawRect(rect)
        h = 4
        y = self.rect().bottom() - h
        tick_positions = {}
        for i in range(self.minimum(), self.maximum() + 1):
            x = np.round(i / self.maximum() * (self.width() - handle_rect.width()) + handle_rect.width() / 2)

            painter.setPen(QtGui.QColor(self.ticks_color))
            painter.drawLine(x, y, x, y + h)
            tick_positions[i] = x

        self.tick_positions = tick_positions

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.on_resize is not None:
            self.on_resize(ev)


class AnotatedColoredSlider(QtWidgets.QWidget):
    name: str

    def __init__(self):
        super().__init__()

        self.slider = ColoredSlider(None, QtCore.Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.min_value_label = QtWidgets.QLabel("0")
        self.max_value_label = QtWidgets.QLabel("10")

        self.range = None
        self.set_range(0, 10, 11)

        self.slider.setMinimum(0)
        self.slider.setMaximum(10)

        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.layout().addWidget(self.min_value_label)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.max_value_label)

        self.handle_text = QtWidgets.QLabel("1")
        self.handle_text.setParent(self)
        # self.handle_text.setStyleSheet("background-color:red")
        # self.onValueChanged(self.slider.value())

        self.slider.valueChanged.connect(self.onSliderValueChanged)
        self.slider.on_resize = lambda x: self.onSliderValueChanged(self.slider.value())

    def onSliderValueChanged(self, value, format="+5.3f"):
        geometry = self.slider.geometry()

        numeric_value = self.range[0] + value / (self.range[2] - 1) * (self.range[1] - self.range[0])
        self.handle_text.setText(f"{numeric_value:{format}}")

        br = self.handle_text.fontMetrics().boundingRect(self.handle_text.text())
        try:
            s = self.slider.tick_positions[value]
        except:
            s = 0

        self.handle_text.setGeometry(
            int(geometry.x() + s - br.width() / 2 - 5), geometry.y() - br.height(), br.width(), br.height()
        )

    def set_range(self, min, max, n, format="+5.2f"):
        self.range = [min, max, n]
        self.min_value_label.setText(f"{min:{format}}")
        self.max_value_label.setText(f"{max:{format}}")
        self.slider.setMaximum(n - 1)
