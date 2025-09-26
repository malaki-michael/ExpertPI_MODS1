from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np


class DraggableLine(QtWidgets.QGraphicsItemGroup):
    def __init__(self, histogram, drag_function):
        super().__init__()
        self.histogram = histogram
        self.dragging = False
        self.drag_function = drag_function

        self.items = QtWidgets.QGraphicsLineItem

        self.px_select_size = 8  # px
        self.items = {
            "visible_line": QtWidgets.QGraphicsLineItem(0, 0, 0, 0),
            "selection_rect": QtWidgets.QGraphicsRectItem(0, 0, 0, 0),
        }

        self.items["selection_rect"].setBrush(QtGui.QColor(0, 0, 0, 0))
        self.items["selection_rect"].setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0), 0))

        self.items["visible_line"].setPen(QtGui.QPen(QtGui.QColor(200, 200, 255), 2))

        for name, item in self.items.items():
            item.setParentItem(self)
            item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SplitHCursor))

    def setLine(self, x0, y0, x1, y1):
        self.items["visible_line"].setLine(x0, y0, x1, y1)
        self.items["selection_rect"].setRect(x0 - self.px_select_size / 2, y0, self.px_select_size, y1 - y0)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = True

    def mouseMoveEvent(self, e: QtWidgets.QGraphicsSceneMouseEvent):
        rect = self.histogram.sceneRect()
        w = rect.width()
        p = self.histogram.padding
        e.pos()
        self.drag_function((e.pos().x() - p) * 1.0 / (w - 2 * p))

    def mouseReleaseEvent(self, e):
        self.dragging = False
        # TODO leave parent


class HistogramView(QtWidgets.QGraphicsView):
    histogram_changed = QtCore.Signal(str, float, float)
    update_signal = QtCore.Signal()  # incoming signal

    def __init__(self, name, channels: tuple[str, ...]):
        super().__init__()
        self.name = name

        self.channels = [ch for ch in channels]
        self.channel = channels[0]
        self.range = {name: [0, 1] for name in channels}
        self.amplify = {name: 0.0 for name in channels}
        self.auto_range_enabled = {name: False for name in channels}

        self.max_N = 100
        scene = QtWidgets.QGraphicsScene(0, 0, self.max_N, self.max_N)
        self.setStyleSheet("background-color:black")
        self.setScene(scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        points = [[0, 0], [100, 100], [200, 000], [300, 200], [400, 0], [0, 0]]
        self.polygon = QtGui.QPolygonF([QtCore.QPointF(a[0], a[1]) for a in points])
        self.histogramItem = QtWidgets.QGraphicsPolygonItem(self.polygon)
        self.histogramItem.setBrush(QtGui.QBrush(QtGui.QColor(18, 18, 18)))
        self.histogramItem.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))

        self.min_rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 0, 0))
        self.max_rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 0, 0))
        self.min_line = DraggableLine(self, lambda x: self.set_range_manual([x, self.range[self.channel][1]]))
        self.max_line = DraggableLine(self, lambda x: self.set_range_manual([self.range[self.channel][0], x]))

        self.min_rect.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 50, 150)))
        self.max_rect.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 50, 150)))

        self.scene().addItem(self.histogramItem)
        self.scene().addItem(self.max_rect)
        self.scene().addItem(self.min_rect)
        self.scene().addItem(self.max_line)
        self.scene().addItem(self.min_line)

        self.auto_checkbox = QtWidgets.QCheckBox("auto")
        self.auto_checkbox_proxy = self.scene().addWidget(self.auto_checkbox)
        self.auto_checkbox_proxy.setPos(0, 80)
        self.auto_checkbox.setStyleSheet(
            "QCheckBox {spacing: 5px;color:#777;padding-left:2px;background-color:rgba(0,0,0,0)}"
            + "QCheckBox::indicator{width: 0.8em;height: 0.8em;}"
            + "QCheckBox::indicator:unchecked {image: url(expert_pi/gui/style/images/elements/unchecked.svg);}"
            + "QCheckBox::indicator:checked{image: url(expert_pi/gui/style/images/elements/checked.svg);}"
        )
        self.auto_checkbox.stateChanged.connect(lambda value: self.auto_checked(value))

        self.bins = 1024  # do not change
        self.data = np.zeros(self.bins)

        self.update_signal.connect(self.redraw)

        self.padding = 5

        super().hide()
        self.set_range([0, 1])

    def switch_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            # self.image_view.recalculate_histogram()
            self.redraw()

    def set_channel(self, channel):
        self.channel = channel
        alpha, beta = self.set_range(self.range[self.channel])

        self.auto_checkbox.blockSignals(True)
        self.auto_checkbox.setChecked(self.auto_range_enabled[self.channel])
        self.auto_checkbox.blockSignals(False)
        return alpha, beta, self.amplify[self.channel]

    def set_range(self, range_):
        if range_[0] < 0:
            range_[0] = 0
        if range_[1] > 1:
            range_[1] = 1
        if range_[0] > range_[1] - 1 / 2048:
            range_[0] = range_[1] - 1 / 2048

        self.range[self.channel] = range_

        r = self.range[self.channel][1] - self.range[self.channel][0]

        alpha = 255 / (2**16 - 1) / r * 2 ** self.amplify[self.channel]
        beta = -self.range[self.channel][0] / 2 ** self.amplify[self.channel] * (2**16 - 1) * alpha

        return alpha, beta

    def set_range_manual(self, range_, emit=True):
        self.auto_checkbox.blockSignals(True)
        self.auto_checkbox.setChecked(False)
        self.auto_checkbox.blockSignals(False)

        alpha, beta = self.set_range(range_)

        self.redraw()
        if emit:
            self.histogram_changed.emit(self.channel, alpha, beta)

    def auto_range(self, emit=True):
        mask = self.data > 0
        alpha, beta = self.set_range([np.argmax(mask) / len(mask), (len(mask) - np.argmax(mask[::-1])) / len(mask)])

        self.redraw()
        if emit:
            self.histogram_changed.emit(self.channel, alpha, beta)

    def auto_checked(self, value):
        self.auto_range_enabled[self.channel] = value
        self.auto_range()

    def resizeEvent(self, e):
        self.setSceneRect(self.visibleRegion().boundingRect())
        self.recalculate_polygons()
        self.redraw()

    def wheelEvent(self, event):
        delta = event.angleDelta().x() / 2880 + event.angleDelta().y() / 2880  # in degrees
        if delta > 0:
            factor = +1
        else:
            factor = -1
        self.amplify[self.channel] = min(8, max(0, self.amplify[self.channel] + factor))  # in bits
        self.set_range_manual(self.range[self.channel], emit=True)
        # self.image_view.recalculate_histogram()
        self.redraw()
        # self.image_view.update_from_raw_image()

    def recalculate_polygons(self, data=None):
        if data is not None:
            self.data = data
        rect = self.sceneRect()
        h = rect.height()
        log_data = np.log(1 + self.data)

        max_value = max(log_data)
        y_scale = 0
        if max_value != 0:
            y_scale = h / max_value
        bin_width = (rect.width() - 2 * self.padding) / self.bins

        points = [QtCore.QPointF(self.padding, h)]
        i = 0
        for value in log_data:  # TODO speed up?
            points.append(QtCore.QPointF(i * bin_width + self.padding, h - value * y_scale))
            points.append(QtCore.QPointF((i + 1) * bin_width + self.padding, h - value * y_scale))
            i += 1
        points.append(QtCore.QPointF(self.bins * bin_width + self.padding, h))

        self.polygon = QtGui.QPolygonF(points)

    def redraw(self):
        range_ = self.range[self.channel]
        rect = self.sceneRect()
        w = rect.width()
        p = self.padding
        self.min_rect.setRect(QtCore.QRectF(0, 0, p + (w - 2 * p) * range_[0], rect.height()))
        self.max_rect.setRect(QtCore.QRectF(p + (w - 2 * p) * range_[1], 0, rect.width(), rect.height()))

        self.min_line.setLine(p + (w - 2 * p) * range_[0], 0, p + (w - 2 * p) * range_[0], rect.height())
        self.max_line.setLine(p + (w - 2 * p) * range_[1], 0, p + (w - 2 * p) * range_[1], rect.height())

        self.histogramItem.setPolygon(self.polygon)
