from PySide6 import QtGui, QtWidgets

from expert_pi.gui.tools import base


def round_to_125(value):
    e = f"{value:1e}"
    if value < 0:
        digit = int(e[1])
    else:
        digit = int(e[0])
    factor = int(f"{value:1e}".split("e")[1])
    if digit >= 5:
        return 5, factor
    elif digit >= 2:
        return 2, factor
    else:
        return 1, factor


class SizeBar(base.Tool):
    def __init__(self, view, diffraction_units=False, color=QtGui.QColor(255, 255, 255)):
        super().__init__(view)
        self.diffraction_units = diffraction_units

        self.w = 3

        self.lines = [
            QtWidgets.QGraphicsLineItem(-100, 0, 100, 0),
            QtWidgets.QGraphicsLineItem(-100, -self.w, -100, self.w),
            QtWidgets.QGraphicsLineItem(100, -self.w, 100, self.w),
        ]

        for line in self.lines:
            line.setParentItem(self)
            line.setPen(QtGui.QPen(color, 3))

        self.text = QtWidgets.QGraphicsSimpleTextItem()
        self.text.setFont(QtGui.QFont("Courier", 9))
        self.text.setBrush(color)
        self.text.setText("1um")
        self.text.setParentItem(self)
        self.hide()
        self.view.scene().addItem(self)
        self.update()

    def update(self, *args, **kwargs):
        rect = self.view.sceneRect()

        rect.width()

        start_px = [rect.width() * 2 / 3, rect.height() - 20]
        end_px = [rect.width() - 20, rect.height() - 20]

        start_real = self.view.map_to_area(start_px)
        end_real = self.view.map_to_area(end_px)

        size = end_real[0] - start_real[0]
        digit, factor = round_to_125(size)

        size_adjusted = digit * 10**factor

        start_px_ad = [
            (start_px[0] + end_px[0]) / 2 - (end_px[0] - start_px[0]) / 2 * size_adjusted / size,
            start_px[1],
        ]
        end_px_ad = [(start_px[0] + end_px[0]) / 2 + (end_px[0] - start_px[0]) / 2 * size_adjusted / size, end_px[1]]

        if self.diffraction_units:
            text = f"{int(size_adjusted + 0.5)} mrad"
        elif factor < -3:
            text = f"{float(size_adjusted) * 1000.0:f} nm"
        elif factor < 0:
            text = f"{int(size_adjusted * 1000 + 0.5)} nm"
        elif factor < 3:
            text = f"{int(size_adjusted + 0.5)} um"
        else:
            text = f"{int(size_adjusted / 1000 + 0.5)} mm"

        self.text.setText(text)

        self.text.setPos(
            (start_px[0] + end_px[0]) / 2 - self.text.boundingRect().width() / 2, (start_px[1] + end_px[1]) / 2 + 5
        )

        self.lines[0].setLine(start_px_ad[0], start_px_ad[1], end_px_ad[0], end_px_ad[1])
        self.lines[1].setLine(start_px_ad[0], -self.w + start_px_ad[1], start_px_ad[0], self.w + end_px_ad[1])
        self.lines[2].setLine(end_px_ad[0], -self.w + start_px_ad[1], end_px_ad[0], self.w + end_px_ad[1])
