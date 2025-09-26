import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi import grpc_client


class Stage2DRanges(QtWidgets.QGraphicsItemGroup):

    def __init__(self, view, type="xy"):
        super().__init__()
        self.view = view
        self.type = type

        self.lines = [QtWidgets.QGraphicsPathItem()]

        for line in self.lines:
            line.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 255), 0))
            line.setParentItem(self)

        self.points = np.array([])
        self.hide()
        self.view.graphics_area.addItem(self)

    def download_ranges(self):
        if self.type == "xy":
            results = grpc_client.stage.get_x_y_range()
            factor = 1e6  # to um
        elif self.type == "ab":
            results = grpc_client.stage.get_alpha_beta_range()
            factor = 1e3  # to mrad

        self.points = np.array([[r["x"]*factor for r in results],
                                [r["y"]*factor for r in results]])

    def update(self, center_stage=None, rotation=0, transform2x2=np.eye(2)):
        if self.type == "xy":
            # recalculate to scanning plane:
            xy_transformed = np.linalg.solve(transform2x2, self.points - center_stage.reshape(2, 1))
        else:
            xy_transformed = self.points

        points = []
        for p in xy_transformed.T:
            points.append(QtCore.QPointF(p[0], -p[1]))

        polygon = QtGui.QPolygonF(points)
        path = QtGui.QPainterPath()
        path.addPolygon(polygon)
        self.lines[0].setPath(path)
        self.setRotation(-rotation/np.pi*180)
