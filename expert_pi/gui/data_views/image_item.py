from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np


class ImageItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, image: np.ndarray, fov, shift=None):
        self.image = image
        self.format = self.extract_format(image)

        self.pix_map = QtGui.QPixmap(QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], self.format))
        super().__init__(self.pix_map)

        self.fov = fov
        self.rotation = 0
        self.shift = [0, 0] if shift is None else shift
        self.transform2x2 = np.eye(2)  # scanning to stage transform #interferes with rotation

    def set_shift(self, x, y):
        self.shift = [x, y]
        self.update_transforms()

    def set_fov(self, fov):
        self.fov = fov
        self.update_transforms()

    def set_rotation(self, value):
        self.rotation = value
        self.update_transforms()

    def set_transform2x2(self, value):
        self.transform2x2 = value
        self.update_transforms()

    def extract_format(self, image: np.ndarray):
        if image.ndim == 2:
            if image.dtype == np.uint8:
                return QtGui.QImage.Format.Format_Grayscale8
            else:
                return QtGui.QImage.Format.Format_Grayscale16
        elif image.shape[2] == 4:
            return QtGui.QImage.Format.Format_RGBA8888
        else:
            return QtGui.QImage.Format.Format_RGB888

    def set_image(self, image: np.ndarray, rectangle=None):
        format_ = self.extract_format(image)

        if (self.format != format_
            or image.shape[:2] != (self.pix_map.height(), self.pix_map.width())
                or rectangle is None):

            self.image = image
            self.format = format_
            self.pix_map = QtGui.QPixmap(QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], format_))
        else:
            r = rectangle
            sub_image = np.ascontiguousarray(image[r[0]: r[0] + r[2], r[1]: r[1] + r[3]])
            stride = sub_image.shape[1] * sub_image.strides[-1]
            if len(image.shape) == 3:
                stride *= image.shape[2]
            sub_pix_map = QtGui.QPixmap(QtGui.QImage(sub_image, sub_image.shape[1], sub_image.shape[0], stride, format_))
            rect = QtCore.QRect(r[1], r[0], r[3], r[2])
            QtGui.QPainter(self.pix_map).drawPixmap(rect, sub_pix_map)

    def update_transforms(self):
        R = np.array([[np.cos(self.rotation), np.sin(self.rotation)],
                      [-np.sin(self.rotation), np.cos(self.rotation)]])
        transform = np.dot(R, self.transform2x2)
        warp_b = np.dot(
            np.eye(2) - transform,
            self.fov / self.image.shape[0] * np.array(self.image.shape[:2][::-1]).reshape(2, 1) / 2,
        )
        warp_mat = np.vstack([np.hstack([transform, warp_b]), [0, 0, 1]])

        self.setScale(self.fov / self.image.shape[0])
        self.setPos(
            self.shift[0] - self.fov / 2 / self.image.shape[0] * self.image.shape[1], self.shift[1] - self.fov / 2
        )
        self.setTransform(QtGui.QTransform(*warp_mat.T.flatten()))

    def update_image(self):
        self.update_transforms()
        self.setPixmap(self.pix_map)
