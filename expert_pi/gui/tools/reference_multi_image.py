import numpy as np

from expert_pi.gui.data_views import image_item
from expert_pi.gui.style import coloring
from expert_pi.gui.tools import base
from expert_pi.stream_processors import functions


class ReferenceMultiImage(base.Tool):
    def __init__(self, view):
        super().__init__(view)
        self.image_items = []
        self.aligned_image = image_item.ImageItem(np.zeros((2, 2)), 0)
        self.aligned_image.update_transforms()
        self.aligned_image.setParentItem(self)
        self.setZValue(-1)
        self.aligned_image.setZValue(0)
        self.image_alpha = 0.7
        self.ref_tint = [0.8, 0.9, 0.8]

        self.reference_images_object = None

        self.hide()
        self.view.graphics_area.addItem(self)

    def clear(self):
        while len(self.image_items) > 0:
            self.image_items[0].setParentItem(None)
            self.view.scene().removeItem(self.image_items[0])
            del self.image_items[0]
        self.image_items = []

    def set_aligned_image(self, fov, image, offset, red_factor, transform2x2, position=None):
        if len(image.shape) == 3:
            image_8b = np.zeros((image.shape[1], image.shape[2], 4), dtype="uint8")
            image_8b[:, :, :3] = coloring.get_colored_differences(image[0, :, :], image[1, :, :])
            image_8b[:, :, 3] = 255 * self.image_alpha
        else:
            image_8b = functions.calc_cast_numba(image.ravel(), self.view.normalizer.alpha, self.view.normalizer.beta)
            image_8b = image_8b.reshape(image.shape)

            zeros = np.zeros(image_8b.shape, dtype="uint8")

            image_8b = np.dstack([
                image_8b,
                image_8b * (1 - red_factor),
                image_8b * (1 - red_factor),
                zeros + 255 * self.image_alpha,
            ]).astype("uint8")

        self.aligned_image.set_image(image_8b)
        self.aligned_image.fov = fov
        self.aligned_image.shift = [offset[0], -offset[1]]
        if position is not None:
            self.aligned_image.shift[0] += position[0]
            self.aligned_image.shift[1] += position[1]

        self.update_transforms(0, transform2x2)  # TODO rotation

    def redraw(self):
        self.aligned_image.update_image()
        for item in self.image_items:
            if item.fov < self.aligned_image.fov:
                item.hide()
            else:
                item.show()
                item.update_transforms()

    def update_transforms(self, rotation, transform2x2):
        # TODO rotation
        ref_transform2x2 = np.linalg.solve(transform2x2, self.reference_images_object.scanning_to_sample_transform)
        for item in self.image_items:
            item.transform2x2 = ref_transform2x2

    def set_reference_images(self, reference_images_object, positions=None):
        self.reference_images_object = reference_images_object
        self.clear()
        for i, (fov, image) in enumerate(reference_images_object.images.items()):
            image_8b = functions.calc_cast_numba(image.ravel(), self.view.normalizer.alpha, self.view.normalizer.beta)
            image_8b = image_8b.reshape(image.shape)
            image_8b = np.dstack([
                image_8b * self.ref_tint[0],
                image_8b * self.ref_tint[1],
                image_8b * self.ref_tint[2],
            ]).astype("uint8")

            fov_real = (
                reference_images_object.fovs_total[i]
                / reference_images_object.total_pixels[i]
                * reference_images_object.rectangles[i][2]
            )

            self.image_items.append(image_item.ImageItem(image_8b, fov_real))
            if positions is not None:
                self.image_items[-1].set_shift(
                    reference_images_object.offsets[i][0], reference_images_object.offsets[i][1]
                )
            self.image_items[-1].update_transforms()
            self.image_items[-1].setZValue(-1)
            self.image_items[-1].setParentItem(self)

        # # need to be filled
        # self.shift_electronic = None
        # self.stage_xy = None
        # self.rotation = None
        # self.scanning_to_sample_transform = np.eye(2)
        # self.stage_z = None
        # self.stage_ab = None
