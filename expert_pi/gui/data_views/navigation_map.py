import time

import cv2
import numpy as np
from PySide6 import QtWidgets

from expert_pi.config import get_config
from expert_pi.gui.data_views import image_item


class Tile(image_item.ImageItem):
    def __init__(self, z, i, j, image, mosaic_group):
        self.zid = (z, i, j)
        self.config = get_config().navigation
        tile_size = self.config.max_size / 2**z
        self.mosaic_group = mosaic_group
        self.raw_image = image
        super().__init__(
            self.process_image(self.raw_image),
            tile_size,
            shift=[tile_size * (j + 0.5) - self.config.max_size / 2, tile_size * (i + 0.5) - self.config.max_size / 2],
        )  # y axis is in pixels!!!
        self.setZValue(-self.config.max_zoom + z - 1)

    def process_image(self, image):
        image2 = np.clip(image / 256, 0, 255).astype("uint8")
        if self.mosaic_group.anotate:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (self.config.tile_n // 2 - 100, self.config.tile_n // 2)
            font_scale = 1
            color = (255, 0, 0)
            thickness = 2
            text = str(self.zid)
            image2 = cv2.putText(image2, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            image2[:2, :] = 0
            image2[:, :2] = 0
            image2[-2:, :] = 0
            image2[:, -2:] = 0

        return image2

    def set_image(self, image):
        self.raw_image = image
        super().set_image(self.process_image(self.raw_image))

    # def drag(self, x, y):
    #     self.mosaic_group.drag(x, y)

    # def set_start_drag_position(self, x, y):
    #     self.mosaic_group.set_start_drag_position(x, y)


class NavigationMap(QtWidgets.QGraphicsItemGroup):
    def __init__(self, view):
        self.view = view
        super().__init__()
        self.tiles = {}
        self.view.graphics_area.addItem(self)
        self.anotate = False
        self.drag_start = [0, 0]

    def remove_tiles(self, tile_ids):
        for tile_id in tile_ids:
            self.tiles[tile_id].setParentItem(None)
            self.view.scene().removeItem(self.tiles[tile_id])
            del self.tiles[tile_id]

    def update_tiles(self, tile_ids, tile_images, clean_range):
        """Tile images needs to be 8bit."""
        start = time.perf_counter()
        for ind in range(len(tile_ids)):
            if tile_ids[ind] in self.tiles:
                self.tiles[tile_ids[ind]].set_image(tile_images[ind])
                self.tiles[tile_ids[ind]].update_image()
            else:
                z, i, j = tile_ids[ind]
                self.tiles[tile_ids[ind]] = Tile(z, i, j, tile_images[ind], self)
                self.tiles[tile_ids[ind]].setParentItem(self)

            self.tiles[tile_ids[ind]].update_transforms()  # might be to slow to do separately
        print("updated", time.perf_counter() - start)
        if clean_range is not None:
            self.clean_tiles(*clean_range)
        print("cleaned", time.perf_counter() - start)

    def refresh(self):
        for t in self.tiles.values():
            t.update_image()

    def clean_tiles(self, z, ij0, ij2):
        to_remove = []
        for tid in self.tiles.keys():
            if tid[0] > z:
                to_remove.append(tid)
            else:
                dz = z - tid[0]
                if tid[1] < ij0[0] // 2**dz or tid[1] >= ij2[0] // 2**dz:
                    to_remove.append(tid)
                elif tid[2] < ij0[1] // 2**dz or tid[2] >= ij2[1] // 2**dz:
                    to_remove.append(tid)
        print("removing tiles", z, ij0, ij2, to_remove)
        self.remove_tiles(to_remove)

    # def drag(self, x, y):
    #     self.setPos(x - self.drag_start[0], y - self.drag_start[1])

    # def set_start_drag_position(self, x, y):
    #     self.drag_start = [x - self.pos().x(), y - self.pos().y()]
    #     # self.on_drag_callback(x, y)
