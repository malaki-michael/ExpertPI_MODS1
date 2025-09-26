import os
import threading
from ast import literal_eval as make_tuple

import cv2
import h5py
import numpy as np

from expert_pi.config import Config


class NavigationCache:
    def __init__(self, configs: Config):
        self.configs = configs
        self.f = None
        self.layers = None
        self.lock = threading.RLock()
        self.tile_status_lock = threading.RLock()

    def initialize(self, filename=None, mode="r+"):
        print("init", filename, mode)
        if self.f is not None:
            self.f.close()
        if filename is None:
            if os.path.exists(self.configs.data.navigation_cache):
                self.f = h5py.File(self.configs.data.navigation_cache, mode)
            else:
                self.f = h5py.File(self.configs.data.navigation_cache, "w")

        else:
            if not os.path.exists(filename):
                self.f = h5py.File(filename, "w")

            self.f = h5py.File(filename, mode)
        if "layers" not in self.f.keys():
            self.layers = self.f.create_group("layers")
        else:
            self.layers = self.f["layers"]

    def write_tile_with_overlaps(self, z, i, j, images):
        # TODO add registration ?
        add = self.configs.navigation.tile_overlap_add
        n = self.configs.navigation.tile_n

        rects = {
            (-1, -1): ([0, add, 0, add], [n - add, add, n - add, add]),
            (0, -1): ([add, n, 0, add], [0, n, n - add, add]),
            (1, -1): ([add + n, add, 0, add], [0, add, n - add, add]),
            (-1, 0): ([0, add, add, n], [n - add, add, 0, n]),
            (0, 0): ([add, n, add, n], [0, n, 0, n]),
            (1, 0): ([add + n, add, add, n], [0, add, 0, n]),
            (-1, 1): ([0, add, add + n, add], [n - add, add, 0, add]),
            (0, 1): ([add, n, add + n, add], [0, n, 0, add]),
            (1, 1): ([add + n, add, add + n, add], [0, add, 0, add]),
        }

        result = None
        for key, rect2 in rects.items():
            rect = rect2[0]
            rect_out = rect2[1]

            if rect[1] * rect[3] == 0:
                continue

            new_tile_images = {c: images[c][rect[0] : rect[0] + rect[1], rect[2] : rect[2] + rect[3]] for c in images}

            raw_images = None
            if key == (0, 0):
                rect_out = None
                result = new_tile_images
                raw_images = images
            else:
                continue
                # TODO properly write combined interpolated+partly acquired tiles

            if self.get_tile_status(z, i + key[0], j + key[1]) < 1 or key == (0, 0):
                self.write_tile(z, i + key[0], j + key[1], new_tile_images, rect=rect_out, raw_images=raw_images)
        return result

    def write_tile(self, z, i, j, images, rect=None, raw_images=None):
        """Images dict channel:image rect [x,w,y,h]."""
        str_z = str(z)
        str_ij = str((i, j))

        with self.lock:
            if str_z not in self.layers:
                self.layers.create_group(str_z)

            if str_ij not in self.layers[str_z]:
                tile = self.layers[str_z].create_group(str_ij)
            else:
                tile = self.layers[str_z][str_ij]

            if "mask" not in tile:
                tile.create_dataset(
                    "mask",
                    data=np.zeros((self.configs.navigation.tile_n, self.configs.navigation.tile_n), dtype="bool"),
                    dtype="bool",
                    compression="lzf",
                )
            if "status" not in tile:
                tile.create_dataset("status", data=0, shape=(), dtype="int8")

            if rect is None:
                tile["mask"][:] = True
                tile["status"][()] = 1  # fully filled
            else:
                tile["mask"][rect[0] : rect[0] + rect[1], rect[2] : rect[2] + rect[3]] = True

            if raw_images is not None:
                if "raw" in tile:
                    raw_images_group = tile["raw"]
                else:
                    raw_images_group = tile.create_group("raw")

            for channel, image in images.items():
                if channel not in tile:
                    tile.create_dataset(
                        channel,
                        shape=(self.configs.navigation.tile_n, self.configs.navigation.tile_n),
                        dtype="uint16",
                        compression="lzf",
                    )  # TODO support different dtypes?

                if raw_images is not None:
                    if channel in raw_images_group:
                        raw_images_group[channel][:] = raw_images[channel]
                    else:
                        raw_images_group.create_dataset(channel, data=raw_images[channel], compression="lzf")

                if rect is None:
                    tile[channel][:] = image
                else:
                    tile[channel][rect[0] : rect[0] + rect[1], rect[2] : rect[2] + rect[3]] = image

    def get_tile(self, z, i, j, channel="BF"):
        str_z = str(z)
        str_ij = str((i, j))
        with self.lock:
            if self.get_tile_status(z, i, j) >= 0:
                image = self.layers[str_z][str_ij][channel][:]
            else:
                image = np.zeros((self.configs.navigation.tile_n, self.configs.navigation.tile_n), dtype="uint16")
                if self.configs.navigation.show_interpolated:
                    zu = z * 1
                    iu = i * 1
                    ju = j * 1
                    while zu > 0:
                        zu -= 1
                        iu = iu // 2
                        ju = ju // 2
                        if self.get_tile_status(zu, iu, ju) >= 0:
                            tile = self.get_tile(zu, iu, ju, channel=channel)
                            B = 2 ** (z - zu)
                            size = self.configs.navigation.tile_n // B
                            if size < 1:
                                break
                            rect = [size * (i % B), size, size * (j % B), size]

                            image = (
                                tile[rect[0] : rect[0] + rect[1], rect[2] : rect[2] + rect[3]]
                                .repeat(B, axis=0)
                                .repeat(B, axis=1)
                            )
                            break

                # if self.get_tile_status(z,i,j)==0:
                #     mask=self.layers[str_z][str_ij]["mask"][:]
                #     image[mask] = self.layers[str_z][str_ij][channel][mask]

        return image

    def get_image(self, xy0, xy2, width_px, height_px, channel="BF"):
        xy0, xy2, z, ij0, ij2 = self.get_tiles_from_rect(xy0, xy2, width_px)

        image_px_size = (xy2[0] - xy0[0]) / height_px

        tile_px_size = self.configs.navigation.max_size / 2**z / self.configs.navigation.tile_n

        scaled_tile_size = int(self.configs.navigation.tile_n * tile_px_size / image_px_size)

        img = np.zeros((width_px, height_px), dtype="uint16")

        for i in range(ij0[0], ij2[0]):
            for j in range(ij0[1], ij2[1]):
                tile = self.get_tile(z, i, j, channel=channel)
                tile_rect_xy = self.tile_rect(z, i, j)

                origin = [
                    int(-(tile_rect_xy[1] + tile_rect_xy[3] - xy0[1]) / image_px_size),
                    int((tile_rect_xy[0] - xy0[0]) / image_px_size),
                ]

                tile_resized = cv2.resize(tile, (scaled_tile_size, scaled_tile_size), interpolation=cv2.INTER_NEAREST)

                input_rect = [0, 0, scaled_tile_size, scaled_tile_size]
                output_rect = [origin[0], origin[1], scaled_tile_size, scaled_tile_size]

                for k in range(2):
                    if origin[k] < 0:
                        output_rect[k] = 0
                        input_rect[k] -= origin[k]
                        input_rect[k + 2] += origin[k]
                        output_rect[k + 2] += origin[k]

                    diff = output_rect[k] + output_rect[k + 2] - img.shape[k]
                    if diff > 0:
                        output_rect[k + 2] -= diff
                        input_rect[k + 2] -= diff

                if input_rect[2] * input_rect[3] > 0:
                    O = output_rect
                    I = input_rect
                    img[O[0] : O[0] + O[2], O[1] : O[1] + O[3]] = tile_resized[I[0] : I[0] + I[2], I[1] : I[1] + I[3]]

        return img

    def get_tiles_from_rect(self, xy0, xy2, pixels_width):
        # closest z:
        if pixels_width > 0:
            pixel_size = (xy2[0] - xy0[0]) / pixels_width
            z = np.round(
                np.log(self.configs.navigation.max_size / (pixel_size * self.configs.navigation.tile_n)) / np.log(2)
            ).astype("int")
            z = np.clip(z, 0, self.configs.navigation.max_zoom)
        else:
            z = self.configs.navigation.max_zoom

        _tile_size = self.configs.navigation.max_size / 2**z

        ij0 = self.pos_to_tile_index(z, xy0)
        ij2 = self.pos_to_tile_index(z, xy2)

        i0 = np.clip(np.round(ij0[0]).astype(int), 0, 2**z)
        i2 = np.clip(np.round(ij2[0]).astype(int) + 1, 0, 2**z)

        j0 = np.clip(np.round(ij0[1]).astype(int), 0, 2**z)
        j2 = np.clip(np.round(ij2[1]).astype(int) + 1, 0, 2**z)

        return xy0, xy2, z, (i0, j0), (i2, j2)

    def get_tile_status(self, z, i, j):
        # 1=="done"
        # 0=="partly"
        # -1=="missing"

        str_z = str(z)
        str_ij = str((i, j))
        with self.tile_status_lock:
            if str_z in self.layers:
                if str_ij in self.layers[str_z]:
                    return self.layers[str_z][str_ij]["status"][()]
        return -1

    def get_tile_mask(self, z, i, j):
        str_z = str(z)
        str_ij = str((i, j))
        with self.lock:
            if str_z in self.layers:
                if str_ij in self.layers[str_z]:
                    return self.layers[str_z][str_ij]["mask"][:]

    def fill_upper_layers(self, z, i, j, images, z_limit=0, force=False):
        rect = [0, self.configs.navigation.tile_n, 0, self.configs.navigation.tile_n]
        z0 = z * 1
        i0 = i * 1
        j0 = j * 1
        while z > z_limit:
            z -= 1
            z_add = z0 - z
            b = 2**z_add
            size = self.configs.navigation.tile_n // b
            if size < 1:
                break
            rect = [size * (i0 % b), size, size * (j0 % b), size]

            i = i0 // b
            j = j0 // b

            if not force and self.get_tile_status(z, i, j) == 1:
                break

            for channel, image in images.items():
                images[channel] = (
                    image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2)
                    .mean(axis=1)
                    .mean(axis=-1)
                    .astype("uint16")
                )

            self.write_tile(z, i, j, images, rect=rect)

    def reset(self, z, ij0, ij2):
        with self.lock:
            for zz in range(z, self.configs.navigation.max_zoom + 1):
                str_z = str(zz)
                if str_z in self.layers:
                    for key in self.layers[str_z]:  # it might not be best to iterate over all keys...
                        ij = make_tuple(key)
                        if ij0[0] <= ij[0] < ij2[0] and ij0[1] <= ij[1] < ij2[1]:
                            del self.layers[str_z][key]
                            print("del", str_z, key)

                ij0 = [ij0[0] * 2, ij0[1] * 2]
                ij2 = [ij2[0] * 2, ij2[1] * 2]

    def tile_rect(self, z, i, j):
        tile_size = self.configs.navigation.max_size / 2**z
        return np.array([
            j * tile_size - self.configs.navigation.max_size / 2,
            (2**z - 1 - i) * tile_size - self.configs.navigation.max_size / 2,
            tile_size,
            tile_size,
        ])

    def pos_to_tile_index(self, z, xy):
        tile_size = self.configs.navigation.max_size / 2**z
        xyt = (np.array(xy) + self.configs.navigation.max_size / 2) / tile_size
        i = 2**z - xyt[1] - 0.5
        j = xyt[0] - 0.5
        return [i, j]

    def tile_index_to_pos(self, z, ij):
        tile_size = self.configs.navigation.max_size / 2**z
        xyt = np.array([ij[1] + 0.5, 2**z - (ij[0] + 0.5)])
        xy = xyt * tile_size - self.configs.navigation.max_size / 2
        return xy
