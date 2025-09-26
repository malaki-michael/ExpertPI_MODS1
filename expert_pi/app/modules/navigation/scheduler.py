import matplotlib.pyplot as plt
import numpy as np

from expert_pi.config import NavigationConfig

temp = None


def get_next_pos(tile_mask, ij_pos, fov, stage_fact=[4, 1.5], center_fact=[1.5, 1.5]):
    """Stage_pos,fov in tile units."""
    global temp

    n = tile_mask.shape[0]
    m = tile_mask.shape[1]

    ii = np.arange(n)
    jj = np.arange(m)
    i_ind, j_ind = np.meshgrid(ii, jj, indexing="ij")

    k_centers = np.zeros((n, m))
    k_corners = np.zeros((n - 1, m - 1))

    # centers
    d_center = int(fov / 2 - 0.5)
    d_corner = int(fov / 2)

    # centers of tiles

    for i in range(n):
        for j in range(m):
            k_centers[i, j] = np.sum(tile_mask[i - d_center : i + d_center + 1, j - d_center : j + d_center + 1])

    functional_centers = k_centers / (
        1e-5
        + k_centers
        + np.sqrt(center_fact[0] * (i_ind + 0.5 - n / 2) ** 2 + center_fact[1] * (j_ind + 0.5 - m / 2) ** 2)
        + np.sqrt(stage_fact[0] * (i_ind - ij_pos[0]) ** 2 + stage_fact[1] * (j_ind - ij_pos[1]) ** 2)
    )

    # corners of tiles

    for i in range(n - 1):
        for j in range(m - 1):
            k_corners[i, j] = np.sum(
                tile_mask[i - d_corner + 1 : i + d_corner + 1, j - d_corner + 1 : j + d_corner + 1]
            )

    functional_corners = k_corners / (
        1e-5
        + k_corners
        + np.sqrt(
            center_fact[0] * (i_ind[:-1, :-1] + 1 - n / 2) ** 2 + center_fact[1] * (j_ind[:-1, :-1] + 1 - m / 2) ** 2
        )
        + np.sqrt(
            stage_fact[0] * (i_ind[:-1, :-1] + 0.5 - ij_pos[0]) ** 2
            + stage_fact[1] * (j_ind[:-1, :-1] + 0.5 - ij_pos[1]) ** 2
        )
    )

    temp = [ij_pos, functional_centers, functional_corners, k_centers, i_ind, j_ind, n, m, center_fact, stage_fact]

    centers_max_ij = np.unravel_index(np.argmax(functional_centers), functional_centers.shape)
    use_centers = True
    if functional_corners.shape[0] * functional_corners.shape[1] > 0:
        corners_max_ij = np.unravel_index(np.argmax(functional_corners), functional_corners.shape)
        if functional_centers[centers_max_ij] < functional_corners[corners_max_ij]:
            use_centers = False

    if use_centers:
        i_min, j_min = centers_max_ij[0], centers_max_ij[1]

        rect = [i_min - d_center, i_min + d_center + 1, j_min - d_center, j_min + d_center + 1]
        pos = [i_min, j_min]

    else:
        i_min, j_min = corners_max_ij[0], corners_max_ij[1]

        rect = [i_min - d_corner + 1, i_min + d_corner + 1, j_min - d_corner + 1, j_min + d_corner + 1]
        pos = [i_min + 0.5, j_min + 0.5]

    rect = [np.clip(rect[0], 0, n), np.clip(rect[1], 0, n), np.clip(rect[2], 0, m), np.clip(rect[3], 0, m)]

    return rect, pos


def test_path(tile_mask, ij_pos, fov, stage_fact=[4, 1.5], center_fact=[1.5, 1.5], max_steps=100):
    ij_positions = [ij_pos]
    while np.sum(tile_mask) > 0 and len(ij_positions) < max_steps:
        rect, pos = get_next_pos(tile_mask, ij_pos, fov, stage_fact=stage_fact, center_fact=center_fact)

        ij_positions.append(pos)
        ij_pos = pos

        tile_mask_old = tile_mask * 1
        print(rect, pos)
        tile_mask[rect[0] : rect[1], rect[2] : rect[3]] = False
    _, ax = plt.subplots(1)
    ax.imshow(tile_mask + tile_mask_old)
    ax.plot([s[1] - 0.5 for s in ij_positions], [s[0] - 0.5 for s in ij_positions], "-o")
    plt.show()


class TileJob:
    def __init__(self, cache, xy0, xy2, z, ij0, ij2, max_fov, config: NavigationConfig):
        self.cache = cache
        self.config = config
        self.xy0, self.xy2 = xy0, xy2  # um coordinates but still i,j directions
        self.z = z
        self.ij0, self.ij2 = ij0, ij2
        self.N = self.ij2[0] - self.ij0[0]
        self.M = self.ij2[1] - self.ij0[1]
        self.max_fov = max_fov  # um
        self.tile_mask = self.get_tile_mask(self.z, self.ij0, self.ij2)

        tile_size = self.config.max_size / 2**self.z
        fov = min(self.config.max_supertile, self.max_fov / tile_size)
        self.subsample_mode = False
        if fov < 1:
            # we need to measure different layer for this:
            self.subsample_mode = True
            self.z_add = int(np.ceil(-np.log(fov) / np.log(2)))
            self.tile_mask_sub_sample = self.get_tile_mask(
                self.z + self.z_add, np.array(self.ij0) * 2**self.z_add, np.array(self.ij2) * 2**self.z_add
            )

    def get_tile_mask(self, z, ij0, ij2):
        tile_mask = np.zeros((ij2[0] - ij0[0], ij2[1] - ij0[1]), dtype="int8")
        for i in range(ij0[0], ij2[0]):  # assuming small number of tiles - double for cycle
            for j in range(ij0[1], ij2[1]):
                tile_mask[i - ij0[0], j - ij0[1]] = self.cache.get_tile_status(z, i, j)
        return tile_mask

    def get_done_tiles(self, parts=True):
        if parts:
            indices = np.argwhere(self.tile_mask >= 0)  # include part tilt
        else:
            indices = np.argwhere(self.tile_mask == 1)
        tile_ids = []
        for tid in indices:
            tile_ids.append((self.z, tid[0] + self.ij0[0], tid[1] + self.ij0[1]))
        return tile_ids

    def get_all_tiles(self):
        indices = np.argwhere(self.tile_mask >= -1)  # include part tilt
        tile_ids = []
        for tid in indices:
            tile_ids.append((self.z, tid[0] + self.ij0[0], tid[1] + self.ij0[1]))
        return tile_ids

    def get_next(self, current_xy_position):
        if self.subsample_mode:
            tile_mask = self.tile_mask_sub_sample
            z = self.z + self.z_add
            ij0 = (self.ij0[0] * 2**self.z_add, self.ij0[1] * 2**self.z_add)
        else:
            tile_mask = self.tile_mask
            z = self.z
            ij0 = self.ij0

        if np.sum(tile_mask < 1) == 0:
            return None

        ij_pos = self.cache.pos_to_tile_index(z, current_xy_position)

        tile_size = self.config.max_size / 2**z
        fov = min(self.config.max_supertile, self.max_fov / tile_size)

        to_acquire_mask = tile_mask < 1

        rect, ij_pos_new = get_next_pos(
            to_acquire_mask,
            np.array([ij_pos[0] - ij0[0], ij_pos[1] - ij0[1]]),
            fov,
            stage_fact=self.config.stage_fact,
            center_fact=self.config.center_fact,
        )

        new_stage_position = self.cache.tile_index_to_pos(z, np.array([ij_pos_new[0] + ij0[0], ij_pos_new[1] + ij0[1]]))

        tile_ids = []
        distances = []
        for i in range(rect[0], rect[1]):
            for j in range(rect[2], rect[3]):
                if tile_mask[i, j] < 1:
                    tile_ids.append((z, i + ij0[0], j + ij0[1]))
                    distances.append((i - ij_pos_new[0]) ** 2 + (j - ij_pos_new[1]) ** 2)
                    tile_mask[i, j] = 1

        if len(tile_ids) > 2:
            zipped = zip(distances, tile_ids)
            zipped = list(zipped)

            res = sorted(zipped, key=lambda x: x[0])
            distances, tile_ids = zip(*res)
            tile_ids = list(tile_ids)

        return new_stage_position, self.z, tile_ids, np.sum(to_acquire_mask)
