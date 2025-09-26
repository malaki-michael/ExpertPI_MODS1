import time

import cv2
import numpy as np

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.config import NavigationConfig
from expert_pi.stream_clients import CacheClient


def acquire_tile(tile_rect, z, i, j, stage_position, config: NavigationConfig, cache_client: CacheClient):
    tile_rect[0] -= (tile_rect[2] / 2) * (config.tile_overlap - 1)
    tile_rect[1] -= (tile_rect[3] / 2) * (config.tile_overlap - 1)
    tile_rect[2] *= config.tile_overlap
    tile_rect[3] *= config.tile_overlap

    fov = 2 * np.max(
        np.abs([
            tile_rect[0] - stage_position[0],
            tile_rect[1] - stage_position[1],
            tile_rect[0] + tile_rect[2] - stage_position[0],
            tile_rect[1] + tile_rect[3] - stage_position[1],
        ])
    )
    grpc_client.scanning.set_field_width(fov * 1e-6)

    tile_size = config.max_size / 2**z
    total_size = int(np.round(fov / tile_size * config.tile_n))

    start = np.round(
        np.array([
            (stage_position[1] + fov / 2) - (tile_rect[1] + tile_rect[3]),
            tile_rect[0] - (stage_position[0] - fov / 2),
        ])
        / fov
        * total_size
    ).astype("int")

    n = config.tile_n + 2 * config.tile_overlap_add

    rectangle = [start[0], start[1], n, n]

    start = time.perf_counter()
    scan_id = scan_helper.start_rectangle_scan(config.pixel_time * 1e-6, total_size, frames=1, rectangle=rectangle)
    _header, data = cache_client.get_item(int(scan_id), n**2)

    images = {channel: data["stemData"][channel].reshape(n, n) for channel in data["stemData"]}

    return images


def simulate_tile(tile_rect, z, i, j, stage_position, config: NavigationConfig):
    tile_size = config.max_size / 2**z

    x = np.linspace(i * tile_size, (i + 1) * tile_size, num=config.tile_n)
    y = np.linspace((j) * tile_size, (j + 1) * tile_size, num=config.tile_n)
    xg, yg = np.meshgrid(x, y, indexing="ij")

    ws = [
        [1000, [1, 0], [0, 1], 1],
        [100, [1, 1], [0, 0], 0.5],
        [10, [1, -1], [0, 0], 0.5],
        [1, [1, -1], [1, 1], 0.1],
        [0.1, [1, 2], [0, 1], 0.1],
        [0.01, [1, 2], [1, 0], 0.1],
        [0.001, [1, -1], [0, 1], 0.01],
    ]

    # ws = [[4000, [1, 0], [0, 1],1]]

    index = xg * 0
    for w in ws:
        index += (
            w[3]
            * np.cos((xg * w[1][0] + yg * w[1][1]) * np.pi / w[0]) ** 2
            * np.cos((xg * w[2][0] + yg * w[2][1]) * np.pi / w[0]) ** 2
        )

    index[:3, :] = 0
    index[:, :3] = 0

    index[-3:, :] = 0
    index[:, -3:] = 0

    index = (index * 2**16 / 2.5).astype("uint16")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (512 // 2 - 100, 512 // 2)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    text = str((z, i, j))

    # Using cv2.putText() method
    # I = cv2.putText(I, text, org, font,
    # fontScale, color, thickness, cv2.LINE_AA)

    time.sleep(0.1)
    return {"BF": index}
