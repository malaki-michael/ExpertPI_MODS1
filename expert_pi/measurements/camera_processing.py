import numpy as np
import matplotlib.pyplot as plt


def show_camera_map(header, data, vmin=None, vmax=None, cut=None):
    shape_4d = (
        header["scanDimensions"][1],
        header["scanDimensions"][2],
        data["cameraData"].shape[1],
        data["cameraData"].shape[2],
    )
    camera_data = data["cameraData"].reshape(shape_4d)

    block = []
    for i in range(camera_data.shape[0]):
        block.append([])
        for j in range(camera_data.shape[1]):
            if cut is None:
                block[i].append(camera_data[i, j, :, :].T[::-1, ::-1])
            else:
                block[i].append(camera_data[i, j, cut[0][0] : cut[0][1], cut[1][0] : cut[1][1]].T[::-1, ::-1])
    plt.figure()
    plt.imshow(np.block(block), vmin=vmin, vmax=vmax)
