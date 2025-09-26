import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_camera_map(header, data, vmin=None, vmax=None, cut=None, binning=1):
    shape4D = (header['scanDimensions'][1], header['scanDimensions'][2], data['cameraData'].shape[1], data['cameraData'].shape[2])
    camera_data = data['cameraData'].reshape(shape4D)

    block = []
    for i in range(camera_data.shape[0]):
        block.append([])
        for j in range(camera_data.shape[1]):
            if cut is None:
                block[i].append(camera_data[i, j, :, :].T[::-1, ::-1])
            else:
                block[i].append(camera_data[i, j, cut[0][0]:cut[0][1], cut[1][0]:cut[1][1]].T[::-1, ::-1])
            if binning > 1:
                K = block[i][-1].shape[0]//binning  # must be square and dividable by binning
                block[i][-1] = np.mean(np.mean(block[i][-1].reshape(K, binning, K, binning), axis=-1), axis=1)
    plt.figure()
    plt.imshow(np.block(block), vmin=vmin, vmax=vmax, cmap="gray")
    return np.block(block)


stack = show_camera_map(header, data, binning=4, cut=[[256 - 64, 256 + 64]]*2, vmax=1000)
stack[stack > 1000] = 1000

save_folder = "c:/temp/"
im = Image.fromarray((stack/np.max(stack)*255).astype("uint8"))
im.save(save_folder + '/4D_stem.tif')
