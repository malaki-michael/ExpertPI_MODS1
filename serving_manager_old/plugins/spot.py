import glob
import io
import os
import tempfile

import cv2
import h5py
import numpy as np
import requests


def detect_spot_image(host, port, image, timeout):
    """
    Detects spots in an image using the spot detection plugin.

    Args:
        host (str): The host of the serving manager.
        port (int): The port of the serving manager.
        image (np.ndarray): The image to detect spots in.
        timeout (int): The timeout for the request.

    Returns:
        np.ndarray: The detected spots.
    """
    _, buffer = cv2.imencode('.tif', image)
    image_bytes = np.array(buffer).tobytes()
    response = requests.post(
        f"http://{host}:{port}/spot/detect_image",
        files={"image": ("image.tif", image_bytes, "image/tif")},
        timeout=timeout
    )
    return np.array(response.json()["result"])[0, ...]


def detect_spot_batched(host, port, images, use_tempfile=False):
    """
    Detects spots in an image using the spot detection plugin.

    Args:
        host (str): The host of the serving manager.
        port (int): The port of the serving manager.
        images (np.ndarray | list[np.ndarray]): The images to detect spots in. Images should be in shape [B, H, W],
            where B is the batch size, H is the height of the image, and W is the width of the image. Other possible
            input is list of [H, W] images.

    Returns:
        np.ndarray: The detected spots.
    """
    if isinstance(images, list):
        images = np.stack(images)
    if use_tempfile:
        with tempfile.TemporaryDirectory() as tmpdir:
            with h5py.File(os.path.join(tmpdir, "hdf.h5"), "w") as hdf:
                hdf.create_dataset("images", data=images)
            with open(os.path.join(tmpdir, "hdf.h5"), "rb") as f:
                response = requests.post(
                    f"http://{host}:{port}/spot/detect_images_hdf",
                    files={"hdf_file": f},
                    timeout=1000
                )
    else:
        io_bytes = io.BytesIO()
        with h5py.File(io_bytes, "w") as hdf:
            hdf.create_dataset("images", data=images)
        io_bytes.seek(0)
        response = requests.post(
            f"http://{host}:{port}/spot/detect_images_hdf",
            files={"hdf_file": io_bytes},
            timeout=1000
        )
    return np.array(response.json()["result"])



if __name__ == "__main__":
    import random
    from time import time
    from matplotlib import pyplot as plt
    
    PATH = "/home/brani/tescan/data/strain/TSMC_2/"
    images = glob.glob(os.path.join(
        PATH, 
        "*.tif"
    ))[:100]
    random.shuffle(images)
    start = time()
    images = np.stack([cv2.imread(image, cv2.IMREAD_UNCHANGED).astype(np.float32) for image in images])
    end = time()
    print(f"Ellapsed time: {end - start}")
    
    start = time()
    spots = detect_spot_batched("localhost", 5000, images, False)
    end = time()
    print(f"Ellapsed time: {end - start}")
    print(spots.shape)
    for image, spot in zip(images, spots):
        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        for single_spot in spot:
            if single_spot[-1] > 0.5:
                cv2.circle(image, (int(single_spot[1]), int(single_spot[0])), 3, (0, 1, 0), -1)
            else:
                cv2.circle(image, (int(single_spot[1]), int(single_spot[0])), 3, (1, 0, 0), -1)
        plt.imshow(image)
        plt.show()
