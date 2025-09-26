from typing import Any
import requests

import cv2
import numpy as np

from serving_manager.utils.decorators import exception_handler


def _encode_image(image, image_encoder: str = ".tif"):
    _, encoded_image = cv2.imencode(image_encoder, image)
    return encoded_image.tobytes()


@exception_handler(Exception)
def infer_image_litserve(image: np.ndarray, port: str = "8000", host: str = "127.0.0.1", metadata: dict[str, Any] = None, timeout: int = 10, image_encoder: str = ".tif"):
    """
    Infer image using litserver. Please note api is not stable yet.
    Args:
        image (np.ndarray): image to infer
        port (str): port to use
        host (str): host to use
        metadata (dict): metadata to send to litserver
        timeout (int): timeout for litserver
        image_encoder (str): image encoder to use
    Returns:
        dict: inference result
    """
    if metadata is None:
        metadata = {}
    endpoint = f"http://{host}:{port}/predict"
    image_bytes = _encode_image(image, image_encoder)
    response = requests.post(endpoint, data={"image_bytes": image_bytes.hex()} | metadata, timeout=timeout)
    return response.json()


def infer_multiple_images_litserve(images: list[np.ndarray], port: str = "8000", host: str = "127.0.0.1", metadata: dict[str, Any] = None, timeout: int = 10, image_encoder: str = ".tif"):
    """
    Infer multiple images using litserver. Please note api is not stable yet.
    """
    outputs = []
    for image in images:
        outputs.append(infer_image_litserve(image, port=port, host=host, metadata=metadata, timeout=timeout, image_encoder=image_encoder))
    return outputs


if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    print(infer_image_litserve(image))