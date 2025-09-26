import json
from typing import Callable, Dict, List

import numpy as np

from serving_manager.api import InferenceClient, infer_multiple_images, infer_single_image
from serving_manager.utils.exceptions import FutureModelException
from serving_manager.utils.preprocessing import preprocess_image, decode_base64_image, encode


def registration_model(
        image: List[np.ndarray] | np.ndarray,
        model_name: str,
        host: str,
        port: str,
        return_only_homography: bool = False,
        image_encoder: str = "jpg",
        left_masks: List[np.ndarray] | np.ndarray = None,
        right_masks: List[np.ndarray] | np.ndarray = None,
        ) -> List[Dict[str, List]] | Dict[str, List] | List:
    """Perform registration on a single image or a list of images.
    Image is concatenated src with dst along x-axis.

    Args:
        image (List[np.ndarray  |  str]): Input image collection or a single image.
        model_name (str): Name of the model for registration.
        host (str): Host IP address.
        port (str): Port, where GRPC server is running.
        return_only_homography (bool, optional): Whether to return only fine homography. Defaults to False.
        image_encoder (str, optional): Image encoder jpg | png. Defaults to "jpg".
        left_masks (List[np.ndarray  |  str], optional): Left masks for registration. Defaults to None.
        right_masks (List[np.ndarray  |  str], optional): Right masks for registration. Defaults to None.

    Returns:
        List[Dict[str, List]] | Dict[str, List] | List: Model output (homography vs full output).
    """
    if not isinstance(image, list):
        image = [image, ]

    if left_masks is not None and not isinstance(left_masks, list):
        left_masks = [left_masks, ]

    if right_masks is not None and not isinstance(right_masks, list):
        right_masks = [right_masks, ]

    inference_client = InferenceClient(model_name, host, port, image_encoder=image_encoder)
    output = []
    for idx, img in enumerate(image):
        if left_masks is None and right_masks is None:
            single_output = inference_client(
                image=preprocess_image(img)
            )
        else:
            single_output = inference_client(
                image=preprocess_image(img),
                left_mask=encode(left_masks[idx]),
                right_mask=encode(right_masks[idx])
            )
        output.append(single_output)

    if return_only_homography:
        output = [output["homography_fine"] for output in output] if \
            isinstance(output, list) else output["homography_fine"]
    return output


def ronchigram_model(
    *_, **__
):
    raise FutureModelException("Ronchigram model is not implemented yet")


def point_matcher_model(
    image: List[np.ndarray] | np.ndarray,
    points: List[List[List[float]]] | List[List[float]],
    model_name: str,
    host: str,
    port: str,
    preprocess_fn: Callable = preprocess_image,
    image_encoder: str = "jpg",
    left_masks: List[np.ndarray] | np.ndarray | None = None,
    right_masks: List[np.ndarray] | np.ndarray | None = None
):
    """Point matching model. Given an image which is a concatenation of src and dst images along x-axis,
    and a set of points belonging to src, find the corresponding points in dst. Model is an adaptation
    of the registration model.

    Args:
        image (List[np.ndarray] | np.ndarray): Image collection or a single image.
        points (List[List[List[float]]] | List[List[float]]): Points collection or a single set of points.
        model_name (str): Model name.
        host (str): Host IP address.
        port (str): Port, where GRPC server is running.
        preprocess_fn (Callable, optional): Preprocessing fn, like minmax normalization. Defaults to preprocess_image.
        image_encoder (str, optional): Image encoder can be "png" or "jpg". Defaults to "jpg".
        left_masks (List[np.ndarray] | np.ndarray, optional): Left masks. Defaults to None.
        right_masks (List[np.ndarray] | np.ndarray, optional): Right masks. Defaults to None.

    Returns:
        List[dict]: Model output.
    """
    if not isinstance(image, list):
        image = [image, ]
        points = [points, ]

    assert len(image) == len(points), "Number of images and points must be the same."
    if left_masks is not None and not isinstance(left_masks, list):
        left_masks = [left_masks, ]

    if right_masks is not None and not isinstance(right_masks, list):
        right_masks = [right_masks, ]

    inference_client = InferenceClient(model_name, host, port, image_encoder=image_encoder)
    output = []

    for idx, (img, pts) in enumerate(zip(image, points)):
        if left_masks is None and right_masks is None:
            single_output = inference_client(
                image=preprocess_fn(img),
                points=json.dumps(pts).encode("utf-8")
            )
        else:
            single_output = inference_client(
                image=preprocess_fn(img),
                points=json.dumps(pts).encode("utf-8"),
                left_mask=encode(left_masks[idx]),
                right_mask=encode(right_masks[idx]),
            )
        output.append(single_output)
    return output


def super_resolution_model(
    image: List[np.ndarray] | np.ndarray,
    model_name: str,
    host: str,
    port: str,
    preprocess_fn: Callable = preprocess_image,
    image_encoder: str = "jpg"
):
    """Perform super resolution on a single image or a list of images.

    Args:
        image (List[np.ndarray] | np.ndarray): Input image collection or a single image.
        model_name (str): Name of the model for super resolution.
        host (str): Host IP address.
        port (str): Port, where GRPC server is running.
        preprocess_fn (Callable, optional): Function to apply to the image first. Defaults to preprocess_image.
        image_encoder (str, optional): How is image encoded. Defaults to "jpg".

    Returns:
        np.ndarray | List[np.ndarray]: Model output.
    """
    if isinstance(image, list):
        output = infer_multiple_images(
            [preprocess_image(single_image) for single_image in image],
            model_name,
            host,
            port,
            image_encoder=image_encoder
        )
        output = [decode_base64_image(output["image"]) for output in output]
    else:
        output = decode_base64_image(infer_single_image(
            preprocess_fn(image),
            model_name,
            host,
            port,
            image_encoder=image_encoder)["image"])


    return output