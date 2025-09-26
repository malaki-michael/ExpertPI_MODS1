from typing import Any, Callable, Dict, List

import numpy as np

from serving_manager.management.torchserve_base_manager import ConfigProperties
from serving_manager.management.torchserve_grpc_manager import TorchserveGrpcManager
from serving_manager.management.rest_infer import process_image


class InferenceClient:

    def __init__(self, model_name: str, host: str, port: str, image_encoder: str = "jpg") -> None:
        """InferenceClient for GRPC torchserve inference

        Args:
            model_name (str): Model name
            host (str): Hostname IP
            port (str): Port to GRPC inference
            image_encoder (str, optional): Image encoder jpg | png. Defaults to "jpg".
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.image_encoder = image_encoder
        self.manager = TorchserveGrpcManager(inference_port=port, host=host, cache_stub=True, image_encoder=image_encoder)

    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Call inference for the given model

        Args:
            image (np.ndarray): Input image

        Returns:
            Dict[str, Any]: Output of the model, output is a dict with values of Any type according to model.
        """
        return self.manager.infer(self.model_name, image, **kwargs)


def infer_single_image(image: np.ndarray, model_name: str, host: str, port: str, image_encoder: str = "jpg") -> Dict[str, Any]:
    """Infer single image

    Args:
        image (np.ndarray): Single image to infer
        model_name (str): Model name
        host (str): Hostname IP | localhost
        port (str): Port with GRPC inference
        image_encoder (str, optional): Image encoder jpg | png. Defaults to "jpg".
    Returns:
        Dict[str, Any]: Model output
    """
    return infer_multiple_images([image, ], model_name, host, port, image_encoder=image_encoder)[0]


def infer_multiple_images(images: List[np.ndarray], model_name: str, host: str, port: str, image_encoder: str = "jpg") -> List[Dict[str, Any]]:
    """Infer multiple images in a sequence

    Args:
        images (List[np.ndarray]): Input images
        model_name (str): Model name
        host (str): Hostname IP | localhost
        port (str): Port with GRPC inference
        image_encoder (str, optional): Image encoder jpg | png. Defaults to "jpg".

    Returns:
        List[Dict[str, Any]]: Model outputs
    """
    client = InferenceClient(model_name, host, port, image_encoder=image_encoder)
    return [client(image) for image in images]


def infer_on_mar_file(
        image: np.ndarray | List[np.ndarray],
        mar_file_path: str,
        port: str = "7443",
        preprocessing_fn: Callable = lambda image: image,
        image_encoder: str = "jpg",
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
    """ Runs model on a single .mar file.

    NOTE: this function may be slower to run, please use context manager for better performance.
        First and second image may be significantly slower to run.

    Args:
        image (np.ndarray | List[np.ndarray]): Image or list of images to run inference on.
        mar_file_path (str): Path to .mar file.
        port (str, optional): Port, where the model should run. Defaults to "7443".
        preprocessing_fn (image, optional): How image should be preprocessed.
            Defaults to lambda image:image.
        image_encoder (str, optional): Image encoder jpg | png. Defaults to "jpg".

    Returns:
        List of model outputs.
    """
    props = ConfigProperties(
        grpc_inference_port=port,
    )

    manager = TorchserveGrpcManager(
        config_properties=props,
        inference_port=port,
        model_path=mar_file_path,
        stop_if_running=True,
        host="localhost",
        image_encoder=image_encoder
    )
    manager.run()

    if isinstance(image, list):
        output = [manager.infer("TEMRegistration", preprocessing_fn(img)) for img in image]

    elif isinstance(image, np.ndarray):
        output = manager.infer("TEMRegistration", preprocessing_fn(image))
    manager.stop()
    return output


def health_check(host: str, port: str) -> Dict[str, str]:
    """Checks whether the server is running

    Args:
        host (str): Hostname IP | localhost
        port (str): Port with GRPC inference

    Returns:
        Dict[str, str]: health status {'status': 'Healthy'} in case of success
    """
    return TorchserveGrpcManager(inference_port=port, host=host).health_check()


def infer_rest_single_image(image: np.ndarray, model_name: str, host: str, port: str) -> Dict[str, Any]:
    """Infer image via REST API

    Args:
        image (np.ndarray): Input image
        model_name (str): Model name
        host (str): Hostname IP | localhost
        port (str): Port with GRPC inference

    Returns:
        Dict[str, Any]: Model output
    """
    return process_image(image, model_name=model_name, host=host, port=port)
