import base64
import logging
import time
import warnings
from collections.abc import Callable
from functools import partial

import cv2
import numpy as np
import requests

from expert_pi.app import scan_helper
from expert_pi.automations.utils.minimization import BestResult, FunctionMinimizer
from expert_pi.config import InferenceConfig
from expert_pi.grpc_client import scanning
from expert_pi.grpc_client.modules.scanning import DetectorType
from expert_pi.measurements import kernels
from expert_pi.stream_clients import CacheClient

try:
    from serving_manager.api import TorchserveRestManager
except (ModuleNotFoundError, ImportError):
    logging.getLogger(__name__).warning("TorchServeManagerREST not available, using old method")
    TorchserveRestManager = None

pentochron = np.array([
    [-1 / 2, -np.sqrt(3) / 6, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
    [1 / 2, -np.sqrt(3) / 6, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
    [0, np.sqrt(3) / 3, -np.sqrt(6) / 12, -np.sqrt(10) / 20],
    [0, 0, np.sqrt(6) / 4, -np.sqrt(10) / 20],
    [0, 0, 0, np.sqrt(10) / 5],
])


def _is_inference_available(config: InferenceConfig) -> bool:
    """Check if the inference service is available and if the specified model is loaded.

    This function tries to connect to the inference service using the host and ports specified in the config.
    If the connection is successful, it checks if the specified model is loaded in the inference service.

    Parameters:
        config (InferenceConfig): The configuration object containing the host, ports, and model name.

    Returns:
        bool: True if the inference service is available and the model is loaded, False otherwise.
    """
    logger = logging.getLogger(__name__)
    if TorchserveRestManager is None:
        logger.warning("TorchServeManagerREST not available, using old method")
        return False

    manager = TorchserveRestManager(
        host=config.host,
        management_port=config.management_port,
        inference_port=config.inference_port,
    )
    try:
        models = list(map(lambda x: x["modelName"], manager.list_models()["models"]))
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        logger.warning("TorchServeManagerREST not available, using old method")
        return False

    if config.model_name not in models:
        logger.warning(f"Model {config.model_name} not found in TorchServe")
        return False
    return True


def _exit_callback_fn(set_function: Callable[[np.ndarray], None], _: np.ndarray, best_result: BestResult):
    set_function(best_result.value)
    logging.getLogger(__name__).info(f"best result: {best_result.value}")


class StopError(Exception):
    """Custom exception class for stopping the execution of a program."""

    pass


class GracefulStop:
    """This class is used to handle graceful stopping of a function execution.

    It provides a callable interface which checks if a stop condition is met and if so,
    it executes an exit function and raises a StopException.

    Attributes:
        stop_callback (Callable[..., bool]): A callback function that returns True if the stop condition is met.
        exit_fn (Callable[..., None]): A function to be executed when the stop condition is met.

    Raises:
        StopException: If the stop condition is met during the function execution.
    """

    def __init__(self, stop_callback: Callable[..., bool], exit_fn: Callable[..., None]):
        self.stop_callback = stop_callback
        self.exit_fn = exit_fn

    def __call__(self, x, best_result):
        if self.stop_callback():
            self.exit_fn(x, best_result)
            raise StopError("stopped")


def focus_pivot_points(
    cache_client: CacheClient,
    init_step_diagonal: float = 5,
    init_step_non_diagonal: float = 1,
    detector=DetectorType.BF,
    kernel="sobel",
    blur="median",
    n=64,
    rectangle=None,
    pixel_time=13890e-9,
    max_iterations=20,
    stop_callback=lambda: False,
    set_callback=lambda x: False,
):
    """This method focuses pivot points by maximizing the variance value using a simplex method.

    Parameters:
        init_step_diagonal (int): Initial step size for diagonal movement. Default is 5.
        init_step_non_diagonal (int): Initial step size for non-diagonal movement. Default is 1.
        detector (DetectorType): Detector type to be used. Default is DetectorType.BF (Bright field).
        kernel (str): Kernel for variance to be used. Default is 'sobel'.
        blur (str): Blurring method to be used. Default is 'median'.
        N (int): Size of the image. Default is 64.
        rectangle (None, optional): Rectangle area to be focused. Default is None.
        pixel_time (float): Time for pixel scanning. Default is 13890e-9 s.
        max_iterations (int): Maximum number of iterations for the focusing process. Default is 20.
        stop_callback (Callable[..., bool]): A callback function that returns True if the stop condition is met. Default
        is a function that always returns False.
        set_callback (Callable[..., bool]): A function to be executed when the stop condition is met.
        Default is a function that always returns False.

    Returns:
        The result of the post minimization process.
    """
    init_pp_correction = np.array(scanning.get_precession_height_correction()) * 1e6
    # pentachoron:
    init_simplex = pentochron * 1
    init_simplex[:, [0, 3]] *= init_step_diagonal
    init_simplex[:, [1, 2]] *= init_step_non_diagonal

    # data=[]

    w, h = n, n
    if rectangle is not None:
        w, h = rectangle[2], rectangle[3]

    def contrast_function():
        if stop_callback():
            raise StopError("stopped")
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=pixel_time, total_size=n, rectangle=rectangle, detectors=[detector], is_precession_enabled=True
        )
        _header, data = cache_client.get_item(scan_id, int(w * h))  # use int instead of int32 for json seriability

        img = data["stemData"][detector.name].reshape(w, h)
        return kernels.contrast_function(img, kernel=kernel, blur=blur)

    def set_function(value):
        scanning.set_precession_height_correction((init_pp_correction + value) * 1e-6)
        set_callback(init_pp_correction + value)

    def minimize_function(x: np.ndarray) -> float:
        set_function(x)
        contrast = contrast_function()
        # NOTE: return -1 * contrast to maximize
        return -contrast

    minimizer = FunctionMinimizer(
        minimized_fn=minimize_function,
        callback_fn=GracefulStop(
            stop_callback=stop_callback,
            exit_fn=partial(_exit_callback_fn, set_function),
        ),
    )
    result = minimizer.execute(
        initial_guess=np.zeros(4),
        number_of_iterations=int(2.5 * max_iterations),
        method="Nelder-Mead",
        options={
            "initial_simplex": init_simplex,
            "maxiter": max_iterations,
            "maxfev": int(2.5 * max_iterations),
        },
    )
    return _post_minimization(minimize_function, result, set_function)


def _encode(image: np.ndarray, encoder: str = ".tif"):
    if image.dtype == np.uint32:
        image = image.astype(np.float32)
    return base64.b64encode(cv2.imencode(encoder, image)[1].tobytes()).decode("utf-8")


def _compute_deprecession_metric(img, config: InferenceConfig, kernel: str, blur: str, is_inference_available: bool):
    if not is_inference_available:
        return -kernels.contrast_function(img, kernel=kernel, blur=blur)
    endpoint = config.endpoint
    response = requests.post(
        endpoint,
        json={
            "image": _encode(img),
            "host": config.host,
            "port": config.inference_port,
            "model_name": config.model_name,
        },
        timeout=5,
    ).json()
    # TODO: remove when finished
    # print(json.dumps(response, indent=4))
    return response[config.metric_name]


def optimize_deprecession(
    cache_client: CacheClient,
    init_step_diagonal: int = 5,
    init_step_non_diagonal: int = 1,
    kernel: str = "sobel",
    blur: str = "gaussian",
    n: int = 256,
    ij: tuple[int, int] = (128, 128),
    camera_time: float = 0.005,
    max_iterations: int = 20,
    stop_callback: Callable[..., bool] = lambda: False,
    set_callback: Callable[[np.ndarray], bool] = lambda x: False,
):
    """This method optimizes deprecession by aligning diffraction spots using a simplex method. It minimizes variance or spot area or weighted spot area.

    Parameters:
        init_step_diagonal (int): Initial step size for diagonal movement. Default is 5.
        init_step_non_diagonal (int): Initial step size for non-diagonal movement. Default is 1.
        kernel (str): Kernel for variance to be used. Default is 'sobel'.
        blur (str): Blurring method to be used. Default is 'gaussian'.
        N (int): Size of the image. Default is 256.
        ij (tuple): Tuple containing the coordinates of the center of the image. Default is (128, 128).
        camera_time (float): Time for camera scanning. Default is 0.005 s.
        max_iterations (int): Maximum number of function evaluation for the focusing process. Default is 20.
        stop_callback (Callable[..., bool]): A callback function that returns True if the stop condition is met. Default is a function that always returns False.
        set_callback (Callable[..., bool]): A function to be executed when the stop condition is met. Default is a function that always returns False.

    Returns:
        The result of the post minimization process.
    """
    config = InferenceConfig()
    use_inference = kernel == "spot_detection"
    is_inference_available = use_inference and _is_inference_available(config)
    # init_step in mrad
    init_pp_correction = np.array(scanning.get_deprecession_tilt_correction()) * 1e3

    if use_inference and not is_inference_available:
        warnings.warn("Inference method is picked, but inference server is not available!")

    init_simplex = pentochron * 1
    init_simplex[:, [0, 3]] *= init_step_diagonal
    init_simplex[:, [1, 2]] *= init_step_non_diagonal

    def contrast_function():
        start = time.perf_counter()
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=camera_time,
            total_size=n,
            rectangle=[ij[0], ij[1], 1, 1],
            detectors=[DetectorType.Camera],
            is_precession_enabled=True,
        )
        print("start", (time.perf_counter() - start) * 1000, "ms")
        _header, data = cache_client.get_item(scan_id, 1)
        print("get", (time.perf_counter() - start) * 1000, "ms")
        img = data["cameraData"][0]
        result = _compute_deprecession_metric(img, config, kernel, blur, is_inference_available)
        print("process", (time.perf_counter() - start) * 1000, "ms")
        return result

    def set_function(value):
        scanning.set_deprecession_tilt_correction((init_pp_correction + value) * 1e-3)
        set_callback(init_pp_correction + value)

    def minimize_function(x):
        set_function(x)
        contrast = contrast_function()
        return contrast

    minimizer = FunctionMinimizer(
        minimized_fn=minimize_function,
        callback_fn=GracefulStop(stop_callback=stop_callback, exit_fn=partial(_exit_callback_fn, set_function)),
    )
    result = minimizer.execute(
        initial_guess=np.zeros(4),
        number_of_iterations=int(2.5 * max_iterations),
        method="Nelder-Mead",
        options={
            "initial_simplex": init_simplex,
            "maxiter": max_iterations,
            "maxfev": int(2.5 * max_iterations),
        },
    )
    return _post_minimization(minimize_function, result, set_function)


def _post_minimization(minimize_function, result, set_function):
    if result is None:
        result = np.zeros(4)
    set_function(result)
    final_contrast = minimize_function(result)
    logging.getLogger(__name__).warning(f"Final contrast value: {final_contrast:.2f}.")
    return result
