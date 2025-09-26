import os.path

import cv2
import numpy as np
import requests

from serving_manager.management.torchserve_base_manager import (
    ConfigProperties,
    error_wrapper,
    TorchserveBaseManager
)


def _get_url(host: str, port: int, endpoint: str) -> str:
    return f"http://{host}:{port}/{endpoint}"


def _encode(np_image, extension: str = ".png"):
    return cv2.imencode(extension, np_image)[1].tostring()


class TorchserveRestManager(TorchserveBaseManager):

    def __init__(
        self,
        inference_port: str | None = None,
        management_port: str | None = None,
        host: str = "localhost",
        model_store_path: str = "model_store",
        model_path: str = "",
        stop_if_running: bool = False,
        config_properties: dict | ConfigProperties = ConfigProperties(),
        max_torchserve_waiting_time: int = 60,
        image_encoder: str = ".jpg",
        delete_temp_folder: bool = True,
        timeout: float = 30.0,
        **_
    ):
        super().__init__(
            inference_port,
            management_port,
            host,
            model_store_path,
            model_path,
            stop_if_running,
            config_properties,
            max_torchserve_waiting_time,
            image_encoder,
            delete_temp_folder,
            timeout=timeout
        )

    def create_config_properties(self, port_settings: dict | None = None):
        if port_settings is None:
            port_settings = {}
        if self.inference_port:
            self.config_properties["inference_address"] = f"http://{self.host}:{self.inference_port}"
        if self.management_port:
            self.config_properties["management_address"] = f"http://{self.host}:{self.management_port}"
        super().create_config_properties(port_settings)

    @error_wrapper(error_return_value={"status": "Unhealthy"})
    def health_check(self):
        try:
            status = requests.get(
                _get_url(self.host, self.inference_port, "ping"),
                timeout=self.timeout
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            return {"status": "Unavailable", "error": str(e)}

        if status.status_code != 200:

            return {"status": "Unhealthy", "error": status.text, "status_code": status.status_code}
        return status.json()

    @error_wrapper(error_return_value=False)
    def register(self, model_name: str, path: str = "", initial_workers: int = 0):
        mar_file = os.path.join(path, f"{model_name}.mar")
        self.logger.info(f"Registering model {model_name} from {mar_file}")
        params = {
            "url": mar_file,
            "initial_workers": initial_workers,
            "synchronous": True,
            "model_name": model_name
        }
        register_response = requests.post(
            _get_url(self.host, self.management_port, "models"),
            params=params,
            timeout=self.timeout
        )
        return register_response.status_code == 200

    @error_wrapper(error_return_value=False)
    def unregister(self, model_name: str):
        self.logger.info(f"Unregistering model {model_name}")
        unregister_response = requests.delete(
            _get_url(self.host, self.management_port, f"models/{model_name}"),
            params={"synchronous": True},
            timeout=self.timeout
        )
        return unregister_response.status_code == 200

    @error_wrapper(error_return_value={})
    def scale(self, model_name: str, min_worker: int = 1, max_worker: int | None = None, synchronous: bool = True):
        self.logger.info(f"Scaling model {model_name} to {min_worker} workers")
        params = {
            "min_worker": min_worker,
            "synchronous": synchronous,
        }
        if max_worker is not None:
            params["max_worker"] = max_worker

        scale_response = requests.put(
            _get_url(self.host, self.management_port, f"models/{model_name}"),
            params=params,
            timeout=self.timeout
        )
        return scale_response.status_code == 200

    def _infer_without_worker_check(self, model_name, image, **kwargs):
        image = _encode(image, self.image_encoder)
        if not kwargs:
            data = image
        else:
            data = {"image": image} | kwargs
        inference_response = requests.post(_get_url(self.host, self.inference_port, f"predictions/{model_name}"),
                                           data=data,
                                           timeout=self.timeout
                                           )
        return inference_response.json()

    @error_wrapper(error_return_value=False)
    def infer(self, model_name: str, image: str | np.ndarray, **kwargs):
        try:
            return self._infer_without_worker_check(model_name, image, **kwargs)
        except Exception as e:
            self.logger.error(f"Error while inferring: {e}")
            self.scale(model_name, min_worker=1)
            return self._infer_without_worker_check(model_name, image, **kwargs)

    @error_wrapper(error_return_value=())
    def list_models(self, limit: int = 100, next_page_token: str = ""):
        return requests.get(
            _get_url(self.host, self.management_port, "models"),
            timeout=self.timeout
        ).json()

    @error_wrapper(error_return_value=())
    def describe_model(self, model_name: str):
        return requests.get(
            _get_url(self.host, self.management_port, f"models/{model_name}"),
            timeout=self.timeout).json()


if __name__ == "__main__":
    manager = TorchserveRestManager(
        inference_port=8080,
        management_port=8081,
        model_store_path="model_store",
        model_path="model",
        stop_if_running=True,
        max_torchserve_waiting_time=60,
        image_encoder="jpg",
        delete_temp_folder=True
    )
    print(manager.health_check())
