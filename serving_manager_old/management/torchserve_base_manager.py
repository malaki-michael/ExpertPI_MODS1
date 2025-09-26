from abc import ABCMeta, abstractmethod
import dataclasses
import logging
import os
import shutil
import subprocess
import uuid
from time import time, sleep
from typing import Callable

import numpy as np
import requests


def error_wrapper(exceptions: tuple[Exception] | Exception = Exception, error_return_value=None, callback_fn: Callable | None = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                logging.exception(f"Error in {func.__name__}")
                if callback_fn is not None:
                    callback_fn()
                return error_return_value
        return wrapper
    return decorator


@dataclasses.dataclass()
class ConfigProperties:
    grpc_inference_port: str = "7443"
    grpc_management_port: str = "7444"
    inference_address: str = "http://0.0.0.0:8080"
    management_address: str = "http://0.0.0.0:8081"
    metrics_address: str = "http://0.0.0.0:8082"
    install_py_dep_per_model: str = "true"
    default_workers_per_model: str = "1"


class TemporaryFolder:
    def __init__(self, path: str | None = None, delete: bool = True):
        self.path = path
        self.delete = delete

    def __enter__(self):
        cwd = os.getcwd()
        if self.path is None:
            self.path = os.path.join(cwd, str(uuid.uuid4()).split("-")[0])
            os.makedirs(self.path, exist_ok=False)
        return self.path

    def __exit__(self, *_):
        if self.path is not None and self.delete:
            shutil.rmtree(self.path)


class TorchserveBaseManager(metaclass=ABCMeta):
    CONFIG_PROPERTIES_FILE = "config.properties"

    def __init__(
            self,
            inference_port: str | None = None,
            management_port: str | None = None,
            host: str = "localhost",
            model_store_path: str = "model_store",
            model_path: str = "",
            stop_if_running: bool = False,
            config_properties: dict | None = None,
            max_torchserve_waiting_time: int = 60,
            image_encoder: str = ".jpg",
            delete_temp_folder: bool = True,
            timeout: float = 30.0,
            ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.inference_port = inference_port
        self.management_port = management_port
        self.host = host
        self.model_store_path = model_store_path
        self.model_path = model_path
        self.config_properties = config_properties
        self.stop_if_running = stop_if_running
        self.max_torchserve_waiting_time = max_torchserve_waiting_time
        if not image_encoder.startswith("."):
            image_encoder = "." + image_encoder
        self.image_encoder = image_encoder
        self.delete_temp_folder = delete_temp_folder
        self.timeout = timeout

    @abstractmethod
    def create_config_properties(self, port_settings: dict | None = None):
        if port_settings is None:
            port_settings = {}
        with open(self.CONFIG_PROPERTIES_FILE, "w") as f:
            config_properties = dataclasses.asdict(self.config_properties) if isinstance(
                self.config_properties, ConfigProperties) else self.config_properties
            config_properties = config_properties | port_settings
            for key, value in config_properties.items():
                f.write(f"{key}={value}\n")

    def run(self):
        """Starts torchserve instance. If model_path is not None, this .mar is used, otherwise
        torchserve starts under model_store_path.
        """

        if self.stop_if_running:
            self.stop()  # stops any running torchserve server

        # TODO: implement specific model run, not always all models are needed.
        self.create_config_properties()

        # TODO: is it ok to create tempdir everytime?
        with TemporaryFolder(delete=self.delete_temp_folder) as tmpdir:
            if self.model_path is not None:
                shutil.copy(self.model_path, tmpdir)
                self.model_store_path = tmpdir
            self.logger.info("Trying to start the torchserve server.")
            subprocess.Popen(["torchserve", "--start", "--ncs",
                              f"--model-store={self.model_store_path}", "--models=all"],
                             stdout=None, stderr=None, stdin=None, close_fds=True)
            start_time = time()
            while not self.health_check()["status"] == "Healthy" and time() - start_time < self.max_torchserve_waiting_time:
                self.logger.warning("Waiting for the torchserve server to start.")
                sleep(1)
            if not self.health_check()["status"] == "Healthy":
                self.logger.error("Torchserve server did not start in time.")

    @abstractmethod
    def health_check(self):
        pass

    @abstractmethod
    def infer_without_worker_check(self, model_name: str, image: str | np.ndarray, **kwargs):
        pass

    @abstractmethod
    def infer(self, model_name, image, **kwargs):
        pass

    @abstractmethod
    def register(self, model_name:str, path: str = "", initial_workers: int = 0):
        pass

    @abstractmethod
    def unregister(self, model_name:str):
        pass

    @abstractmethod
    def describe_model(self, model_name: str):
        pass

    def scale_all_to_zero(self):
        models = self.list_all_models()
        for model in models:
            self.scale(model, 0)

    def list_all_models(self):
        models = self.list_models()
        if not models:
            return []
        return [model["modelName"] for model in models["models"]]

    @abstractmethod
    def list_models(self, limit: int = 100, next_page_token: str = ""):
        self.logger.info("Listing all the models")

    @abstractmethod
    def scale(self, model_name: str, min_worker: int = 1, max_worker: int | None = None, synchronous: bool = True):
        self.logger.info("Scaling the model")
        max_worker = max_worker if max_worker else min_worker

    def __enter__(self):
        self.logger.info("TorchServe manager has been entered as a context manager.")
        self.run()
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self):
        self.logger.info("Trying to gracefully shutdown the torchserve server")
        subprocess.run(["torchserve", "--stop"])
        if os.path.exists(self.CONFIG_PROPERTIES_FILE):
            os.remove(self.CONFIG_PROPERTIES_FILE)