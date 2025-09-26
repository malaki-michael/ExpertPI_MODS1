import json
import os
import subprocess
from time import sleep, time
from typing import Dict

import cv2
import grpc
import numpy as np

from serving_manager.management import inference_pb2, inference_pb2_grpc, management_pb2, management_pb2_grpc
from serving_manager.management.torchserve_base_manager import ConfigProperties, error_wrapper, TorchserveBaseManager


class TorchserveGrpcManager(TorchserveBaseManager):

    def __init__(
            self,
            inference_port: str | None = None,
            management_port: str | None = None,
            host: str = "localhost",
            model_store_path: str = "./model_store",
            model_path: str | None = None,
            stop_if_running: bool = False,
            config_properties: Dict[str, str] | ConfigProperties = ConfigProperties(),
            max_torchserve_waiting_time: int = 60,
            image_encoder: str = ".jpg",
            delete_temp_folder: bool = True,
            cache_stub: bool = True,
            timeout: float = 30.0,
            ) -> None:
        """Torchserve manager constructor. This class is used to run, manage, setup and control
        torchserve server. It uses solely gRPC API to communicate with the server. Currently, inference
        parameters are not supported.

        Args:
            inference_port (str | None, optional): Inference port, where GRPC runs. Defaults to None.
            management_port (str | None, optional): Management port where GRPC torchserve may be controlled.
                Defaults to None.
            host (str, optional): Hostname. Defaults to "localhost".
            model_store_path (str, optional): Where models are stored, path.
                Defaults to "./model_store".
            model_path (str | None, optional): Path to .mar file, replaces model_store_path.
                Defaults to None.
            cache_stub (bool, optional): Whether to cache stub or reacreate new each time.
                Defaults to True.
            stop_if_running (bool, optional): Stop torchserve if it is already running.
                Defaults to False.
            config_properties (Dict[str, str] | ConfigProperties, optional): Torchserve config.
                Defaults to ConfigProperties().
            max_torchserve_waiting_time (int, optional): Max time to wait for torchserve to start.
            image_encoder (str, optional): Image encoder, currently only jpg | png is supported.
            delete_temp_folder (bool, optional): Delete temporary folder after torchserve is stopped.
                Defaults to True.
            timeout (float, optional): Timeout for gRPC requests. Defaults to 2.0.
        """
        super().__init__(
            inference_port=inference_port,
            management_port=management_port,
            host=host,
            model_store_path=model_store_path,
            model_path=model_path,
            stop_if_running=stop_if_running,
            config_properties=config_properties,
            max_torchserve_waiting_time=max_torchserve_waiting_time,
            image_encoder=image_encoder,
            delete_temp_folder=delete_temp_folder,
            timeout=timeout,
        )
        self.inference_stub: inference_pb2_grpc.InferenceAPIsServiceStub | None = None
        self.managment_stub: management_pb2_grpc.ManagementAPIsServiceStub | None = None
        self.cache_stub = cache_stub

    def get_inference_stub(self):
        if self.inference_stub is None:
            channel = grpc.insecure_channel(f'{self.host}:{self.inference_port}')
            stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
            if self.cache_stub:
                self.inference_stub = stub
            return stub
        return self.inference_stub

    def get_management_stub(self):
        if self.managment_stub is None:
            channel = grpc.insecure_channel(f'{self.host}:{self.management_port}')
            stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
            if self.cache_stub:
                self.managment_stub = stub
            return stub
        return self.managment_stub

    def create_config_properties(self):
        port_settings = {}
        if self.inference_port:
            port_settings["grpc_inference_port"] = self.inference_port
        if self.management_port:
            port_settings["grpc_management_port"] = self.management_port
        super().create_config_properties(port_settings=port_settings)

    def stop(self):
        self.logger.info("Trying to gracefully shutdown the torchserve server")
        subprocess.run(["torchserve", "--stop"])
        if os.path.exists(self.CONFIG_PROPERTIES_FILE):
            os.remove(self.CONFIG_PROPERTIES_FILE)

    @error_wrapper(error_return_value={"status": "Unhealthy"})
    def health_check(self):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        try:
            return json.loads(self.get_inference_stub().Ping(
                inference_pb2.TorchServeHealthResponse(), timeout=self.timeout).health)
        except grpc._channel._InactiveRpcError:
            return {"status": "Unavailable"}

    @error_wrapper(error_return_value={})
    def infer_without_worker_check(self, model_name: str, image: str | np.ndarray, **kwargs):
        """Method for infering images with torchserve. Additional parameters should be supported
            It does not check if the model is running, so it is faster when error occurs.
        Args:
            model_name (str): Name of the inferred model
            image (str | np.ndarray): Image in base64 or numpy array format
            kwargs: Additional parameters for the model, like masks, thresholds etc.
        Returns:
            Dict: Processed data from the model
        """
        if isinstance(image, str):
            with open(image, 'rb') as im_file:
                data = im_file.read()
        else:
            data = cv2.imencode("{}".format(self.image_encoder), image)[1].tostring()

        response = self.get_inference_stub().Predictions(
            inference_pb2.PredictionsRequest(
                model_name=model_name, input={"data": data, **kwargs}), timeout=self.timeout)
        return json.loads(response.prediction.decode('utf-8'))

    @error_wrapper(error_return_value={})
    def infer(self, model_name: str, image: str | np.ndarray, **kwargs):
        """Method for infering images with torchserve. Additional parameters should be supported
            Compared to infer, this method checks if there are any workers available. If not, it
            will scale the model to 1 worker.

        Args:
            model_name (str): Name of the inferred model
            image (str | np.ndarray): Image in base64 or numpy array format
            kwargs: Additional parameters for the model, like masks, thresholds etc.
        Returns:
            Dict: Processed data from the model
        """
        try:
            return self.infer_without_worker_check(model_name, image, **kwargs)
        except grpc._channel._InactiveRpcError:
            num_workers = self.get_num_workers(model_name)
            if num_workers == 0:
                self.scale(model_name, 1)

            # NOTE: even if num_workers was > 0, retrying is OK.
            return self.infer_without_worker_check(model_name, image, **kwargs)

    @error_wrapper(error_return_value=False)
    def register(self, model_name: str, path: str = "", initial_workers: int = 0):
        marfile = os.path.join(path, f"{model_name}.mar")
        self.logger.info(f"## Register marfile: {marfile}, initial_workers: {initial_workers}\n")
        params = {
            'url': marfile,
            'initial_workers': initial_workers,
            'synchronous': True,
            'model_name': model_name
        }
        if initial_workers == 0:
            del params["initial_workers"]
        response = self.get_management_stub().RegisterModel(
            management_pb2.RegisterModelRequest(**params), timeout=self.timeout)
        return self._return_with_json_check(response)

    @error_wrapper(error_return_value=False)
    def unregister(self, model_name: str):
        self.logger.info(f"Trying to unregister model: {model_name}")
        response = self.get_management_stub().UnregisterModel(
            management_pb2.UnregisterModelRequest(model_name=model_name), timeout=self.timeout)
        return self._return_with_json_check(response)

    @error_wrapper(error_return_value=())
    def list_models(self, limit: int = -1, next_page_token: str = 0):
        self.logger.info("Listing models")
        response = self.get_management_stub().ListModels(
            management_pb2.ListModelsRequest(limit=limit, next_page_token=next_page_token), timeout=self.timeout)
        return self._return_with_json_check(response)

    @error_wrapper(error_return_value=False)
    def scale(self, model_name: str, min_worker: int = 1, max_worker: int | None = None, synchronous: bool = True):
        self.logger.info(f"Scaling model: {model_name}")
        if min_worker == 0:
            self.logger.warning(f"Currently this endpoint unregister model, use REST API to scale to 0 workers.")
            return self._scale_to_zero(model_name)

        max_worker = max_worker if max_worker is not None else min_worker
        response = self.get_management_stub().ScaleWorker(
            management_pb2.ScaleWorkerRequest(model_name=model_name,
                                              model_version=None,
                                              min_worker=min_worker,
                                              max_worker=max_worker,
                                              synchronous=synchronous), timeout=self.timeout)
        return self._return_with_json_check(response)

    # TODO: workarround for scaling models to 0 workers.
    def _scale_to_zero(self, model_name: str):
        self.unregister(model_name)
        return self.register(model_name, initial_workers=0)

    @staticmethod
    def _return_with_json_check(response):
        try:
            return json.loads(response.msg)
        except json.decoder.JSONDecodeError:
            return response.msg

    def get_num_workers(self, model_name: str):
        self.logger.info(f"Getting number of workers for model: {model_name}")
        model_info = self.describe_model(model_name)
        if len(model_info):
            return len(model_info[0]["workers"])
        return -1

    def describe_model(self, model_name: str, version: int = "all"):
        self.logger.info(f"Describing model: {model_name}")
        try:
            return json.loads(self.get_management_stub().DescribeModel(
                management_pb2.DescribeModelRequest(model_name=model_name, model_version=version), timeout=self.timeout).msg)
        except grpc._channel._InactiveRpcError:
            return []

    def clean_up(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            if not len(os.listdir(os.path.split(self.model_path)[0])):
                os.rmdir(os.path.split(self.model_path)[0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TorchServe manager, helping to do the inference")
    parser.add_argument("--inference-port", type=int, default=7443)
    parser.add_argument("--management-port", type=int, default=7444)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model-store-path", type=str, default="model_store")
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--model-name", "-m", type=str, required=True)
    parser.add_argument("--output_json_path", "-o", type=str, default="output.json")
    parser.add_argument("--stop-if-running", action="store_true")

    args = parser.parse_args()

    manager = TorchserveGrpcManager(
        inference_port=args.inferece_port,
        management_port=args.management_port,
        host=args.host,
        model_store_path=args.model_store_path,
        stop_if_running=args.stop_if_running
    )

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    img = img / np.amax(img)
    img = (img * 255).astype(np.uint8)

    output = manager.infer_without_worker_check(args.model_name, img)
    with open(args.output_json_path, "w") as fp:
        json.dump(output, fp, indent=4)
