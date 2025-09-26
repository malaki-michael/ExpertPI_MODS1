import os
from typing import List

import numpy as np

from serving_manager.management.torchserve_base_manager import ConfigProperties
from serving_manager.management.torchserve_grpc_manager import TorchserveGrpcManager
from serving_manager.mlflow.mlflow_manager import MLFlowManager


def clean_mlflow_model(model_name):
    if os.path.exists(model_name):
        os.remove(model_name)


def serve_mlflow_model(
        model_name,
        model_path: str | None = None,
        model_version: str | None = None,
        mlflow_uri: str | None = None,
        mlflow_s3_endpoint: str | None = None,
        inference_port: str = "7443",
        run: bool = True
    ):
    mlflow_manager = MLFlowManager(mlflow_uri, mlflow_s3_endpoint)

    if not model_version:
        model_version = mlflow_manager.get_latest_model_version(model_name)

    path_to_mar = mlflow_manager.download_model(model_name, model_version, model_path)
    torchserve_manager = TorchserveGrpcManager(
        model_path=path_to_mar,
        config_properties=ConfigProperties(),
        inference_port=inference_port,
    )
    if run:
        torchserve_manager.run()
    return torchserve_manager


def infer_mlflow_model(
        images: List[np.ndarray] | np.ndarray,
        model_name: str,
        model_path: str | None = None,
        model_version: str | None = None,
        mlflow_uri: str | None = None,
        mlflow_s3_endpoint: str | None = None,
        inference_port: str = "7443",
        clean_up: bool = True,
    ):

    if model_path is None:
        model_path = "./"

    torchserve_manager = serve_mlflow_model(
        model_name=model_name,
        model_path=model_path,
        model_version=model_version,
        mlflow_uri=mlflow_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
        inference_port=inference_port,
        run=False
    )
    with torchserve_manager:
        if clean_up:
            torchserve_manager.clean_up()

        if isinstance(images, np.ndarray):
            return torchserve_manager.infer(model_name, images)
        return [torchserve_manager.infer(model_name, image) for image in images]
