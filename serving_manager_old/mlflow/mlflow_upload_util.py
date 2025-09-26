import logging
import os
import mlflow
from mlflow.pytorch import log_model
import torch


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def _setup_mlflow(hostname: str = "172.19.1.16"):
    response = os.system("ping -c 1 " + hostname)

    if response != 0:
        logging.getLogger(__name__).warning("MLFlow not available")
        return

    mlflow_uri: str = f"http://{hostname}:5000"
    mlflow_s3_endpoint_url: str = f"http://{hostname}:9000"
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    if not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url


USER = "Branislav Hesko"
EXPERIMENT = "PointMatcher"
PATH_TO_MAR_FILE = "/home/branislavhesko/models/model_store/PointMatcher.mar"
MODEL_NAME = "PointMatcher"
DESCRIPTION = "PointMatcher"
PARAMETERS_TO_LOG = {"model_base": "TEMRegistration", "model_base_url": "http://172.19.1.16:5000/#/experiments/3/runs/1fa0af52868047ba80b265d258ffd06f"}
MODEL_INFO = "Point matching model, which based on input detects corresponding point."
GIT_REPO_URL = "https://github.com/tescan-orsay-holding/TEM_registration_toolbox/blob/main/tem_registration/utils/deploy_point_matcher_from_mar.py"

_setup_mlflow()

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.runName", MODEL_NAME)
    mlflow.set_tag("mlflow.source.type", "LOCAL")
    mlflow.set_tag("mlflow.source.name", PATH_TO_MAR_FILE)
    mlflow.set_tag("mlflow.user", "serving_manager")
    mlflow.set_tag("mlflow.source.git.commit", "master")
    mlflow.set_tag("mlflow.source.git.repoURL", GIT_REPO_URL)
    log_model(DummyModel(), "model", registered_model_name=MODEL_NAME)
    client = mlflow.tracking.MlflowClient()
    mlflow.log_artifact(PATH_TO_MAR_FILE, "serving")
    client.update_registered_model(MODEL_NAME,
                                   DESCRIPTION)
    current_model = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])[-1]
    client.transition_model_version_stage(current_model.name, current_model.version, "Production")
    client.update_model_version(current_model.name, current_model.version, description="Serving model is stored in artifact serving.\n\n{}".format(
                                MODEL_INFO))
