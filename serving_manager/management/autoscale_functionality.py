import dataclasses
import enum
import logging
import time
from threading import Event, Thread

import requests
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families


class MetricType(enum.Enum):
    ts_queue_latency_microseconds = "ts_queue_latency_microseconds"
    ts_inference_latency_microseconds = "ts_inference_latency_microseconds"
    ts_inference_requests = "ts_inference_requests"


@dataclasses.dataclass
class MetricHandler:
    metric: Metric

    @property
    def name(self):
        return self.metric.name

    @property
    def samples(self):
        return self.metric.samples

    @property
    def documentation(self):
        return self.metric.documentation

    @property
    def type(self):
        return self.metric.type

    def update(self, metric):
        self.metric = metric
        self.last_used = time.time() - self.last_used

    def __hash__(self):
        return hash(self.name)


@dataclasses.dataclass()
class Model:
    name: str
    requests: int
    last_used: float
    version: str = "default"
    latency: float = 0
    num_workers: int = 0

    def __hash__(self):
        return hash((self.name, self.version))


class CallerType(enum.Enum):
    METRICS = "metrics"
    MANAGEMENT = "management"


class ErrorReturn:

    def __init__(self, value, caller_type: CallerType, callback: callable = lambda: None, error_callback: callable = lambda: None):
        self.text = value
        self.callback = callback
        self.error_callback = error_callback
        self.caller_type = caller_type

    def json(self):
        return self.text


def exception_aware_request(func, error_return_fn: ErrorReturn, *args, **kwargs):
    try:
        return_value = func(*args, **kwargs)
        error_return_fn.callback(error_return_fn.caller_type)
        return return_value
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logging.getLogger(__name__).warning(f"Failed to connect to {args[0]}")
        error_return_fn.error_callback(error_return_fn.caller_type)
        return error_return_fn


class MetricManager:

    def __init__(self, host, metrics_port, management_port):
        self.host = host
        self.metrics_port = metrics_port
        self.management_port = management_port
        self.metrics = {e: None for e in MetricType}
        self.models = {}
        self.num_retries = {CallerType.METRICS: 0, CallerType.MANAGEMENT: 0}
        self.logger = logging.getLogger(__name__)

    def reset_retries(self, caller_type: CallerType):
        self.num_retries[caller_type] = 0

    def increment_retries(self, caller_type: CallerType):
        self.num_retries[caller_type] += 1

    def get(self):
        metrics = exception_aware_request(requests.get, ErrorReturn(
            "",
            caller_type=CallerType.METRICS,
            callback=self.reset_retries,
            error_callback=self.increment_retries
        ), f"http://{self.host}:{self.metrics_port}/metrics", timeout=1).text
        return [metric for metric in text_string_to_metric_families(metrics)]

    def update(self):
        metrics = self.get()
        for metric in metrics:
            self.metrics[MetricType(metric.name)] = MetricHandler(metric)

        self._update_models()
        return self

    def _list_models(self):
        models = exception_aware_request(
            requests.get,
            ErrorReturn(
                {"models": []},
                caller_type=CallerType.MANAGEMENT,
                callback=self.reset_retries,
                error_callback=self.increment_retries
            ),
            f"http://{self.host}:{self.management_port}/models", timeout=1).json()
        print(models)
        model_names = [model["modelName"] for model in models["models"]]
        self.logger.debug(f"Found models: {model_names}")
        return model_names

    def _get_num_workers(self, model_name):
        model = exception_aware_request(
            requests.get,
            ErrorReturn(
                [{"workers": []}],
                caller_type=CallerType.MANAGEMENT,
                callback=self.reset_retries,
                error_callback=self.increment_retries
            ),
            f"http://{self.host}:{self.management_port}/models/{model_name}",
            timeout=1).json()
        # TODO: Unsupported model version. Only default version is supported
        return len(model[0]["workers"])

    def _update_models(self):
        model_names = self._list_models()
        for model_name in model_names:
            if model_name not in self.models:
                    self.models[model_name] = Model(model_name, 0, last_used=time.time())
            self.models[model_name].num_workers = self._get_num_workers(model_name)
        for model, requests in self.total_requests_by_model().items():
            if model not in self.models:
                self.models[model] = Model(model, requests, last_used=time.time())
            elif self.models[model].requests != requests:
                model_latency = self.total_latency_by_model()
                if model in model_latency:
                    self.models[model].latency = self.total_latency_by_model()[model] / requests
                self.models[model].requests = requests
                self.models[model].last_used = time.time()
        return self

    def last_used_by_model(self):
        return {m.name: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(m.last_used)) for m in self.models.values()}

    def last_used_seconds_by_model(self):
        return {m.name: time.time() - m.last_used for m in self.models.values()}

    def delete_all_workers(self, model_name):
        exception_aware_request(
            requests.put,
            ErrorReturn(
                "",
                caller_type=CallerType.MANAGEMENT,
                callback=self.reset_retries,
                error_callback=self.increment_retries
            ),
            f"http://{self.host}:{self.management_port}/models/{model_name}",
            params={"min_worker": 0, "synchronous": "true"}, timeout=1)

    def total_requests_by_model(self):
        metric = self.metrics[MetricType.ts_inference_requests]
        if metric is None:
            return {}
        return {
            m.labels["model_name"]: m.value for m in metric.samples
        }

    def total_latency_by_model(self):
        metric = self.metrics[MetricType.ts_inference_latency_microseconds]
        if metric is None:
            return {}
        return {
            m.labels["model_name"]: m.value / 1000. for m in metric.samples
        }

    def mean_latency_by_model(self):
        total_requests = self.total_requests_by_model()
        for model, latency in self.total_latency_by_model().items():
            total_requests[model] = latency / total_requests[model]
        return total_requests

    def __str__(self) -> str:
        return f"MetricManager(host={self.host}, metrics_port={self.metrics_port}, management_port={self.management_port})"

    def __hash__(self):
        return hash(self.host + self.metrics_port + self.management_port)


class AutoscalerFunctionalityProcess(Thread):

    def __init__(
            self,
            host: str,
            metrics_port: str,
            management_port: str,
            timeout: float = 3600,
            repete_time: float = 5,
            max_num_retries: int = 20,
            *args,
            **kwargs
        ):
        self.repete_time = repete_time
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.metric_manager = MetricManager(host=host, metrics_port=metrics_port, management_port=management_port)
        self.max_num_retries = max_num_retries
        super(AutoscalerFunctionalityProcess, self).__init__(*args, **kwargs)
        self._stop = Event()
        self.daemon = True

    def last_used_by_model(self):
        return self.metric_manager.last_used_by_model()

    def total_requests_by_model(self):
        return self.metric_manager.total_requests_by_model()

    def total_latency_by_model(self):
        return self.metric_manager.total_latency_by_model()

    def mean_latency_by_model(self):
        return self.metric_manager.mean_latency_by_model()

    def check_num_retries(self):
        if any([m >= self.max_num_retries for m in self.metric_manager.num_retries.values()]):
            self.logger.warning(f"AutoscalerFunctionalityProcess with MetricsManager {self.metric_manager} stopped. "
                                f"Number of retries exceeded maximum number of retries: {dict({k.name: v for k, v in self.metric_manager.num_retries.items()})}.")
            self.stop()

    def run(self):
        while not self._stop.wait(self.repete_time):
            self.metric_manager.update()
            dif_last_used = self.metric_manager.last_used_seconds_by_model()
            for model, last_used in dif_last_used.items():
                if self.metric_manager.models[model].num_workers > 0 and last_used > self.timeout:
                    self.metric_manager.delete_all_workers(model)
                    self.logger.info(f"Model {model} scaled down")
            self.check_num_retries()
        self.logger.warning(f"AutoscalerFunctionalityProcess with MetricsManager {self.metric_manager} stopped.")

    def stop(self):
        self._stop.set()

    def terminate(self):
        self.stop()

    def __hash__(self) -> int:
        return hash(self.metric_manager)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, AutoscalerFunctionalityProcess):
            return False
        return hash(self) == hash(__value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    x = AutoscalerFunctionalityProcess(host="localhost", metrics_port="8089", management_port="8081", repete_time=1, timeout=1e5)
    x.start()
    time.sleep(600)
    x.terminate()