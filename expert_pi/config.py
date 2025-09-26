"""Configuration of ExpertPI application."""
import dataclasses
import os
import easygui as g
from pyaml import yaml


ports = {
    "local": "127.0.0.1",
    "F2": "tensor-f2",
    "P1":"172.19.1.22",
    "P3": "172.19.1.15",
    "P4":"192.168.34.69"}

port_selected = g.choicebox("select microscope","Scope",ports)
print(f"Connecting to {port_selected} on {ports[port_selected]}")

microscope = ports[port_selected]

@dataclasses.dataclass
class ConnectionConfig:
    """Contains all the configurations for the connection."""

    host: str = microscope #"127.0.0.1"
    camera_port: int = 861
    stem_port: int = 862
    edx_port: int = 863
    cache_port: int = 871

    mqtt_broker_port: int = 853
    mqtt_stem_server_name: str = "stem"


@dataclasses.dataclass
class InferenceConfig:
    """Contains all the configurations for the inference."""

    host: str = "192.168.50.3"
    inference_port: int = 8600
    management_port: int = 8081
    plugin_port: int = 7443
    deprecession_endpoint: str = "deprecession_mask_body"
    metric_name: str = "best_1_area"
    model_name: str = "diffraction_spot_segmentation"
    endpoint: str = ""
    num_objects: int = 10

    def __post_init__(self):
        self.endpoint = f"http://{self.host}:{self.plugin_port}/quality/{self.deprecession_endpoint}"


@dataclasses.dataclass
class UIConfig:
    """Contains all the configurations for the UI."""

    panel_size: int = 180
    load_controllers: bool = True
    available_measurements: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataConfig:
    """Contains all the configurations for the data."""

    save_metadata: bool = True
    measurement_size_memory_limit: int = 1  # GB
    data_folder: str = "./data"
    navigation_cache: str = "./data/navigation_map.h5"
    stem_4d_cache: str = "./data/survey_4d.h5"


@dataclasses.dataclass
class NavigationConfig:
    """Contains all the configurations for the navigation."""

    tile_n: int = 512
    max_size: int = 4000  # um
    max_zoom: int = 16

    max_supertile: int = 8  # max number of tiles to be acquired at single stage position
    max_fov: float = 251 * 1.25  # limit due to LM cropping

    pixel_time: float = 0.3  # us

    tile_overlap: int = 1
    tile_overlap_add: int = 0

    show_interpolated: bool = True

    # for determining the order of tile acquisitions:
    stage_fact: tuple[float, float] = dataclasses.field(
        default_factory=lambda: (4.0, 1.5)
    )  # prefered x direction since it is faster
    center_fact: tuple[float, float] = dataclasses.field(default_factory=lambda: (1.5, 1.5))


class Config:
    """Contains all the configurations for the ExpertPI."""

    connection: ConnectionConfig
    inference: InferenceConfig
    ui: UIConfig
    data: DataConfig
    navigation: NavigationConfig

    def __init__(self, config_file: str | None = None):
        """Initialize the Config object.

        Args:
            config_file (str | None): Path to the configuration file. If None, an empty configuration dictionary is
                                      created. If the file does not exist, a default configuration dictionary
                                      is created.
        """
        if config_file is None:
            config_dict = {}
        elif not os.path.exists(config_file):
            config_dict = {}
            print(f"Config file not found: {config_file}, use default configurations.")
        else:
            with open(config_file, encoding="utf-8") as file:
                config_dict = yaml.safe_load(file)

        mapping = {
            "connection": ConnectionConfig,
            "inference": InferenceConfig,
            "ui": UIConfig,
            "data": DataConfig,
            "navigation": NavigationConfig,
        }

        # TODO: use dataclasses.fields() to check fields
        for key, value in mapping.items():
            if key in config_dict:
                setattr(self, key, value(**config_dict[key]))
            else:
                setattr(self, key, value())


"""Global configuration object."""
_config: Config | None = None


def get_config() -> Config:
    """Return the configuration object."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(configs: Config):
    """Set the configuration object."""
    global _config
    _config = configs
