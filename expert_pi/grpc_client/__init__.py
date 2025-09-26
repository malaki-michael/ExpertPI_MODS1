# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256
from packaging.version import parse
import warnings

from .modules import gun 
from .modules import illumination 
from .modules import microscope 
from .modules import projection 
from .modules import scanning 
from .modules import server 
from .modules import stage 
from .modules import stem_detector 
from .modules import xray 
from . import channel


def connect(host: str, port=881):
    """Connect to a pySTEM server

    Args:
        host (str): Hostname or IP address of the server
        port (int, optional): Port number of the server. Defaults to 881.
    """
    channel.connect(f"{host}:{port}")

    gun.connect()
    illumination.connect()
    microscope.connect()
    projection.connect()
    scanning.connect()
    server.connect()
    stage.connect()
    stem_detector.connect()
    xray.connect()

    server_version = server.get_version().split(",")[0]
    client_version = "1.16.1"
    if parse(server_version) < parse(client_version):
        warnings.warn(f"Server version {server_version} is lower than client version {client_version}")
