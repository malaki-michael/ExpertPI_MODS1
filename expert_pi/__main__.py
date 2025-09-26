# ruff: noqa: F401
from importlib import import_module, reload

import matplotlib.pyplot as plt
import numpy as np

from expert_pi import (
    grpc_client,
    version,
)

# import it rather then create directly since startup dialog an other features can be started before this module
from expert_pi.gui.qt_app import qt_app

from expert_pi import config
from expert_pi.app import app
from expert_pi.gui import console_threads
from expert_pi.gui import main_window

# set global configurations before vreating the application
_configs = config.Config("./config/config.yml")
config.set_config(_configs)

is_online = app.check_connection(_configs.connection.host)

window = main_window.MainWindow()
controller = None
if _configs.ui.load_controllers:
    controller = app.MainApp(window)
    camera_client = controller.cache_client
    stem_client = controller.stem_client
    edx_client = controller.edx_client
    cache_client = controller.cache_client

for name, module_name in _configs.ui.available_measurements.items():
    module = import_module(f"expert_pi.extensions.{module_name}")
    window.measurements[name] = getattr(module, "MainView")(window)
    if controller is not None:
        controller.measurements[name] = getattr(module, "MainController")(
            window.measurements[name], controller, controller.states_holder, cache_client
        )


if is_online and controller is not None:
    controller.reconnect_to_microscope()


window.survey_view.show()
window.showMaximized()

if __name__ == "__main__":
    qt_app.exec()
