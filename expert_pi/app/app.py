import json

import grpc
from PySide6 import QtCore

from expert_pi import grpc_client, mqtt, stream_clients
from expert_pi.app import microscope_controller
from expert_pi.app.modules import (
    acquisition_controller,
    adjustments_controller,
    diffraction_tools_controller,
    diffraction_view_controller,
    image_view_controller,
    live_stream_controller,
    navigation,
    stem_4d_controller,
    stem_tools_controller,
    survey_controller,
    tilt_axes_controller,
    xray_controler,
    xyz_tracking_controller,
)
from expert_pi.app.states import states_holder
from expert_pi.gui import main_window
from expert_pi.config import get_config


class MqttController(QtCore.QObject):
    """Controller for handling MQTT messages."""

    state_changed_signal = QtCore.Signal(str, bool, bool, bool)
    log_message_signal = QtCore.Signal(str)

    def __init__(
        self, window: main_window.MainWindow, states: states_holder.StatesHolder, mqtt_client: mqtt.MqttClient
    ):
        """Initialize the mqtt controller.

        Args:
            window: Main window.
            states: States holder.
            mqtt_client: MQTT client.
        """
        super().__init__()
        self._window = window
        self._states = states
        self._mqtt_client = mqtt_client

        self.state_changed_signal.connect(window.optics.state_button.change_state)
        self.log_message_signal.connect(self._process_pystem_log_message)

        self._mqtt_client.on_topic("tem/events/systemState", self._change_microscope_state)
        self._mqtt_client.on_topic("pystem/sample/sampleCoordinates", self._window.stem_adjustments.stage_update)
        self._mqtt_client.on_topic("pystem/sample/sampleCoordinates", self._window.diffraction.stage_update)
        self._mqtt_client.on_topic("pystem/logging/records", lambda x: self.log_message_signal.emit(x.decode()))

    def _change_microscope_state(self, x):
        data: dict = json.loads(x.decode())
        # '{"state":"Active","substate":"Standby","busy":false,"errors":[],"waitingEvents":[]}'

        del data["waitingEvents"]
        if data != self._states.last_microscope_state:
            self._states.last_microscope_state = data
            if data["state"] != "Active" or data["substate"] == "Error":
                if data["substate"] == "Error":
                    text = data["substate"]
                else:
                    text = data["state"]
                selected, busy, error = False, False, True
            else:
                text = data["substate"]
                selected = data["substate"] in {"Ready", "Acquiring"}
                busy, error = data["busy"], False

            self.state_changed_signal.emit(text, selected, busy, error)

    def _process_pystem_log_message(self, data):
        decoded = json.loads(data)

        if decoded["level"] == "WARNING":
            self._window.statusBar().setStyleSheet("orange")
            timeout = 10000
        elif decoded["level"] in {"ERROR", "FATAL"}:
            self._window.statusBar().setStyleSheet("color:red")
            timeout = 60000
        else:
            self._window.statusBar().setStyleSheet("")
            timeout = 2000
        msg = f"{decoded['created']} {decoded['name']} {decoded['msg']}"
        self._window.statusBar().showMessage(msg, timeout)


class MainApp:
    """Main controller for the application."""

    def __init__(self, window: main_window.MainWindow):
        """Initialize the main controller.

        Args:
            window: Main window.
            configs: Configuration object.
        """
        self.window = window

        config = get_config()
        host = config.connection.host
        com = config.connection
        self.mqtt_client = mqtt.MqttClient(config.connection.host, config.connection.mqtt_broker_port)
        self.camera_client = stream_clients.CameraLiveStreamClient(host, com.camera_port)
        self.stem_client = stream_clients.StemLiveStreamClient(host, com.stem_port)
        self.edx_client = stream_clients.EDXLiveStreamClient(host, com.edx_port)
        self.cache_client = stream_clients.CacheClient(host, com.cache_port)

        self.states_holder = states_holder.StatesHolder()
        self.states_controller = microscope_controller.MicroscopeController(window, self.states_holder)

        self.mqtt_controller = MqttController(window, self.states_holder, self.mqtt_client)

        self.image_view_controller = image_view_controller.ImageViewController(window, self.states_holder)

        self.diffraction_view_controller = diffraction_view_controller.DiffractionViewController(
            window, self.states_holder
        )

        self.live_streams_controller = live_stream_controller.LiveProcessingThreadController(
            self.window,
            self.states_holder,
            self.stem_client,
            self.camera_client,
            self.edx_client,
            self.image_view_controller.normalizer,
            self.diffraction_view_controller.normalizer,
        )

        self.adjustments_controller = adjustments_controller.AdjustmentsController(
            window, self.states_holder, self.live_streams_controller.emitters, self.cache_client
        )
        self.stem_tools_controller = stem_tools_controller.StemToolsController(window)
        self.xyz_tracking_controller = xyz_tracking_controller.XYZTrackingController(window, self.states_holder)
        self.tilt_axes_controller = tilt_axes_controller.TiltAxesController(
            window, self.states_holder, self.xyz_tracking_controller
        )

        self.xray_controller = xray_controler.XrayController(window, self.live_streams_controller.edx_map_processor)
        self.diffraction_tool_controller = diffraction_tools_controller.DiffractionsToolsController(
            window, self.states_holder
        )
        self.acquisition_controller = acquisition_controller.AcquisitionController(
            window, self.states_holder, self.live_streams_controller.emitters, self.cache_client
        )

        self.stem_4d_controller = stem_4d_controller.Stem4DController(
            window,
            self.states_holder,
            self.image_view_controller.normalizer,
            self.diffraction_view_controller.normalizer,
            self.live_streams_controller.fft_processor,
            self.cache_client,
        )

        self.survey_controller = survey_controller.SurveyController(window, self.states_holder)
        self.navigation_controller = navigation.NavigationController(window, self.states_holder, self.cache_client)

        self.measurements = {}

        self.window.tool_bar.menu.reconnect_action.triggered.connect(self.reconnect_to_microscope)

    def reconnect_to_microscope(self):
        """Reconnect to the microscope."""
        grpc_client.connect(get_config().connection.host)
        grpc_client.scanning.stop_scanning()
        grpc_client.server.enable_live_streaming(True)

        self.stem_client.disconnect()
        self.stem_client.connect()

        self.camera_client.disconnect()
        self.camera_client.connect()

        self.edx_client.disconnect()
        self.edx_client.connect()

        self.states_controller.synchronize()
        # self.acquisition_controller.synchronize()
        acquisition_controller.synchronize(self.window, self.states_holder)
        if self.window.scanning.fov_spin.value() != 0:
            self.window.image_view.set_fov(self.window.scanning.fov_spin.value())
        adjustments_controller.synchronize(self.window, self.states_holder)

        self.xyz_tracking_controller.synchronize()

        # measurement_name = self.window.measurement_central_widget.selected
        # if measurement_name is not None and measurement_name in self.window.measurement_central_widget.measurements:
        #     widget = self.window.measurement_central_widget.measurements[measurement_name]
        #     if widget.isVisible():
        #         if hasattr(widget, "synchronise"):
        #             widget.synchronise()

        self.mqtt_client.start()


def check_connection(host):
    channel = grpc.insecure_channel(f"{host}:881")
    future = grpc.channel_ready_future(channel)
    try:
        future.result(timeout=1)
    except grpc.FutureTimeoutError:
        return False

    return True
