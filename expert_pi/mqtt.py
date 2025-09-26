import time
from collections.abc import Callable

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


class MqttClient:
    """Client to connect to the MQTT broker and subscribe to topics."""

    def __init__(self, host: str, port: int):
        """Initializes the MqttClient.

        Args:
            host (str): The host of the MQTT broker.
            port (int): The port of the MQTT broker.
        """
        self._registered_subscriptions: dict[str, dict[int, Callable[[bytes], None]]] = {}

        self._host = host
        self._port = port
        self._mqttc = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1, transport="websockets")
        self._mqttc.on_message = self._on_message
        self._mqttc.on_connect = self._on_connect
        self._mqttc.on_disconnect = self._on_disconnect

    def start(self):
        """Connect to the MQTT broker and subscribes to the registered topics."""
        if not self._mqttc.is_connected():
            self._mqttc.connect(host=self._host, port=self._port)
            self._mqttc.loop_start()
            for topic in self._registered_subscriptions:
                self._mqttc.subscribe(topic)

    def stop(self):
        """Disconnects from the MQTT broker."""
        if self._mqttc.is_connected():
            self._mqttc.disconnect()
        self._mqttc.loop_stop()

    # TODO: automatically parse messages and add typing
    def on_topic(self, topic: str, callback: Callable[[bytes], None]) -> int:
        """Subscribes to a topic.

        Args:
            topic (str): The topic to subscribe to.
            callback (Callable[[str], None]): The callback to call when a message is received.

        Returns:
            int: The handle id for the subscription.
        """
        if topic not in self._registered_subscriptions:
            self._registered_subscriptions[topic] = {}
        handle_id = len(self._registered_subscriptions[topic])
        self._registered_subscriptions[topic][handle_id] = callback
        self._mqttc.subscribe(topic)
        return handle_id

    def off_topic(self, topic: str, handle_id: int):
        """Unsubscribes from a topic.

        Args:
            topic (str): The topic to unsubscribe from.
            handle_id (int): The handle id for the subscription.
        """
        del self._registered_subscriptions[topic][handle_id]
        if len(self._registered_subscriptions[topic]) == 0:
            self._mqttc.unsubscribe(topic)
            del self._registered_subscriptions[topic]

    def get_all_topics(self, cumulate_time: float = 3.0) -> list[str]:
        """Get all the topics from the MQTT broker.

        Args:
            cumulate_time (float, optional): The time to wait for the topics. Defaults to 3.
        """
        data = []

        def on_message(msg):
            if msg.topic not in data:
                data.append(msg.topic)

        handle_id = self.on_topic("#", on_message)
        time.sleep(cumulate_time)
        self.off_topic("#", handle_id)
        return data

    def _on_message(self, mqttc, obj, msg):
        # print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic in self._registered_subscriptions:
            for handle_id, callback in self._registered_subscriptions[msg.topic].items():
                callback(msg.payload)
        if "#" in self._registered_subscriptions:
            for handle_id, callback in self._registered_subscriptions["#"].items():
                callback(msg)

    def wait_for_event(
        self, topic: str, condition: Callable[[str], bool] = lambda x: True, timeout: float = 10.0, output: bool = True
    ):
        """Waits for an event on the given topic.

        Args:
            topic (str): The topic to wait for.
            condition (Callable[[str], bool], optional): The condition to check for the event.
                                                         Defaults to lambda x: True.
            timeout (float, optional): The time to wait for the event. Defaults to 10.0.
            output (bool, optional): Whether to print the output. Defaults to True.
        """
        start = time.monotonic()
        t = 0
        done = False
        result = None

        def on_message(data):
            nonlocal done, result
            if condition(data):
                result = data
                done = True

        handle_id = self.on_topic(topic, on_message)

        while t < timeout:
            if done:
                break
            if output:
                print(f"{t:6.2f} waiting for event", topic, end="\r")
            time.sleep(0.2)
            t = time.monotonic() - start
        else:
            if output:
                print("waiting for " + topic + " timeout exceed")
        self.off_topic(topic, handle_id)
        return result

    @staticmethod
    def _on_connect(mqttc, obj, flags, rc):
        print("Connected to Mqtt")

    @staticmethod
    def _on_disconnect(mqttc, *args, **kwargs):
        print("Mqtt disconnected")
