import json
import os

import grpc
from PySide6 import QtGui, QtWidgets

from expert_pi.config import Config
from expert_pi.app.states import preserved_state_saver


class StartDialog(QtWidgets.QDialog):
    def __init__(self, hosts, configs: Config):
        super().__init__()
        self.hosts = hosts
        self.configs = configs

        self.setWindowTitle("Select host to connect:")
        self.setWindowIcon(QtGui.QIcon("src/gui/style/images/icon.png"))

        self.setStyleSheet("font-size:20px")

        self.resize(300, 80)
        self.setLayout(QtWidgets.QHBoxLayout())

        self.host_select = QtWidgets.QComboBox()

        for name, ip in self.hosts.items():
            self.host_select.addItem(f"{name} ({ip})")

        if not os.path.exists("data/"):
            os.makedirs("data/")

        self.offline = True

        preserved_state_saver.load(configs.data.data_folder)

        try:
            with open("data/state.json", encoding="utf-8") as f:  # TODO state management
                _ = json.load(f)
            index = list(self.hosts.keys()).index(preserved_state_saver.actual_state["last_host"])
            self.offline = preserved_state_saver.actual_state["offline"]
        except:
            index = 0

        self.host_select.setCurrentIndex(index)  # TODO last state

        self.confirm_button = QtWidgets.QPushButton("OK")

        self.status = QtWidgets.QLabel("-")

        self.layout().addWidget(self.status)
        self.layout().addWidget(self.host_select)
        self.layout().addWidget(self.confirm_button)

        self.host_select.currentIndexChanged.connect(self.host_changed)

        self.confirm_button.clicked.connect(self.confirm_clicked)

        self.channel_future = None

        self.host_changed(index)

    def grpc_server_on(self, ip):
        if self.channel_future is not None:
            self.channel_future.cancel()

        self.status.setText("\u2717")
        channel = grpc.insecure_channel(ip + ":881")
        future = grpc.channel_ready_future(channel)
        future.add_done_callback(self.connectivity_changed)

    def connectivity_changed(self, result):
        self.status.setText("\u2713")
        self.offline = False

    def host_changed(self, index):
        ip = list(self.hosts.values())[index]
        self.offline = True
        self.grpc_server_on(ip)

    def confirm_clicked(self):
        name, host = list(self.hosts.items())[self.host_select.currentIndex()]
        self.configs.connection.host = host
        preserved_state_saver.actual_state["last_host"] = name
        preserved_state_saver.actual_state["offline"] = self.offline
        preserved_state_saver.save(self.configs.data.data_folder)

        self.accept()
