import numpy as np
from PySide6 import QtCore, QtWidgets

from expert_pi.automations import xyz_tracking
from expert_pi.gui.elements import buttons, combo_box, labels, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class Acquisition(base.Toolbar):
    def __init__(self, panel_size: int):
        base.Toolbar.__init__(self, name="3DED", icon=images_dir + "section_icons/3DED.svg", panel_size=panel_size)

        self.reference_button = buttons.ToolbarStateButton(
            "TAKE REF",
            icon_off=images_dir + "tools_icons/xyz_tracking_take.svg",
            icon=images_dir + "tools_icons/stop.svg",
        )

        self.test_button = buttons.ToolbarStateButton(
            "TEST", icon_off=images_dir + "tools_icons/start.svg", icon=images_dir + "tools_icons/stop.svg"
        )

        self.acquire_button = buttons.ToolbarStateButton(
            "ACQUIRE", icon_off=images_dir + "tools_icons/start.svg", icon=images_dir + "tools_icons/stop.svg"
        )

        self.pause_button = buttons.ToolbarStateButton(
            "CONTINUE",
            name_off="PAUSE",
            icon_off=images_dir + "tools_icons/pause.svg",
            icon=images_dir + "tools_icons/start.svg",
        )

        self.test_button.setEnabled(False)
        self.acquire_button.setEnabled(False)
        self.pause_button.setEnabled(False)

        for button in [self.reference_button, self.test_button, self.acquire_button, self.pause_button]:
            button.setProperty("class", "toolbarButton big")
            button.setIconSize(QtCore.QSize(20, 20))
            button.setStyleSheet("text-align:left;padding-left:15px")

        self.content_layout.addWidget(self.reference_button, 0, 0, 1, 2)
        self.content_layout.addWidget(self.test_button, 1, 0, 1, 2)
        self.content_layout.addWidget(self.acquire_button, 2, 0, 1, 2)
        self.content_layout.addWidget(self.pause_button, 3, 0, 1, 2)

        self.max_angle = spin_box.PrefixSpinBoxWithUnits(
            55, [0.1, 89.9], 10, "deg", send_immediately=True, tooltip="max alpha angle"
        )
        self.num_measurements = spin_box.SpinBoxWithUnits(
            111, [2, 1e8], 10, "x", decimals=0, send_immediately=True, tooltip="number of measurements"
        )

        self.max_angle.set_signal.connect(lambda x: self.acquisition_parameters_updates())
        self.num_measurements.set_signal.connect(lambda x: self.acquisition_parameters_updates())

        self.content_layout.addWidget(self.max_angle, 4, 0, 1, 1)
        self.content_layout.addWidget(self.num_measurements, 4, 1, 1, 1)

        self.alpha_speed = spin_box.SpinBoxWithUnits(
            xyz_tracking.settings.alpha_speed, [0.1, 40], 5, "deg/s", send_immediately=True, tooltip="max alpha speed"
        )

        self.alpha_speed.set_signal.connect(lambda x: self.acquisition_parameters_updates())

        self.content_layout.addWidget(self.alpha_speed, 5, 0, 1, 2)

        self.stem_checkbox = QtWidgets.QCheckBox("STEM:")
        self.content_layout.addWidget(self.stem_checkbox, 6, 0, 1, 2)
        self.stem_checkbox.setChecked(True)

        self.pixel_time = spin_box.SpinBoxWithUnits(0.1, [0.1, 100_000_000], 1, "us")
        self.pixel_time.setToolTip("pixel time")

        self.pixels = combo_box.ToolbarComboBox()
        for i in range(4, 14):
            self.pixels.addItem(f"{2**i} px")
        self.pixels.setCurrentIndex(8 - 4)

        self.pixels.setToolTip("number of pixels of STEM image")

        self.content_layout.addWidget(self.pixel_time, 7, 0, 1, 1)
        self.content_layout.addWidget(self.pixels, 7, 1, 1, 1)

        self.stabilization_label = QtWidgets.QLabel("stability condition:")
        self.content_layout.addWidget(self.stabilization_label, 8, 0, 1, 2)
        self.xy_stability = spin_box.SpinBoxWithUnits(
            5, [0.01, 100], 1, "%", tooltip="xy required stability as a factor of fov"
        )
        self.z_stability = spin_box.SpinBoxWithUnits(
            10, [0.01, 500], 1, "%", tooltip="z required stability as a factor of fov"
        )
        self.content_layout.addWidget(self.xy_stability, 9, 0, 1, 1)
        self.content_layout.addWidget(self.z_stability, 9, 1, 1, 1)

        self.xy_stability.set_signal.connect(lambda x: self.stability_condition_changed())
        self.z_stability.set_signal.connect(lambda x: self.stability_condition_changed())

        self.stabilization_time = spin_box.SpinBoxWithUnits(
            xyz_tracking.settings.stabilization_time, [0.1, 10], 1, "s", tooltip="'wait' time after stable"
        )
        self.content_layout.addWidget(self.stabilization_time, 9, 2, 1, 1)

        self.skip_tracking_label = QtWidgets.QLabel("skip stability <")
        self.skip_angle = spin_box.SpinBoxWithUnits(
            2, [0.0001, 100], 0.5, " deg", tooltip="use tracking just for steps more then this value"
        )
        self.content_layout.addWidget(self.skip_tracking_label, 10, 0, 1, 1)
        self.content_layout.addWidget(self.skip_angle, 10, 1, 1, 1)

        self.diff_4d_label = QtWidgets.QLabel("4D STEM:")

        self.content_layout.addWidget(self.diff_4d_label, 11, 0, 1, 1)

        self.diff_4d_method = combo_box.ToolbarComboBox()
        self.diff_4d_method.addItem("point")
        self.diff_4d_method.addItem("raster")
        self.diff_4d_method.addItem("random")
        self.diff_4d_method.setCurrentIndex(0)

        self.diff_4d_method.currentIndexChanged.connect(self.method_changed)

        self.content_layout.addWidget(self.diff_4d_method, 11, 1, 1, 1)

        self.camera_exposure = spin_box.SpinBoxWithUnits(
            20, [0, 100000], 1, "ms", tooltip="camera exposure", send_immediately=True
        )
        self.diff_4d_pixels = combo_box.ToolbarComboBox()
        for i in range(10):
            self.diff_4d_pixels.addItem(f"{2**i} px")
        self.diff_4d_pixels.setCurrentIndex(0)

        self.camera_exposure.set_signal.connect(lambda x: self.acquisition_parameters_updates())
        self.diff_4d_pixels.currentIndexChanged.connect(lambda x: self.acquisition_parameters_updates())
        self.diff_4d_pixels.setEnabled(False)

        self.content_layout.addWidget(self.camera_exposure, 12, 0, 1, 1)
        self.content_layout.addWidget(self.diff_4d_pixels, 12, 1, 1, 1)

        self.info_4d = QtWidgets.QLabel("test")
        self.content_layout.addWidget(self.info_4d, 13, 0, 1, 2)

        self.info = labels.ThreadedLabel()
        self.content_layout.addWidget(self.info, 17, 0, 1, 2)

        self.acquisition_parameters_updates()
        self.expand(False)

    def sizeHint(self):
        return self.size()

    def expand(self, value):
        super().expand(value)
        if value:
            self.stabilization_time.show()
        else:
            self.stabilization_time.hide()

    def acquisition_parameters_updates(self, total_time_estimate=None):
        n_cam = int(self.diff_4d_pixels.currentText().split(" ")[0])
        if self.diff_4d_method.currentText() in {"point", "random"}:
            gb = self.num_measurements.value() / 1024  # 16bit frame=1MB
            time = self.camera_exposure.value() * 1e-3  # s
        else:
            # TODO rectangle selection
            gb = self.num_measurements.value() * n_cam**2 / 1024  # 16bit frame=1MB
            time = self.camera_exposure.value() * n_cam**2 * 1e-3  # s

        self.info_4d.setText(f"{gb:.1f} GB  {time:.3f}s/step")

    def method_changed(self):
        if self.diff_4d_method.currentText() == "point":
            self.diff_4d_pixels.setEnabled(False)

        else:
            self.diff_4d_pixels.setEnabled(True)

        self.acquisition_parameters_updates()

    def stability_condition_changed(self):
        xyz_tracking.settings.stabilization_time = self.stabilization_time.value()
        xyz_tracking.settings.allowed_error = np.array([
            self.xy_stability.value() / 100 * self.measured_fov,
            self.xy_stability.value() / 100 * self.measured_fov,
            self.z_stability.value() / 100 * self.measured_fov,
        ])

        self.stabilization_label.setText(
            f"stability: <{xyz_tracking.settings.allowed_error[0]:6.2f} | {xyz_tracking.settings.allowed_error[2]:6.2f} um"
        )
