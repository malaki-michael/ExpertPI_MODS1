import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.gui.elements import buttons, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base

M2x2_names = ["xx", "xy", "yx", "yy"]


class PrecessionButton(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.diffraction = buttons.HoverSignalsButton(
            "Enable",
            selectable=True,
            tooltip="Precession in diffraction view only",
            icon=images_dir + "tools_icons/precession_diffraction.svg",
        )

        layout.addWidget(self.diffraction, 0, 0, 1, 3)
        self.diffraction.setLayoutDirection(QtGui.Qt.LayoutDirection.RightToLeft)  # icon on right side
        self.diffraction.setStyleSheet("text-align:left;padding-right:10px")

        self.image = buttons.HoverSignalsButton(
            "+",
            selectable=True,
            tooltip="Precession in image and diffraction view",
            icon=images_dir + "tools_icons/precession_image.svg",
        )

        layout.addWidget(self.image, 0, 3, 1, 1)
        self.image.setLayoutDirection(QtGui.Qt.LayoutDirection.RightToLeft)  # icon on right side

        self.diffraction.hover_enter.connect(self.hover_enter_diffraction)
        self.diffraction.hover_leave.connect(self.hover_leave_diffraction)

        self.image.hover_enter.connect(self.hover_enter_image)
        self.image.hover_leave.connect(self.hover_leave_image)

    def hover_enter_image(self):
        if not self.image.selected and not self.diffraction.selected:
            self.diffraction.setProperty("hover", True)
            self.diffraction.setStyleSheet(self.diffraction.styleSheet())

    def hover_leave_image(self):
        if self.diffraction.property("hover"):
            self.diffraction.setProperty("hover", False)
        if self.diffraction.property("hover_selected"):
            self.diffraction.setProperty("hover_selected", False)

            self.diffraction.setStyleSheet(self.diffraction.styleSheet())

    def hover_enter_diffraction(self):
        if self.image.selected and self.diffraction.selected:
            self.image.setProperty("hover_selected", True)
            self.image.setStyleSheet(self.image.styleSheet())

    def hover_leave_diffraction(self):
        if self.image.property("hover_selected"):
            self.image.setProperty("hover_selected", False)
            self.image.setStyleSheet(self.image.styleSheet())


class Precession(base.Toolbar):
    precession_clicked_signal = QtCore.Signal()

    def __init__(self, panel_size: int):
        super().__init__(
            "PRECESSION", images_dir + "section_icons/precession.svg", expand_type="right", panel_size=panel_size
        )
        self.enabled = PrecessionButton()
        self.enabled.diffraction.clicked.connect(self.precession_diffraction_clicked)
        self.enabled.image.clicked.connect(self.precession_image_clicked)

        self.precession_angle = spin_box.SpinBoxWithUnits(17, [0, 60], 1, "mrad")
        self.precession_angle.setToolTip("precession angle")

        self.precession_frequency = spin_box.SpinBoxWithUnits(72, [0, 1000], 1, "kHz")

        self.deprecession_enabled = QtWidgets.QCheckBox("deprecession")
        self.deprecession_enabled.setChecked(True)

        self.cycles_in_stem = spin_box.SpinBoxWithUnits(
            1, [1, 1000], 1, "x", decimals=0, tooltip="number of precession cycles in stem image"
        )

        self.precession_method = QtWidgets.QComboBox(self)
        self.precession_method.addItems(["Sobel filter"])
        self.precession_method.setToolTip("Pick metric for precession tunning")

        self.deprecession_method = QtWidgets.QComboBox(self)
        self.deprecession_method.addItems(["Sobel filter"])
        self.deprecession_method.setToolTip("Pick metric for deprecession tunning")

        self.auto_precession = buttons.ToolbarPushButton("Pivot height tune", selectable=True)

        # self.auto_precession.clicked.connect(self.auto_precession_clicked)
        self.content.layout().addWidget(self.auto_precession, 1, 2, 1, 1)

        self.precession_auto_factor = spin_box.SpinBoxWithUnits(1, [0, 1000], 0.1, "x")
        self.precession_auto_factor.setToolTip("factor of fov \n for initial step of pivot point height")
        self.content.layout().addWidget(self.precession_auto_factor, 1, 3, 1, 1)

        self.auto_deprecession = buttons.ToolbarPushButton("Deprecession tune", selectable=True)
        # self.auto_deprecession.clicked.connect(self.auto_deprecession_clicked)
        self.content.layout().addWidget(self.auto_deprecession, 4, 2, 1, 1)
        self.deprecession_auto_factor = spin_box.SpinBoxWithUnits(1, [0, 1000], 0.1, "x")
        self.deprecession_auto_factor.setToolTip("factor of precession angle \n for initial step of autodeprecession")
        self.content.layout().addWidget(self.deprecession_auto_factor, 4, 3, 1, 1)

        self.precession_thread = None

        self.heights = {}
        self.deprecession_corrections = {}
        for i, name in enumerate(M2x2_names):
            self.heights[name] = spin_box.SpinBoxWithUnits(0, [-100_000, 100_000], 1, "nm", send_immediately=True)
            self.heights[name].setToolTip("Precession pivot point height " + name)

            # self.heights[name].set_signal.connect(self.set_precession_height_correction)
            self.content.layout().addWidget(self.heights[name], 2 + i // 2, 2 + i % 2, 1, 1)

            self.deprecession_corrections[name] = spin_box.SpinBoxWithUnits(
                0, [-100_000, 100_000], 0.1, "mrad", send_immediately=True
            )
            self.deprecession_corrections[name].setToolTip("Deprecession angle correction " + name)
            # self.deprecession_corrections[name].set_signal.connect(self.set_deprecession_tilt_correction)
            self.content.layout().addWidget(self.deprecession_corrections[name], 5 + i // 2, 2 + i % 2, 1, 1)

        self.content.layout().addWidget(self.precession_frequency, 0, 0, 1, 1)
        self.content.layout().addWidget(self.cycles_in_stem, 0, 1, 1, 1)
        self.content.layout().addWidget(self.enabled, 0, 2, 1, 1)
        self.content.layout().addWidget(self.precession_angle, 0, 3, 1, 1)

        self.content.layout().addWidget(self.precession_method, 2, 0, 1, 2)
        self.content.layout().addWidget(self.deprecession_enabled, 4, 0, 1, 2)
        self.content.layout().addWidget(self.deprecession_method, 5, 0, 1, 2)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        for widget in [
            self.precession_frequency,
            self.deprecession_enabled,
            self.precession_method,
            self.deprecession_method,
            self.cycles_in_stem,
        ]:
            widget.show() if value else widget.hide()

    def precession_image_clicked(self):
        if self.enabled.image.selected:
            self.enabled.diffraction.set_selected(True)
        self.precession_clicked_signal.emit()

    def precession_diffraction_clicked(self):
        if not self.enabled.diffraction.selected:
            if self.enabled.image.property("hover_selected"):
                self.enabled.image.setProperty("hover_selected", False)
                self.enabled.image.setProperty("hover", True)
            self.enabled.image.set_selected(False)
        self.precession_clicked_signal.emit()

    def get_minimal_pixel_time(self):
        return np.round(1e6 / (self.precession_frequency.value() * 1e3), 2)  # in us

    @staticmethod
    def pick_precession_kernel(combobox_value: str):
        if "sobel" in combobox_value.lower():
            return "sobel"
        elif "gabor" in combobox_value.lower():
            return "gabor"
        raise ValueError(f"Unknown kernel {combobox_value}!")

    @staticmethod
    def pick_deprecession_kernel(combobox_value: str):
        if "sobel" in combobox_value.lower():
            return "sobel"
        elif "spot detection" in combobox_value.lower():
            return "spot_detection"
        raise ValueError(f"Unknown kernel {combobox_value}!")

    # def set_precession_height_correction(self):
    #     M = []
    #     for name in M2x2_names:
    #         M.append(self.heights[name].value()*1e-9)
    #     grpc_client.scanning.set_precession_height_correction(M)

    # def set_deprecession_tilt_correction(self):
    #     M = []
    #     for name in M2x2_names:
    #         M.append(self.deprecession_corrections[name].value()*1e-3)
    #     grpc_client.scanning.set_deprecession_tilt_correction(M)

    # def auto_precession_clicked(self):
    #     if self.auto_precession.selected and (self.precession_thread is None or not self.precession_thread.is_alive()):
    #         fov = self.main_window.scanning.fov.value()*self.precession_auto_factor.value()
    #         N = int(self.main_window.scanning.N.currentText().split(" ")[0])
    #         if self.main_window.image_view.rectangle_selector.rectangle_active():
    #             rectangle = self.main_window.image_view.rectangle_selector.get_scan_rectangle()
    #         else:
    #             rectangle = None

    #         def set_callback(value):
    #             for i, name in enumerate(M2x2_names):
    #                 self.heights[name].setProperty("busy", True)
    #                 self.heights[name].updateStyle.emit()
    #                 self.heights[name].setReadSentSignal.emit(value[i]*1000)

    #         def target():
    #             try:
    #                 precession.focus_pivot_points(init_step_diagonal=fov, init_step_non_diagonal=fov/5,
    #                                               kernel=self._pick_precession_kernel(self.precession_method.currentText()),
    #                                               blur='median', N=N, rectangle=rectangle, pixel_time=1/(self.precession_frequency.value()*1e3),
    #                                               stop_callback=lambda: not self.auto_precession.selected, set_callback=set_callback)
    #             except precession.StopException:
    #                 pass

    #             self.auto_precession.update_selected_signal.emit(False)

    #             for i, name in enumerate(M2x2_names):
    #                 self.heights[name].setProperty("busy", False)
    #                 self.heights[name].updateStyle.emit()

    #         self.precession_thread = console_threads.start_threaded_function(target)

    # def auto_deprecession_clicked(self):
    #     if self.auto_deprecession.selected and (self.precession_thread is None or not self.precession_thread.is_alive()):
    #         angle = self.precession_angle.value()*self.deprecession_auto_factor.value()

    #         if self.main_window.image_view.point_selector.isVisible():
    #             pos = self.main_window.image_view.point_selector.pos()
    #             fov = self.main_window.image_view.image_item.fov
    #             N = self.main_window.image_view.image_item.image.shape[0]
    #             ij = [int(pos.y()/fov*N + 0.5) + N//2, int(pos.x()/fov*N + 0.5) + N//2]

    #         else:
    #             N = 1024
    #             ij = [512, 512]

    #         def set_callback(value):
    #             for i, name in enumerate(M2x2_names):
    #                 self.deprecession_corrections[name].setProperty("busy", True)
    #                 self.deprecession_corrections[name].updateStyle.emit()
    #                 self.deprecession_corrections[name].setReadSentSignal.emit(value[i])

    #         def target():
    #             try:
    #                 precession.optimize_deprecession(init_step_diagonal=angle/2, init_step_non_diagonal=angle/10,
    #                                                  kernel=self._pick_deprecession_kernel(self.deprecession_method.currentText()),
    #                                                  blur='median', N=N, ij=ij, camera_time=self.main_window.camera.exposure.value()*1e-3,
    #                                                  stop_callback=lambda: not self.auto_deprecession.selected, set_callback=set_callback)
    #             except precession.StopException:
    #                 pass

    #             self.auto_deprecession.update_selected_signal.emit(False)

    #             for i, name in enumerate(M2x2_names):
    #                 self.deprecession_corrections[name].setProperty("busy", False)
    #                 self.deprecession_corrections[name].updateStyle.emit()

    #         self.precession_thread = console_threads.start_threaded_function(target)
