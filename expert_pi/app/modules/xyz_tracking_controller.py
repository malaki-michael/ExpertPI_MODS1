import time
from functools import lru_cache

import numpy as np
from PySide6 import QtCore

from expert_pi import grpc_client
from expert_pi.automations import xyz_tracking
from expert_pi.automations.xyz_tracking import objects
from expert_pi.app.modules import acquisition_controller
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.gui.data_views import image_item
from expert_pi.gui.main_window import MainWindow
from expert_pi.gui.style import coloring
from expert_pi.stream_processors import functions


class DebugConnector(QtCore.QObject):
    set_input_signal = QtCore.Signal(object)
    set_output_signal = QtCore.Signal(object, object)
    set_error_signal = QtCore.Signal()

    def __init__(self, input_process, output_process, error_process=lambda: None):
        super().__init__()
        self.set_input_signal.connect(input_process)
        self.set_output_signal.connect(output_process)
        self.set_error_signal.connect(error_process)


class XYZTrackingController(QtCore.QObject):
    redraw_reference_signal = QtCore.Signal()

    def __init__(self, window: MainWindow, states: StatesHolder):
        super().__init__()
        self._window = window
        self._states = states
        self._tracking_widget = window.stem_tools.xyz_tracking_widget
        self._reference_multi_image = self._window.image_view.tools["xyz_tracking"]

        self.redraw_reference_signal.connect(self.redraw_reference)

        self._tracking_widget.show_signal.connect(self.show)
        self._tracking_widget.hide_signal.connect(self.hide)

        self._tracking_widget.start_tracking_button.clicked.connect(self.start_tracking)
        self._tracking_widget.regulate_button.clicked.connect(
            lambda: self.enable_regulation(self._tracking_widget.regulate_button.selected)
        )
        self._tracking_widget.refresh_reference_button.clicked.connect(lambda x: self.refresh_reference())
        self._tracking_widget.customize_button.clicked.connect(self.customize_reference_region)
        self._tracking_widget.goto_button.clicked.connect(lambda: self.goto_reference(False))
        self._tracking_widget.goto_ab_button.clicked.connect(lambda: self.goto_reference(True))
        self._tracking_widget.debug_button.clicked.connect(
            lambda: (
                self._window.stem_tools.tracking_debugger.show()
                if self._tracking_widget.debug_button.selected
                else self._window.stem_tools.tracking_debugger.hide()
            )
        )
        self._tracking_widget.actual_fov.currentIndexChanged.connect(self.manual_fov_changed)
        window.scanning.fov_spin.set_signal.connect(self.update_ref_fov)  # TODO: not here

        xyz_tracking.manager.get_node_by_name("image_viewing").job = self.image_view_callback
        self.stage_shifting_connector = DebugConnector(lambda: None, self.process_stage_shifting)
        xyz_tracking.manager.get_node_by_name("stage_shift").debug_trackers.append(self.stage_shifting_connector)
        self.regulation_connector = DebugConnector(lambda: None, self.process_regulation)
        xyz_tracking.manager.get_node_by_name("regulation").debug_trackers.append(self.regulation_connector)

        self.fov = 0
        self.cached_max_fov = None
        self.original_preview_image = None

        self.image_alpha = 0.7
        self.ref_tint = [0.8, 0.9, 0.8]
        # self.image_items = []

    def show(self):
        self._window.image_view.main_image_item.hide()
        self._window.image_view.tools["xyz_tracking"].show()
        self.synchronize()

    def hide(self):
        acquisition_controller.rescan_stem(self._window, self._states)
        self._window.image_view.main_image_item.show()
        self._window.image_view.tools["xyz_tracking"].hide()

    def synchronize(self):
        self.update_ref_fov(grpc_client.scanning.get_field_width() * 1e6)

    def update_ref_fov(self, fov):
        self.fov = fov
        self._tracking_widget.refresh_reference_button.setText(f"{fov:0.2g} um")

    def start_tracking(self, enabled):
        if enabled:
            if not self._window.detectors.BF_insert.selected and not self._window.detectors.HAADF_insert.selected:
                if not grpc_client.projection.get_is_off_axis_stem_enabled():
                    grpc_client.projection.set_is_off_axis_stem_enabled(True)
                    self._window.scanning.off_axis_butt.set_selected(True)
            elif grpc_client.projection.get_is_off_axis_stem_enabled():
                grpc_client.projection.set_is_off_axis_stem_enabled(False)
                self._window.scanning.off_axis_butt.set_selected(False)

            self._states.image_view_channel = "xyz_tracker"

            if not self._window.image_view.tools["xyz_tracking"].isVisible():
                self.show_reference_multi_image(True)

            if xyz_tracking.registration.reference_image is None:
                self.refresh_reference()
            xyz_tracking.stage_shifting.manual_fov = None
            self._reference_multi_image.aligned_image.show()
            xyz_tracking.start_tracking(self.actual_fov_values[self._tracking_widget.actual_fov.currentIndex()])
            self._window.stem_tools.tracking_debugger.reset_time(time.perf_counter())
        else:
            xyz_tracking.stop_tracking()
            self._states.image_view_channel = self._window.stem_tools.image_type.selected
            self._tracking_widget.start_tracking_button.setText("")

    def get_reference_fovs(self, preferred_fov, fov_max=None):
        fov_max_ref = self.get_max_fov(
            xyz_tracking.settings.pixel_time, xyz_tracking.settings.reference_N, xyz_tracking.settings.tracking_N
        )

        if fov_max is not None:
            fov_max_ref = min(fov_max, fov_max_ref)

        if preferred_fov > fov_max_ref:
            preferred_fov = fov_max_ref * 0.99

        fovs = []
        fov_add = fov_max_ref
        while fov_add > 1.5 * preferred_fov:
            fovs.append(fov_add)
            fov_add /= 2
        fovs.append(preferred_fov)
        return preferred_fov, fovs

    @lru_cache(maxsize=10)
    def get_max_fov(self, pixel_time, reference_N, tracking_N):
        fov_max_ref = grpc_client.scanning.get_field_width_range(pixel_time * 1e-9, reference_N)["end"] * 1e6
        fov_max_track = grpc_client.scanning.get_field_width_range(pixel_time * 1e-9, tracking_N)["end"] * 1e6
        min_fov = min(fov_max_ref, fov_max_track)

        return min_fov

    def refresh_reference(self, fov=None):
        self.original_preview_image = (
            self._window.image_view.main_image_item.fov,
            self._window.image_view.normalizer.raw_image * 1,
        )

        self._reference_multi_image.clear()

        # off axis detector change
        if not self._window.detectors.BF_insert.selected and not self._window.detectors.HAADF_insert.selected:
            if not grpc_client.projection.get_is_off_axis_stem_enabled():
                grpc_client.projection.set_is_off_axis_stem_enabled(True)
                self._window.scanning.off_axis_butt.set_selected(True)
        elif grpc_client.projection.get_is_off_axis_stem_enabled():
            grpc_client.projection.set_is_off_axis_stem_enabled(False)
            self._window.scanning.off_axis_butt.set_selected(False)

        restart = False
        if self._tracking_widget.start_tracking_button.selected:
            xyz_tracking.stop_tracking(self._window.image_view)
            restart = True

        if self._tracking_widget.customize_button.selected:
            self._tracking_widget.customize_button.set_selected(False)
            self._window.image_view.tools["reference_picker"].hide()
            self._window.image_view.tools["xyz_tracking"].show()
            self._window.image_view.main_image_item.hide()

            (
                fovs_total,
                total_pixels,
                rectangles,
                fovs,
                centers,
                fov_rectangles,
            ) = self._window.image_view.tools["reference_picker"].get_fovs_rectangles()

            self.fov = fovs[-1]

            channel = self._states.image_view_channel
            self._states.image_view_channel = "xyz_tracking"
            ref_image = xyz_tracking.acquire_reference(
                self.fov,
                None,  # self._window.image_view,
                fovs=fovs,
                fovs_total=fovs_total,
                total_pixels=total_pixels,
                rectangles=rectangles,
                offsets=centers,
            )

            self._states.image_view_channel = channel
            xyz_tracking.settings.use_rectangle_selection = True
            xyz_tracking.settings.rectangles = rectangles
            xyz_tracking.settings.total_pixels = total_pixels
            xyz_tracking.settings.fovs_total = fovs_total
            xyz_tracking.settings.centers = centers

        else:
            if fov is None:
                fov = self.fov
            fov, fovs = self.get_reference_fovs(fov, fov_max=None)
            self.update_ref_fov(fov)  # will set self.fov

            ref_image = xyz_tracking.acquire_reference(self.fov, None, fovs=fovs)

            xyz_tracking.settings.use_rectangle_selection = False
            centers = None

        xyz_tracking.stage_shifting.manual_xyz_correction = np.zeros(3)

        self._tracking_widget.actual_fov.clear()
        self.actual_fov_values = []
        for fov in ref_image.images.keys():
            self.actual_fov_values.append(fov)
            self._tracking_widget.actual_fov.addItem(f"{fov:.2g} um")
        self._tracking_widget.actual_fov.setCurrentIndex(len(self.actual_fov_values) - 1)
        self._reference_multi_image.set_reference_images(ref_image, positions=centers)

        if restart:
            xyz_tracking.start_tracking(self.fov)
        else:
            self._reference_multi_image.aligned_image.hide()

    def enable_regulation(self, value):
        xyz_tracking.stage_shifting.manual_fov = None
        xyz_tracking.settings.enable_regulation = value

    def goto_reference(self, set_a_b=True):
        ref = xyz_tracking.registration.reference_image
        xyz_tracking.stage_shifting.manual_xyz_correction = np.zeros(3)
        if ref is not None:
            if set_a_b:
                grpc_client.stage.set_x_y_z_a_b(
                    ref.stage_xy[0] * 1e-6, ref.stage_xy[1] * 1e-6, ref.stage_z * 1e-6, ref.stage_ab[0], ref.stage_ab[1]
                )
            else:
                grpc_client.stage.set_x_y_z(ref.stage_xy[0] * 1e-6, ref.stage_xy[1] * 1e-6, ref.stage_z * 1e-6)
            grpc_client.illumination.set_shift(
                {"x": ref.shift_electronic[0] * 1e-6, "y": ref.shift_electronic[1] * 1e-6},
                grpc_client.illumination.DeflectorType.Scan,
            )

    def manual_override(self, shift_electronic, stage_xy, rotation, transform2x2):
        # reset integration term of the regulation
        xyz_tracking.regulation.actual_errors_integral = None
        stage_z = grpc_client.stage.get_z() * 1e6  # TODO add electronic shift

        xy_e = np.dot(transform2x2, shift_electronic)
        xyz_tracking.stage_shifting.manual_xyz_correction = np.concatenate([
            stage_xy + xy_e - xyz_tracking.registration.reference_image.stage_xy,
            [stage_z - xyz_tracking.registration.reference_image.stage_z],
        ])

    def manual_fov_changed(self, i):
        if i < len(self.actual_fov_values):
            xyz_tracking.stage_shifting.manual_fov = self.actual_fov_values[i]

    def image_view_callback(self, input: objects.NodeImage):
        red_factor = 1  # means completely red
        if np.all(input.confidence_xy > 0):
            red_factor = np.clip(np.mean(0.5 - input.confidence_xy) / 2, 0, 1)

        fov = input.fov * input.rectangle[2] / input.reference_N

        self.set_aligned_image(fov, input.image, input.offset_xy, red_factor, input.transform2x2, position=input.center)
        self.redraw_reference_signal.emit()

    def redraw_reference(self):
        self._reference_multi_image.redraw()
        self._window.image_view.redraw()

    def process_regulation(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        # this is already in main ui thread
        regulated_xyz_shift = output.regulated_xyz_shift
        if regulated_xyz_shift is not None:
            for i, name in enumerate(self._tracking_widget.axes):
                self._tracking_widget.axes_labels[name].setText(f"{np.round(regulated_xyz_shift[i], 3):0.2g}")

    def process_stage_shifting(self, statistics: objects.Diagnostic, output: objects.NodeImage):
        # this is already in main ui thread
        text = output.motion_type.name
        if output.motion_type_timer is not None and output.motion_type != objects.MotionType.moving:
            text += f" ({int(time.monotonic() - output.motion_type_timer + 0.5)})"
        self._tracking_widget.start_tracking_button.setText(text)
        next_fov = output.next_fov
        if next_fov is not None:
            for i, fov in enumerate(self.actual_fov_values):  # actual fov consist of just several elements
                if next_fov >= fov:
                    self._tracking_widget.actual_fov.blockSignals(True)
                    self._tracking_widget.actual_fov.setCurrentIndex(i)
                    self._tracking_widget.actual_fov.blockSignals(False)
                    break

    def customize_reference_region(self):
        if self._tracking_widget.customize_button.selected:
            self._window.image_view.tools["reference_picker"].reset_wizard()
            self._window.image_view.tools["reference_picker"].show()
            fov_max = self.get_max_fov(
                xyz_tracking.settings.pixel_time, xyz_tracking.settings.reference_N, xyz_tracking.settings.tracking_N
            )
            self._window.image_view.tools["reference_picker"].fov_max = fov_max
            self._window.image_view.tools["reference_picker"].N = xyz_tracking.settings.reference_N
            self.show_reference_multi_image(False)
            self._tracking_widget.refresh_reference_button.setText("custom")
        else:
            self._window.image_view.tools["reference_picker"].hide()
            self.show_reference_multi_image(True)
            self.synchronize()

    def show_reference_multi_image(self, enabled):
        if enabled:
            self._window.image_view.tools["xyz_tracking"].show()
            self._window.image_view.main_image_item.hide()
        else:
            self._window.image_view.tools["xyz_tracking"].hide()

            if not self._window.image_view.main_image_item.isVisible():
                # exit from xyz tracking
                grpc_client.scanning.set_field_width(self._window.scanning.fov_spin.value() * 1e-6)
                self._window.image_view.main_image_item.set_fov(self._window.scanning.fov_spin.value())
                acquisition_controller.rescan_stem(self._window, self._states)
            self._window.image_view.main_image_item.show()

    # ReferenceMultiImage
    def clear(self):
        while len(self._reference_multi_image.image_items) > 0:
            self._reference_multi_image.image_items[0].setParentItem(None)
            self._window.image_view.scene().removeItem(self._reference_multi_image.image_items[0])
            del self._reference_multi_image.image_items[0]
        self._reference_multi_image.image_items = []

    def set_aligned_image(self, fov, image, offset, red_factor, transform2x2, position=None):
        if len(image.shape) == 3:
            image_8b = np.zeros((image.shape[1], image.shape[2], 4), dtype="uint8")
            image_8b[:, :, :3] = coloring.get_colored_differences(image[0, :, :], image[1, :, :])
            image_8b[:, :, 3] = 255 * self.image_alpha
        else:
            image_8b = functions.calc_cast_numba(
                image.ravel(), self._window.image_view.alpha, self._window.image_view.beta
            )
            image_8b = image_8b.reshape(image.shape)

            zeros = np.zeros(image_8b.shape, dtype="uint8")

            image_8b = np.dstack([
                image_8b,
                image_8b * (1 - red_factor),
                image_8b * (1 - red_factor),
                zeros + 255 * self.image_alpha,
            ]).astype("uint8")

        self._reference_multi_image.aligned_image.set_image(image_8b)
        self._reference_multi_image.aligned_image.fov = fov
        self._reference_multi_image.aligned_image.shift = [offset[0], -offset[1]]
        if position is not None:
            self._reference_multi_image.aligned_image.shift[0] += position[0]
            self._reference_multi_image.aligned_image.shift[1] += position[1]

        self._reference_multi_image.update_transforms(0, transform2x2)  # TODO rotation

    def set_reference_images(self, reference_images_object, positions=None):
        self.reference_images_object = reference_images_object
        self.clear()
        for i, (fov, image) in enumerate(reference_images_object.images.items()):
            image_8b = functions.calc_cast_numba(
                image.ravel(), self._window.image_view.alpha, self._window.image_view.beta
            )
            image_8b = image_8b.reshape(image.shape)
            image_8b = np.dstack([
                image_8b * self.ref_tint[0],
                image_8b * self.ref_tint[1],
                image_8b * self.ref_tint[2],
            ]).astype("uint8")

            fov_real = (
                reference_images_object.fovs_total[i]
                / reference_images_object.total_pixels[i]
                * reference_images_object.rectangles[i][2]
            )

            self._reference_multi_image.image_items.append(image_item.ImageItem(image_8b, fov_real))
            if positions is not None:
                self._reference_multi_image.image_items[-1].set_shift(
                    reference_images_object.offsets[i][0], reference_images_object.offsets[i][1]
                )
            self._reference_multi_image.image_items[-1].update_transforms()
            self._reference_multi_image.image_items[-1].setZValue(-1)
            self._reference_multi_image.image_items[-1].setParentItem(self)
