from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.app import app
    from expert_pi.extensions.diffraction_tomography.view import MainView
    from expert_pi.gui.main_window import MainWindow

import os
import shutil
import time
import traceback

import numpy as np
import stem_measurements.core
from PySide6 import QtCore, QtWidgets

from expert_pi import grpc_client
from expert_pi.app import data_saver, scan_helper
from expert_pi.app.states import preserved_state_saver
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.automations import xyz_tracking
from expert_pi.automations.xyz_tracking import nodes, objects
from expert_pi.config import get_config
from expert_pi.extensions.diffraction_tomography import helpers
from expert_pi.extensions.diffraction_tomography.series_viewer_controller import SeriesViewerController
from expert_pi.measurements.data_formats.diffraction_tomography import DiffractionTiltSeries
from expert_pi.stream_clients import CacheClient
from expert_pi.stream_processors import process_thread


class MainController(QtCore.QObject):
    regulation_loop_signal = QtCore.Signal(bool)

    def __init__(
        self,
        measurement_view: MainView,
        controller: app.MainApp,
        state: StatesHolder,
        cache_client: CacheClient,
    ):
        super().__init__()
        self.name = "3DED"

        self._window: MainWindow = measurement_view.window
        self._measurement_view = measurement_view
        self._controller = controller
        self.states = state
        self.cache_client = cache_client

        self._acquisition = measurement_view.acquisition
        self._analysis = measurement_view.analysis

        self.series_viewer_controller = SeriesViewerController(
            self._measurement_view,
            self._controller,
            self.states,
            self._measurement_view.stem_view,
            self._measurement_view.stem_histogram,
            self._measurement_view.diffraction_view,
            self._measurement_view.diffraction_histogram,
            self._measurement_view.slider,
        )

        self.thread: process_thread.ProcessingThread | None = None

        self._measurement_view.show_signal.connect(self.show)
        self._measurement_view.hide_signal.connect(self.hide)
        # self._measurement_view.save_signal.connect(self.save) #TODO
        # self._measurement_view.load_signal.connect(self.load) #TODO

        # acquisition
        self._acquisition.skip_angle.set_signal.connect(self.set_skip_angle)
        self._acquisition.reference_button.clicked.connect(self.get_reference)
        self._acquisition.test_button.clicked.connect(self.start_test)
        self._acquisition.acquire_button.clicked.connect(self.start_acquire)
        self._acquisition.pause_button.clicked.connect(self.pause_clicked)
        self.regulation_loop_signal.connect(self.set_regulation_loop)

        # visualization
        self._analysis.layout_type.clicked.connect(self.change_layout_type)
        # self._visualization.channels.clicked.connect(self.change_series_view)
        # self._visualization.layout_type.clicked.connect(self.change_layout_type)
        # self._visualization.histogram_button.clicked.connect(self.show_histograms)

        self.reference_fov_hint: float | None = None  # modified at entry to the measurements
        self.measured_fov = 0.0  # modified from taken reference
        self.measuring_counter = 0
        self.saving_node = None
        self.dt = 0.05
        self.measured_data = []
        self.diagnostic_data = []
        self.diagnostic_running = False

        self.has_image_view = False  # we are sharing image view with survey component
        self.has_diffraction_view = False
        self.has_stem_tools = False
        self.last_stem_tools_option = None

        self.tilt_series: DiffractionTiltSeries | None = None
        self.file_name = None

        self._window.tool_bar.file_saving.save_signal.connect(self.save_file)
        self._window.tool_bar.file_saving.new_signal.connect(self.load_file)
        self._window.tool_bar.file_saving.opened_change_signal.connect(self.close_open_file)

        self._window.stem_tools.visualisation.clicked.connect(self.enable_histogram)

    def synchronise(self):
        grpc_client.illumination.get_condenser_defocus_range(type=grpc_client.illumination.CondenserFocusType.C3)
        self._acquisition.camera_exposure.set_read_sent_value(self._window.camera.exposure.value())

    def hide(self):
        pass

    def show(self):
        self.states.interaction_mode = StatesHolder.InteractionMode.measurements
        self.states.measurement_type = self.name

        self.reference_fov_hint = self._window.scanning.fov_spin.value()
        self._controller.xyz_tracking_controller.update_ref_fov(self.reference_fov_hint)
        self.measured_fov = self._window.scanning.fov_spin.value()  # TODO make this separate
        self._acquisition.measured_fov = self.measured_fov
        self.last_stem_tools_option = self._window.stem_tools.selectors.selected

        self._acquisition.reference_button.setText(f"TAKE REF {self.reference_fov_hint:0.2g} um")

        if xyz_tracking.registration.reference_image is not None:
            self._acquisition.test_button.setEnabled(True)
            self._acquisition.acquire_button.setEnabled(True)

        self._window.tool_bar.file_saving.show(None)
        self._window.tool_bar.file_saving.status_signal.emit(None, True, False)

    def save_file(self, file_name: str, close=False):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.measurements
            and self.states.measurement_type == self.name
            and self.tilt_series is not None
        ):
            self.file_name = file_name
            self.tilt_series.save()
            if file_name != self.tilt_series.filename:
                shutil.copyfile(self.tilt_series.filename, file_name)

            if not close:
                self.tilt_series.load(file_name)

    def load_file(self):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.measurements
            and self.states.measurement_type == self.name
        ):
            last_folder = preserved_state_saver.actual_state["last_save_directory"]
            name, *_ = QtWidgets.QFileDialog.getOpenFileName(
                self._window, "Load File", last_folder + "untitled.h5", "diffraction tomography h5 file (*.h5)"
            )
            if not name:
                raise Exception("no name selected")

            preserved_state_saver.actual_state["last_save_directory"] = os.path.dirname(name) + "/"
            preserved_state_saver.save(get_config().data.data_folder)

            self.file_name = name
            self._window.tool_bar.file_saving.status_signal.emit(self.file_name, True, False)
            if self.tilt_series is not None:
                self.tilt_series.file.close()
            self.tilt_series = DiffractionTiltSeries.open(self.file_name)
            self.series_viewer_controller.set_tilt_series(self.tilt_series)
            self._measurement_view.analysis.layout_type.option_clicked("analysis")

    def create_new_series(self):
        n = int(self._acquisition.pixels.currentText().split(" ")[0])
        diff_4d_pixels = int(self._acquisition.diff_4d_pixels.currentText().split(" ")[0])

        acquisition_method = self._acquisition.diff_4d_method.currentText()

        diffractions_parameters = {}
        diffractions_parameters["acquisition_method"] = acquisition_method

        if acquisition_method == "point" and self.measured_fov is not None:
            pos = self._window.image_view.tools["point_selector"].pos()
            ij = [
                int(pos.y() / self.measured_fov * n + 0.5) + n // 2,
                int(pos.x() / self.measured_fov * n + 0.5) + n // 2,
            ]
            diffractions_parameters["total_pixels"] = n
            diffractions_parameters["rectangle"] = np.array([ij[0], ij[1], 1, 1])

        elif self._window.stem_tools.selectors.selected == "rectangle_selector":
            diffractions_parameters["acquisition_method"] = (
                self._acquisition.diff_4d_method.currentText()
            )  # raster, average, random

            x0, y0, x2, y2 = self._window.image_view.tools["rectangle_selector"].get_rectangle()

            fov = self.measured_fov

            if fov is not None:
                max_pixels = grpc_client.scanning.get_maximum_scan_field_number_of_pixels()
                pixel_factor = min(max_pixels / diff_4d_pixels, fov / (x2 - x0))

                diffractions_parameters["total_pixels"] = int(diff_4d_pixels * pixel_factor)
                diffractions_parameters["rectangle"] = np.array([
                    (y0 + fov / 2) / fov * diff_4d_pixels * pixel_factor,
                    (x0 + fov / 2) / fov * diff_4d_pixels * pixel_factor,
                    (y2 - y0) / fov * diff_4d_pixels * pixel_factor,
                    (x2 - x0) / fov * diff_4d_pixels * pixel_factor,
                ]).astype("int")

        else:
            diffractions_parameters["total_pixels"] = diff_4d_pixels
            diffractions_parameters["rectangle"] = np.array([0, 0, diff_4d_pixels, diff_4d_pixels])

        n_cam_i, n_cam_j = diffractions_parameters["rectangle"][2], diffractions_parameters["rectangle"][3]
        if diffractions_parameters["acquisition_method"] in {"random"}:
            n_cam_i = 1
            n_cam_j = 1

        m = int(self._acquisition.num_measurements.value())

        grpc_client.scanning.get_camera_roi()
        if grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Disabled.name:
            camera_shape = (512, 512)
        elif grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_256.name:
            camera_shape = (256, 512)
        elif grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_128.name:
            camera_shape = (128, 512)

        size = (
            1.0 * m * n_cam_i * n_cam_j * camera_shape[0] * camera_shape[1] * 2 / 1024**3
        )  # assuming 16bit camera images

        filename = None
        if self.tilt_series is not None:
            if self._window.tool_bar.file_saving.filename.text() == self.tilt_series.filename:
                dlg = QtWidgets.QMessageBox(self._window)
                dlg.setWindowTitle("")
                dlg.setText(f"Do you want to overwrite {self.tilt_series.filename} ?")
                dlg.setStandardButtons(
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                dlg.setIcon(QtWidgets.QMessageBox.Icon.Question)
                button = dlg.exec()

                if button == QtWidgets.QMessageBox.StandardButton.Yes:
                    if self.tilt_series.is_open():
                        self.tilt_series.file.close()
                    filename = self.tilt_series.filename
                    os.remove(filename)
        if filename is None:
            last_folder = preserved_state_saver.actual_state["last_save_directory"]
            filename, *arg = QtWidgets.QFileDialog.getSaveFileName(
                self._window, "Save File", last_folder + "untitled.h5", "diffraction tomography h5 file (*.h5)"
            )
            if not filename:
                raise Exception("no name selected")

        preserved_state_saver.actual_state["last_save_directory"] = os.path.dirname(filename) + "/"
        preserved_state_saver.save(get_config().data.data_folder)

        self._window.tool_bar.file_saving.status_signal.emit(filename, True, True)

        exposure = self._acquisition.camera_exposure.value()

        # TODO get this from the camera somehow:
        if exposure * 1e-3 < 1 / 2250:
            dtype = np.uint8
        else:
            dtype = np.uint16
        # TODO uint32

        tilt_series = DiffractionTiltSeries.new(
            filename, m, (n_cam_i, n_cam_j, camera_shape[0], camera_shape[1]), dtype_4d=dtype, dtype=np.uint32
        )

        tilt_series.stem_images["BF"] = np.empty((m, n, n), dtype=np.uint16)

        tilt_series.parameters.update(data_saver.get_microscope_metadata(get_config().connection))

        tilt_series.parameters["detectors"]["camera"] = data_saver.get_camera_metadata(
            exposure,
            precession=self._window.precession.enabled.diffraction.selected,  # exposure ms
            precession_angle=self._window.precession.precession_angle.value(),  # angle mrad
            precession_frequency=self._window.precession.precession_frequency.value(),  # frequency kHz
            wavelength=data_saver.get_wavelength(tilt_series.parameters["beam_energy (keV)"] * 1000),
        )

        tilt_series.parameters["detectors"]["stem"] = data_saver.get_stem_metadata(
            self.measured_fov, self._acquisition.pixel_time.value(), channels="BF"
        )  # TODO acquire without BF image

        tilt_series.parameters["diffractions"] = diffractions_parameters

        # diffractions_parameters["c3_stem_defocus"] = (
        #     grpc_client.illumination.get_condenser_defocus(type=grpc_client.illumination.CondenserFocusType.C3) * 1e6
        # )  # um
        # diffractions_parameters["c3_diffraction_defocus"] = (
        #     tilt_series.parameters["c3_stem_defocus"] + self.c3_defocus.value()
        # )  # um

        tilt_series.angles[:] = (
            np.linspace(-self._acquisition.max_angle.value(), self._acquisition.max_angle.value(), num=m) / 180 * np.pi
        )  # to deg
        tilt_series.parameters["reference_index"] = m // 2

        # check fov:
        fov = tilt_series.parameters["detectors"]["stem"]["fov (um)"]
        pixel_time = tilt_series.parameters["detectors"]["stem"]["pixel_time (us)"]

        fov_range = grpc_client.scanning.get_field_width_range(pixel_time * 1e-6, n)
        if fov_range["end"] * 1e6 < fov:
            raise Exception(f"stem image fov not possible for given pixel time, max {fov_range['end'] * 1e6} um")

        tilt_series.dead_pixels = np.array([[p["y"], p["x"]] for p in grpc_client.scanning.get_dead_pixels()])
        tilt_series.dead_pixels[:, 0] -= (512 - camera_shape[0]) // 2
        mask = (0 < tilt_series.dead_pixels[:, 0]) & (tilt_series.dead_pixels[:, 0] < camera_shape[0])
        tilt_series.dead_pixels = tilt_series.dead_pixels[mask]

        return tilt_series

    def reset_series(self):
        self.tilt_series = self.create_new_series()
        self.series_viewer_controller.set_tilt_series(self.tilt_series)

    def get_reference(self):
        if self._acquisition.reference_button.selected:
            if grpc_client.projection.get_is_off_axis_stem_enabled():  # assuming always off axis detector
                grpc_client.projection.set_is_off_axis_stem_enabled(True)

        if self._acquisition.reference_button.selected:
            self._controller.xyz_tracking_controller.refresh_reference(fov=self.reference_fov_hint)
            self.measured_fov = self.reference_fov_hint
            self._acquisition.measured_fov = self.measured_fov

            self.set_regulation_loop(True)
            self._acquisition.test_button.setEnabled(True)
            self._acquisition.acquire_button.setEnabled(True)
        else:
            self.set_regulation_loop(False)

    def measure_step(self, input: objects.NodeImage):
        # print("measuring", self.measuring_counter, input.id, input.stage_ab[0]/np.pi*180)

        self.series_viewer_controller.update_series_slider_colors(
            self.measuring_counter, acquiring=True, emit_update=True
        )

        # xy_electronic_actual = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
        # dxy = input.offset_xy
        # print("adjusting rezidual shift:", dxy)
        # grpc_client.illumination.set_shift({"x": xy_electronic_actual["x"] - dxy[0]*1e-6, "y": xy_electronic_actual["y"] - dxy[1]*1e-6}, grpc_client.illumination.DeflectorType.Scan)

        tilt_series = self.tilt_series

        fov = tilt_series.parameters["detectors"]["stem"]["fov (um)"]
        grpc_client.scanning.set_field_width(fov * 1e-6)

        total_size, rectangle = self.get_diffraction_patterns(input)
        # nned to update the image in series viewer manually:
        # self.tomography_widget.series_viewer.images["Camera"][self.measuring_counter] = helpers.stack_4d_images(tilt_series.diffractions[self.measuring_counter:self.measuring_counter + 1])[0]

        pixel_time = tilt_series.parameters["detectors"]["stem"]["pixel_time (us)"]

        n = tilt_series.stem_images[tilt_series.channels[0]].shape[1]  # assume square
        scan_id = scan_helper.start_rectangle_scan(pixel_time * 1e-6, n, frames=1, detectors=tilt_series.channels)
        _header, data = self.cache_client.get_item(int(scan_id), n**2)

        rectangle = (n / total_size * np.array(rectangle)).astype("int")  # match the stem image

        for channel in tilt_series.channels:
            image = data["stemData"][channel].reshape(n, n)
            # image[rectangle[0] : rectangle[0] + rectangle[2], rectangle[1] : rectangle[1] + rectangle[3]] //= 2
            tilt_series.stem_images[channel][self.measuring_counter] = image

        tilt_series.transforms2x2[self.measuring_counter] = input.transform2x2 * 1.0

        self.measured_data.append(input)

        if self.series_viewer_controller.auto_index_update:
            self.series_viewer_controller.set_index(self.measuring_counter)
        self.measuring_counter += 1

        self.series_viewer_controller.update_series_slider_colors(
            self.measuring_counter, acquiring=False, emit_update=True
        )

    def get_diffraction_patterns(self, input_):
        tilt_series = self.tilt_series
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
        self._window.scanning.off_axis_butt.update_selected_signal.emit(
            False
        )  # need to be through signal , this function is called from separate thread

        diffraction_parameters = tilt_series.parameters["diffractions"]

        diffraction_pixel_time = tilt_series.parameters["detectors"]["camera"]["exposure (ms)"]  # ms
        method = diffraction_parameters["acquisition_method"]

        size = tilt_series.data5D.shape[1:3]

        # modify rectangle position, assuming alpha in x direction
        rectangle_center = (
            diffraction_parameters["rectangle"][0]
            + diffraction_parameters["rectangle"][2] / 2
            - diffraction_parameters["total_pixels"] / 2
        )
        rectangle = diffraction_parameters["rectangle"] * 1
        rectangle[0] = int(
            np.round(
                rectangle_center * np.cos(input_.stage_ab[0])
                - diffraction_parameters["rectangle"][2] / 2
                + diffraction_parameters["total_pixels"] / 2
            )
        )

        if method == "random":
            i = np.random.randint(rectangle[2])
            j = np.random.randint(rectangle[3])
            rectangle = np.array([i + rectangle[0], j + rectangle[1], 1, 1])

        tilt_series.diffraction_selections[self.measuring_counter, :] = rectangle

        total_size = diffraction_parameters["total_pixels"]

        if "precession angle (mrad)" in tilt_series.parameters["detectors"]["camera"]:
            precession_angle = tilt_series.parameters["detectors"]["camera"]["precession angle (mrad)"]
            precession_frequency = tilt_series.parameters["detectors"]["camera"]["precession frequency (kHz)"]
            grpc_client.scanning.set_precession_angle(
                precession_angle * 1e-3
            )  # we need to specifically call this because of xyz tracking
            grpc_client.scanning.set_precession_frequency(precession_frequency * 1e3)

        grpc_client.scanning.set_image_compression(grpc_client.scanning.Compression.Bslz4)
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=diffraction_pixel_time * 1e-3,
            total_size=total_size,
            rectangle=rectangle,
            frames=1,
            detectors=[grpc_client.scanning.DetectorType.Camera],
            is_precession_enabled=self._window.precession.enabled.diffraction.selected,
        )

        batch_size = 256
        remaining = size[0] * size[1]

        i = 0
        j = 0

        self.tilt_series.diffractions[self.measuring_counter, :, :] = 0

        while remaining > 0:
            to_read = min(remaining, batch_size)
            header, data = self.cache_client.get_item(scan_id, to_read, raw=True)

            if not header["cameraData"]["compressionType"] == 2:
                raise Exception(
                    "data from stream must be already compressed, found type:", header["cameraData"]["compressionType"]
                )

            byte_counter = 0
            for k in range(header["pixelCount"]):
                length = header["cameraData"]["lengths"][k]
                tilt_series.data5D.id.write_direct_chunk(
                    [self.measuring_counter, i, j, 0, 0], data["cameraData"][byte_counter : byte_counter + length]
                )

                byte_counter += length
                j += 1
                if j == header["scanDimensions"][2]:
                    j = 0
                    i += 1

            if header["cameraData"]["bytesPerPixel"] == 1:
                dtype = np.uint8
            elif header["cameraData"]["bytesPerPixel"] == 2:
                dtype = np.uint16
            else:
                dtype = np.uint32

            decompressed_data = np.empty(
                (header["pixelCount"], header["cameraData"]["imageWidth"], header["cameraData"]["imageHeight"]),
                dtype=dtype,
            )
            stem_measurements.core.bitshuffle_decompress_batch(
                data["cameraData"], header["cameraData"]["lengths"], decompressed_data, thread_count=8
            )

            self.tilt_series.diffractions[self.measuring_counter, :, :] += np.sum(decompressed_data, axis=0)

            remaining -= header["pixelCount"]

        # if not self.c3_defocus.value() == 0:
        #     grpc_client.illumination.set_condenser_defocus(tilt_series.parameters["c3_stem_defocus"]*1e-6, type=grpc_client.illumination.CondenserFocusType.C3)

        grpc_client.projection.set_is_off_axis_stem_enabled(True)
        self._window.scanning.off_axis_butt.update_selected_signal.emit(
            True
        )  # need to be through signal , this function is called from separate thread

        return total_size, rectangle

    def change_layout_type(self):
        if self._analysis.layout_type.selected == "acquisition":
            self._measurement_view.show_analysis = False
            self._window.central_layout.set_central_layout(self._measurement_view.central_acquisition_layout)
        else:
            self._measurement_view.show_analysis = True
            self._window.central_layout.set_central_layout(self._measurement_view.central_analysis_layout)

    def show_histograms(self):
        for channel in self._visualization.channels.selected:
            if self._visualization.histogram_button.selected:
                self._measurement_view.series_viewer.viewers[channel].recalculate_histogram()
                self._measurement_view.series_viewer.viewers[channel].histogram.show()
                self._measurement_view.series_viewer.viewers[channel].histogram.redraw()
            else:
                self._measurement_view.series_viewer.viewers[channel].histogram.hide()

    # -------------------------------------------------------
    # TODO: multiple method repeats in stem tomography

    def set_skip_angle(self, value):
        xyz_tracking.settings.stabilization_alpha_skip = value

    def init_saving(self):
        if self.saving_node is None:
            node_manager = xyz_tracking.manager
            register_node = xyz_tracking.manager.get_node_by_name("registration")
            self.saving_node = nodes.Node(self.save_diagnostic, name="saving")
            register_node.output_nodes.append(self.saving_node)  # Watchout for reloading of tomography code
            node_manager.nodes.append(self.saving_node)

    def save_diagnostic(self, input: objects.NodeImage):
        if self.diagnostic_running:
            self.diagnostic_data.append(input)

            alpha = input.stage_ab[0]
            xy = (
                input.stage_xy
                - np.dot(input.transform2x2, input.offset_xy)
                + np.dot(input.transform2x2, input.shift_electronic - input.reference_object.shift_electronic)
                - input.reference_object.stage_xy
            )
            z = input.stage_z - input.offset_z - input.reference_object.stage_z
            self._measurement_view.plot_widget.plot_lines["base"].addData(alpha, 0)
            self._measurement_view.plot_widget.plot_lines["x"].addData(alpha, xy[0])
            self._measurement_view.plot_widget.plot_lines["y"].addData(alpha, xy[1])
            self._measurement_view.plot_widget.plot_lines["z"].addData(alpha, z)

            self._measurement_view.plot_widget.update_signal.emit()

    def start_test(self):
        self._acquisition.reference_button.set_selected(False)
        self.init_saving()

        if self._acquisition.test_button.selected and (self.thread is None or not self.thread.is_alive()):
            self._measurement_view.plot_widget.clear_data(keep_model=True)
            self.diagnostic_running = True
            self.set_regulation_loop(True)
            xyz_tracking.settings.alpha_speed = self._acquisition.alpha_speed.value()
            self._acquisition.pause_button.setEnabled(True)
            self.thread = process_thread.ProcessingThread(target=self._run_open_cycle)
            self.thread.start()
        else:
            if self.thread is not None:
                self.thread.stop()
            self.diagnostic_running = False
            self._acquisition.acquire_button.set_selected(False)
            self._acquisition.test_button.set_selected(False)
            self._acquisition.pause_button.set_selected(False)
            self.set_regulation_loop(False)
            xyz_tracking.stage_shifting.change_motion_type(xyz_tracking.objects.MotionType.moving)
            self._controller.xyz_tracking_controller.goto_reference(True)

    def start_acquire(self):
        self._acquisition.reference_button.set_selected(False)

        self.init_saving()

        if self._acquisition.acquire_button.selected and (self.thread is None or not self.thread.is_alive()):
            try:
                self._acquisition.acquire_button.set_error(False)
                self._measurement_view.plot_widget.clear_data(keep_model=True)
                self.diagnostic_running = True
                self.series_viewer_controller.auto_index_update = True
                self.reset_series()
                self.set_regulation_loop(True)

                xyz_tracking.settings.alpha_speed = self._acquisition.alpha_speed.value()
                self._acquisition.pause_button.setEnabled(True)
                self._acquisition.stability_condition_changed()
                self.set_skip_angle(self._acquisition.skip_angle.value())
                self.thread = process_thread.ProcessingThread(target=self._run_measure_positions)
                # self._measurement_view.measuring_time = 0
                # self._measurement_view.moving_time = 0
                self.thread.start()
            except:
                traceback.print_exc()
                self._acquisition.acquire_button.set_error(True)

        else:
            self._acquisition.acquire_button.set_error(False)
            if self.thread is not None and self.thread.is_alive:
                self.thread.stop()
            self.diagnostic_running = False
            self._acquisition.acquire_button.set_selected(False)
            self._acquisition.test_button.set_selected(False)
            self._acquisition.pause_button.set_selected(False)
            self.set_regulation_loop(False)
            xyz_tracking.stage_shifting.change_motion_type(xyz_tracking.objects.MotionType.moving)
            self._controller.xyz_tracking_controller.goto_reference(True)

    def pause_clicked(self):
        if self.thread is not None:
            self.thread.paused = self._acquisition.pause_button.selected

    def set_regulation_loop(self, enabled=True):
        self._window.stem_tools.xyz_tracking_widget.start_tracking_button.set_selected(enabled)
        self._controller.xyz_tracking_controller.start_tracking(enabled)

        if not self._window.detectors.BF_insert.selected and not self._window.detectors.HAADF_insert.selected:
            if not grpc_client.projection.get_is_off_axis_stem_enabled():
                grpc_client.projection.set_is_off_axis_stem_enabled(True)
                self._window.scanning.off_axis_butt.set_selected(True)

        if enabled:
            if self._window.stem_tools.selectors.selected != "xyz_tracking":
                self._window.stem_tools.selectors.option_clicked("xyz_tracking", emit=True)
            self._window.stem_tools.xyz_tracking_widget.regulate_button.set_selected(enabled)
            self._controller.xyz_tracking_controller.enable_regulation(enabled)
        else:
            if self.last_stem_tools_option is None or not self.last_stem_tools_option:
                self._window.stem_tools.selectors.option_clicked("xyz_tracking", emit=True)
            elif self._window.stem_tools.selectors.selected != self.last_stem_tools_option:
                self._window.stem_tools.selectors.option_clicked(self.last_stem_tools_option, emit=True)
            self._controller.xyz_tracking_controller.show()

    def _run_open_cycle(self):
        test_time = 2 * np.pi * self._acquisition.max_angle.value() / xyz_tracking.settings.alpha_speed

        self.diagnostic_data = []
        xyz_tracking.alpha_scheduler.drive_open_cycle(
            self._acquisition.max_angle.value(),
            test_time,
            dt=self.dt,
            repeats=1,
            is_running_callback=self._running_callback,
            finish_callback=self._finish_callback,
            correction_model=self._measurement_view.plot_widget.precalculate_correction_model,
        )

    def _run_measure_positions(self):
        xyz_tracking.acquisition.measurement_acquisition = self.measure_step  # inject our measurement
        self._acquisition.stability_condition_changed()
        self.measuring_counter = 0

        self.measured_data = []
        self.diagnostic_data = []

        xyz_tracking.alpha_scheduler.measure_positions(
            self.tilt_series.angles[:] / np.pi * 180,
            is_running_callback=self._running_callback,
            finish_callback=self._finish_callback,
            correction_model=self._measurement_view.plot_widget.precalculate_correction_model,
        )

    def _running_callback(self, alpha, i, n):
        r = self._measurement_view.slider.range
        self._measurement_view.slider.slider.indicator_position = (alpha - r[0]) / (r[1] - r[0]) * (r[2] - 1)
        self._measurement_view.slider.slider.update_signal.emit()

        t = time.monotonic()

        dt = t - self.thread.start_time
        total = 0
        if i > 0 and dt > 0:
            dt / i * (n - i)
            total = dt / i * n
            # test_button_text = f"TEST ({t_remaining:.1f}s)"
            # self.parameters.test_button.update_text_signal.emit(test_button_text)

        self._acquisition.info.update_text_signal.emit(
            f"{alpha:+5.1f}° {i}/{n}  {helpers.seconds_to_time(dt)}/{helpers.seconds_to_time(total)}"
        )

        return self.thread.running, self.thread.paused

    def _finish_callback(self):
        self.thread.stop()
        self._acquisition.test_button.update_selected_signal.emit(False)
        self._acquisition.acquire_button.update_selected_signal.emit(False)
        self.regulation_loop_signal.emit(False)
        if self.tilt_series is not None:
            filename = self.tilt_series.filename
            self.tilt_series.save()
            self.tilt_series.load(filename)

    def close_open_file(self, value):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.measurements
            and self.states.measurement_type == self.name
        ):
            if not value:
                self.save_file(self.file_name, True)
            elif not self.tilt_series.is_open():
                self.tilt_series.load(self.file_name)

    def enable_histogram(self, name):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.measurements
            and self.states.measurement_type == self.name
        ):
            if "histogram" in self._window.stem_tools.visualisation.selected:
                self.series_viewer_controller.image_normalizer.histogram_enable = True
                self.series_viewer_controller.diffraction_normalizer.histogram_enable = True
                self._measurement_view.stem_histogram.show()
                self._measurement_view.diffraction_histogram.show()
                self.series_viewer_controller.set_index(self.series_viewer_controller.current_index)
            else:
                self.series_viewer_controller.image_normalizer.histogram_enable = False
                self.series_viewer_controller.diffraction_normalizer.histogram_enable = False
                self._measurement_view.stem_histogram.hide()
                self._measurement_view.diffraction_histogram.hide()
