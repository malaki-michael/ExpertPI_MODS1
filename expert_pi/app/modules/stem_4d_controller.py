import os
import shutil
import threading
import time

import numpy as np
import stem_measurements
from PySide6 import QtWidgets

from expert_pi import grpc_client
from expert_pi.app import data_saver
from expert_pi.app.modules import acquisition_controller  # watchout recursive import
from expert_pi.app.states import preserved_state_saver
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.config import get_config
from expert_pi.gui.main_window import MainWindow
from expert_pi.measurements.data_formats import stem4d
from expert_pi.stream_clients import CacheClient
from expert_pi.stream_processors import cache_processor, fft_processing, normalizer, stem_4d_processing


class Stem4DController:
    def __init__(
        self,
        window: MainWindow,
        states: StatesHolder,
        image_normalizer: normalizer.Normalizer,
        diffraction_normalizer: normalizer.Normalizer,
        fft_processor: fft_processing.FFTProcessor,
        cache_client: CacheClient,
    ) -> None:
        self.window: MainWindow = window
        self.states = states

        self.image_normalizer = image_normalizer
        self.diffraction_normalizer = diffraction_normalizer
        self.fft_processor = fft_processor

        self._scanning = window.scanning
        self._stem_4d = window.stem_4d

        self.file_saver = stem_4d_processing.FileSaver4DSTEM()
        self.virtual_stem_processor = stem_4d_processing.Processor4DSTEMLive()
        self.virtual_stem_processor.set_masks({"BF": None})  # TODO
        self.virtual_stem_processor.set_consumers(
            [self.update_virtual_image], info_consumers=[self.update_acquisition_info]
        )
        self.receiver = cache_processor.Receiver(
            cache_client, [self.virtual_stem_processor, self.file_saver]
        )  # add file saver

        self.virtual_stem_processor_offline = stem_4d_processing.Processor4DSTEMFile()
        self.virtual_stem_processor_offline.set_consumers(
            [self.update_virtual_image], info_consumers=[self.update_analysis_info]
        )

        self.start_tm = 0

        self.acquiring = False
        self.analysing = False
        self.dead_neighbours = None

        self.last_position = [0, 0]

        self.file_4d: stem4d.STEM4D | None = None
        self.filename = os.path.abspath(get_config().data.stem_4d_cache)
        self.file_access_lock = threading.Lock()
        self.update_diffraction_lock = threading.Lock()

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self.window = window
        self._scanning = window.scanning
        self._stem_4d = window.stem_4d

        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self._stem_4d.pixel_time_spin.set_signal: lambda _: self.update_info(),
            self._stem_4d.size_combo.currentIndexChanged: lambda _: self.update_info(),
            self._stem_4d.acquire_button.clicked: self.acquisition_button_clicked,
            self._stem_4d.progress_button.clicked: self.progress_button_clicked,
            self.window.image_view.point_selection_signal: self.update_diffraction,
            self.window.diffraction_view.mask_selector_changed: self.analyze_start,
            self.window.diffraction_tools.image_type.clicked: self.set_diffraction_channel,
            # histogram update functions:
            self.window.stem_tools.image_type.clicked: self.set_stem_channel,
            self.window.image_view.selection_changed_signal: self.update_rect_diffraction,
            self.window.stem_tools.visualisation.clicked: (
                lambda name: self.stem_emit() if name == "histogram" else None
            ),
            self.window.diffraction_tools.visualisation.clicked: (
                lambda name: self.camera_emit() if name == "histogram" else None
            ),
            self.window.image_histogram.histogram_changed: lambda channel, alpha, beta: self.stem_emit(),
            self.window.diffraction_histogram.histogram_changed: lambda channel, alpha, beta: self.camera_emit(),
            self.window.tool_bar.file_saving.save_signal: self.save_file,
            self.window.tool_bar.file_saving.new_signal: self.new_file,
            self.window.tool_bar.file_saving.opened_change_signal: self.close_open_file,
        }
        return signals

    def generate_start_info(self):
        n = int(self._stem_4d.size_combo.currentText().split(" ")[0])
        acquisition_time = (
            max(1 / grpc_client.scanning.get_camera_max_fps(), self._stem_4d.pixel_time_spin.value() * 1e-3) * n**2
        )  # TODO rectangle selection
        if acquisition_time < 10:
            acquisition_time_text = f"{acquisition_time:.1f} s"
        elif acquisition_time < 60:
            acquisition_time_text = f"{int(np.round(acquisition_time))} s"
        else:
            minutes = int(acquisition_time) // 60
            seconds = int(acquisition_time) % 60
            acquisition_time_text = f"{minutes}:{seconds:02d}"

        # TODO ROI:
        size = n**2 * 512**2 / 1024**3
        if self._stem_4d.pixel_time_spin.value() * 1e-3 < 1 / 2250:
            size *= 2
        elif self._stem_4d.pixel_time_spin.value() * 1e-3 < 1 / 1125:  # TODO where is exact limit?
            size *= 4

        if size < 1:
            size_text = f"{int(size * 1000)} MB"
        else:
            size_text = f"{size:.1f} GB"

        return f"Acquire ({acquisition_time_text}, <{size_text}) "

    def update_info(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            text = self.generate_start_info()
            self.window.stem_4d.acquire_button.names[1] = text
            self.window.stem_4d.acquire_button.setText(text)

    def acquisition_button_clicked(self):
        if self._stem_4d.acquire_button.selected:
            # ENTER 4DSTEM
            self.states.interaction_mode = StatesHolder.InteractionMode.stem_4d
            self.window.tool_bar.file_saving.show(self.filename)
            self.acquisition_start()
            self._stem_4d.progress_button.set_progress(0, "Acquiring")

            self.window.stem_4d.progress_button.setEnabled(True)

            alpha, beta, amplify = self.window.image_histogram.set_channel(self.states.image_view_channel + "_4DSTEM")
            self.image_normalizer.set_alpha_beta(alpha, beta, amplify)
            alpha, beta, amplify = self.window.diffraction_histogram.set_channel(
                self.states.diffraction_view_channel + "_4DSTEM"
            )
            self.diffraction_normalizer.set_alpha_beta(alpha, beta, amplify)

        else:
            # EXIT 4DSTEM
            if self.analysing:
                self.virtual_stem_processor_offline.stop()
                self.analysing = False
            if self.acquiring:
                acquisition_controller.stop_stem(self.window, self.states)
                self.acquiring = False
                self.virtual_stem_processor.stop()
                self.file_saver.stop()
                self.receiver.stop()

            alpha, beta, amplify = self.window.image_histogram.set_channel(self.states.image_view_channel)
            self.image_normalizer.set_alpha_beta(alpha, beta, amplify)
            alpha, beta, amplify = self.window.diffraction_histogram.set_channel(self.states.diffraction_view_channel)
            self.diffraction_normalizer.set_alpha_beta(alpha, beta, amplify)

            self._stem_4d.progress_button.set_progress(0, "previous data")
            self.update_info()
            self.states.interaction_mode = StatesHolder.InteractionMode.survey
            self.window.tool_bar.file_saving.hide()

            self.window.image_view.request_update_signal.emit()

    def progress_button_clicked(self):
        if self.acquiring:
            acquisition_controller.stop_stem(self.window, self.states)
            self.receiver.stop()
        elif self.analysing:
            self.virtual_stem_processor_offline.stop()
        elif self.states.interaction_mode != StatesHolder.InteractionMode.stem_4d and self.file_4d is not None:
            self._stem_4d.acquire_button.set_selected(True)
            self.states.interaction_mode = StatesHolder.InteractionMode.stem_4d
            self.window.tool_bar.file_saving.show(self.filename)
            self._stem_4d.progress_button.set_progress(0, "reanalyze")
            self.update_virtual_image(self.file_4d.virtual_images)
            alpha, beta, amplify = self.window.diffraction_histogram.set_channel(
                self.states.diffraction_view_channel + "_4DSTEM"
            )
            self.diffraction_normalizer.set_alpha_beta(alpha, beta, amplify)
        else:
            self.analyze_start()

    def acquisition_start(self):
        grpc_client.scanning.set_image_compression(grpc_client.scanning.Compression.Bslz4)
        scan_id, n, _rectangle, pixel_time = acquisition_controller.start_4dstem(self.window, self.states, frames=1)

        camera_bit_depth = grpc_client.scanning.get_camera_bith_depth()

        if n < 64:
            self.receiver.batch_size = 128
        elif n < 128:
            self.receiver.batch_size = 256
        elif n < 256:
            self.receiver.batch_size = 1024
        else:
            self.receiver.batch_size = 4096

        if camera_bit_depth == 8:
            dtype = np.uint8
            self.window.diffraction_histogram.amplify["camera_4DSTEM"] = 8
        elif camera_bit_depth == 16:
            dtype = np.uint16
            self.window.diffraction_histogram.amplify["camera_4DSTEM"] = 1
        elif camera_bit_depth == 32:
            dtype = np.uint32
            self.window.diffraction_histogram.amplify["camera_4DSTEM"] = 1  # TODO
        else:
            raise Exception("unsupported camera bit depth")

        if grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Disabled.name:
            camera_shape = (512, 512)
        elif grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_256.name:
            camera_shape = (256, 512)
        elif grpc_client.scanning.get_camera_roi()["roi_mode"].name == grpc_client.scanning.RoiMode.Lines_128.name:
            camera_shape = (128, 512)
        else:
            raise Exception("unsupported camera ROI")

        if self.file_4d is not None:
            self.file_4d.close()

        self.file_4d = stem4d.STEM4D.new(self.filename, (n, n, camera_shape[0], camera_shape[1]), dtype)

        self.file_4d.parameters.update(data_saver.get_microscope_metadata(get_config().connection))
        self.file_4d.parameters["detectors"]["camera"] = data_saver.get_camera_metadata(
            pixel_time,
            precession=self.window.precession.enabled.diffraction.selected,  # exposure ms
            precession_angle=self.window.precession.precession_angle.value(),  # angle mrad
            precession_frequency=self.window.precession.precession_frequency.value(),  # frequency kHz
        )

        self.file_4d.parameters["scanning"] = {
            "fov (um)": grpc_client.scanning.get_field_width() * 1e6,
            "pixel_time (us)": pixel_time,
        }

        self.file_4d.dead_pixels = np.array([[p["y"], p["x"]] for p in grpc_client.scanning.get_dead_pixels()])

        self.file_4d.dead_pixels[:, 0] -= (512 - camera_shape[0]) // 2
        mask = (0 < self.file_4d.dead_pixels[:, 0]) & (self.file_4d.dead_pixels[:, 0] < camera_shape[0])
        self.file_4d.dead_pixels = self.file_4d.dead_pixels[mask]
        self.dead_neighbours = None

        self.window.tool_bar.file_saving.status_signal.emit(self.file_4d.filename, True, True)

        self.receiver.start(scan_id, n**2)  # TODO image ROI

        self.window.diffraction_view.tools["mask_selector"].shape = camera_shape
        masks = self.window.diffraction_view.tools["mask_selector"].generate_masks()

        self.virtual_stem_processor.set_masks(masks)

        self.virtual_stem_processor.start((n, n, camera_shape[0], camera_shape[1]), scan_id, self.file_4d)  # camera ROI
        self.file_saver.start(n**2, scan_id, self.file_4d)
        self.acquiring = True

    def analyze_start(self):
        if (
            self.states.interaction_mode != StatesHolder.InteractionMode.stem_4d
            or self.acquiring
            or self.file_4d is None
        ):
            return
        masks = self.window.diffraction_view.tools["mask_selector"].generate_masks()
        self.virtual_stem_processor_offline.analyze(masks, self.file_4d, self.file_access_lock)
        self.window.tool_bar.file_saving.status_signal.emit(self.file_4d.filename, True, True)
        self.analysing = True

    def update_rect_diffraction(self):
        if (
            self.states.interaction_mode != StatesHolder.InteractionMode.stem_4d
            or not self.window.image_view.tools["rectangle_selector"].is_active
            or self.file_4d is None
        ):
            return
        rectangle = self.window.image_view.tools["rectangle_selector"].get_scan_rectangle()

        if self.states.diffraction_view_channel == "camera":
            if self.file_4d.shape[2:] != self.diffraction_normalizer.shape:
                self.diffraction_normalizer.set_shape(self.file_4d.shape[2:])
            # TODO set fov if necessary
            if self.update_diffraction_lock.locked():  # multiple emits of ui during analysis causes to crash?
                return
            with self.update_diffraction_lock:
                dtype = self.file_4d.dtype
                image = np.zeros(self.file_4d.shape[2:], dtype=np.uint64)
                # with self.file_access_lock:
                # might be quite slow for big rectangles
                for i in range(rectangle[0], rectangle[0] + rectangle[2]):
                    for j in range(rectangle[1], rectangle[1] + rectangle[3]):
                        # image += self.file_4d.data4D[i, j, :, :]
                        chunk = self.file_4d.read_chunk((i, j))
                        one_image = np.empty(image.shape, dtype=dtype)
                        image += stem_measurements.bitshuffle_decompress(
                            np.frombuffer(chunk, dtype=np.uint8), one_image
                        )

                if self.file_4d.dtype == np.uint8:
                    image = image * 256 // (rectangle[2] * rectangle[3])  # boost up to 16 bit
                else:
                    image = image // (rectangle[2] * rectangle[3])
                self.diffraction_normalizer.set_chunk(image.ravel(), 0, 0, 0)

    def update_diffraction(self, pos):
        # use this just from UI
        if self.states.interaction_mode != StatesHolder.InteractionMode.stem_4d or self.file_4d is None:
            return

        if self.dead_neighbours is None:
            self.dead_neighbours = stem_measurements.dead_pixels_map.generate_correction_map(
                self.file_4d.dead_pixels, shape=self.file_4d.shape[2:]
            )

        self.last_position = pos

        n = self.file_4d.shape[0]  # assuming square
        fov = self.file_4d.parameters["scanning"]["fov (um)"]
        xy = ((np.array(pos) / fov + 0.5) * n).astype(np.int32)
        xy = np.clip(xy, 0, n - 1)

        ij = [xy[1], xy[0]]

        shape = self.file_4d.shape

        if self.states.diffraction_view_channel == "camera":
            if shape[2:] != self.diffraction_normalizer.shape:
                self.diffraction_normalizer.set_shape(shape[2:])
            # TODO set fov if necessary
            if self.update_diffraction_lock.locked():  # multiple emits of ui during analysis causes to crash?
                return
            with self.update_diffraction_lock:
                # with self.file_access_lock:
                chunk = self.file_4d.read_chunk((ij[0], ij[1]))
                # _, chunk = self.file_4d.data4D.id.read_direct_chunk((ij[0], ij[1], 0, 0))
                # chunk = self.file_4d.data4D[ij[0], ij[1], :, :].flatten()  # create copy

                # stem_measurements.dead_pixels_map.correct_dead_pixels(
                #     chunk.reshape(shape[2:]), dict(self.dead_neighbours)
                # )  # inplace
                dtype = self.file_4d.dtype
                image = np.empty((shape[2] * shape[3],), dtype=dtype)
                stem_measurements.bitshuffle_decompress(np.frombuffer(chunk, dtype=np.uint8), image.reshape(shape[2:]))

                self.diffraction_normalizer.set_chunk(image, 0, 0, 0)

    def update_virtual_image(self, virtual_images):
        # call from thread   or main thread
        if self.states.interaction_mode != StatesHolder.InteractionMode.stem_4d or self.file_4d is None:
            return

        self.file_4d.virtual_images = virtual_images

        n = 0
        channel = self.states.image_view_channel
        if channel not in virtual_images:
            for name in virtual_images.keys():
                if name[: len(channel)] == channel:
                    n += 1
            if n == 0:
                return

        fov = self.file_4d.parameters["scanning"]["fov (um)"]
        if self.window.image_view.main_image_item.fov != fov:
            self.window.image_view.main_image_item.fov = fov  # do not use set_fov - since this is not amain thread

        if n == 0:
            norm_image = (virtual_images[channel] * ((2**16 - 1) / np.max(virtual_images[channel]))).astype(np.uint16)

            if norm_image.shape != self.image_normalizer.shape:
                self.image_normalizer.set_shape(norm_image.shape)

            self.image_normalizer.set_chunk(norm_image.ravel(), 0, 0, 0)

        else:
            # TODO separate thread this is 40ms for 512x512px
            image_colored = self.get_colorized_img(virtual_images, channel, n, phi0=0)

            if image_colored.shape != self.image_normalizer.shape:
                self.image_normalizer.set_shape(image_colored.shape)

            self.image_normalizer.set_chunk(image_colored.reshape(-1, 3), 0, 0, 0)

    @staticmethod
    def get_colorized_img(virtual_images, channel, n, phi0=0):
        shape = virtual_images[f"{channel}_0"].shape

        img = np.zeros(shape, dtype=np.complex64)
        for i in range(n):
            img += virtual_images[f"{channel}_{i}"] * (np.cos(2 * np.pi * i / n) + 1j * np.sin(2 * np.pi * i / n))

        phi = np.arctan2(img.real, img.imag) + phi0

        img_abs = np.abs(img)

        cc = (img_abs / np.max(img_abs) * 65535).astype(np.uint16).ravel()
        xx = (cc * np.abs(phi.ravel() / (np.pi / 3) % 2 - 1)).astype(np.uint16)

        img_color = np.zeros((shape[0] * shape[1], 3), dtype=np.uint16)

        index = np.clip(((phi / np.pi + 1) * 3).astype(np.int8), 0, 5).ravel()  # to 0-6

        mask = index == 0
        img_color[mask, 0] = cc[mask]
        img_color[mask, 1] = xx[mask]
        img_color[mask, 2] = 0

        mask = index == 1
        img_color[mask, 0] = xx[mask]
        img_color[mask, 1] = cc[mask]
        img_color[mask, 2] = 0

        mask = index == 2
        img_color[mask, 0] = 0
        img_color[mask, 1] = cc[mask]
        img_color[mask, 2] = xx[mask]

        mask = index == 3
        img_color[mask, 0] = 0
        img_color[mask, 1] = xx[mask]
        img_color[mask, 2] = cc[mask]

        mask = index == 4
        img_color[mask, 0] = xx[mask]
        img_color[mask, 1] = 0
        img_color[mask, 2] = cc[mask]

        mask = index == 5
        img_color[mask, 0] = cc[mask]
        img_color[mask, 1] = 0
        img_color[mask, 2] = xx[mask]

        img_color = img_color.reshape(shape[0], shape[1], 3)

        return img_color

    def update_analysis_info(self, counter, total):
        # call from thread
        if counter in {-1, total}:
            if self.analysing:
                self._stem_4d.progress_button.set_progress_signal.emit(0, "reanalyze")
            self.analysing = False
            return

        if counter == 0:
            self.window.stem_4d.acquire_button.update_selected_signal.emit(True)
            self.start_tm = time.perf_counter()

        dt = time.perf_counter() - self.start_tm
        progress = counter / total
        if counter == 0:
            text = f"Analyzing {progress * 100:.1f}%"
        else:
            eta = dt / counter * (total - counter)

            text = f"Analyzing {progress * 100:.1f}% ETA:{int(eta)}s "

        if self.analysing:
            self._stem_4d.progress_button.set_progress_signal.emit(progress, text)

    def update_acquisition_info(self, counter, total):
        # call from thread
        if counter == -1:
            self.acquiring = False
            self.analysing = False
            return

        if counter == total:
            self._stem_4d.progress_button.set_progress_signal.emit(0, "reanalyze")
            self.acquiring = False
            return

        if counter == 0:
            self.start_tm = time.perf_counter()

        dt = time.perf_counter() - self.start_tm
        progress = counter / total
        if counter == 0:
            text = f"Acquiring {progress * 100:.1f}%"
        else:
            eta = dt / counter * (total - counter)

            text = f"Acquiring {progress * 100:.1f}% ETA:{int(eta)}s "

        if self.acquiring:
            self._stem_4d.progress_button.set_progress_signal.emit(progress, text)

    def set_stem_channel(self, channel):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d and self.file_4d is not None:
            if channel in self.file_4d.virtual_images or channel + "_0" in self.file_4d.virtual_images:
                self.update_virtual_image(self.file_4d.virtual_images)

    def set_diffraction_channel(self, channel):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
            if channel == "camera":
                self.camera_emit()
            elif channel == "fft":
                self.fft_processor.set_image(self.image_normalizer.raw_image)

    def stem_emit(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
            self.set_stem_channel(self.states.image_view_channel)

    def camera_emit(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d:
            if self.states.diffraction_view_channel == "camera":
                self.update_diffraction(self.last_position)
            elif self.states.diffraction_view_channel == "fft":
                self.fft_processor.emit()

    def save_file(self, filename, close):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d and self.file_4d is not None:
            self.file_4d.save()
            if filename != self.file_4d.filename:
                shutil.copyfile(self.file_4d.filename, filename)

            if not close:
                self.file_4d.load(filename)

    def new_file(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d and self.file_4d is not None:
            last_folder = preserved_state_saver.actual_state["last_save_directory"]
            name, *_arg = QtWidgets.QFileDialog.getSaveFileName(
                self.window, "Save File", last_folder + "untitled.h5", "stem 4D h5 file (*.h5)"
            )
            if not name:
                raise Exception("no name selected")

            preserved_state_saver.actual_state["last_save_directory"] = os.path.dirname(name) + "/"
            preserved_state_saver.save(get_config().data.data_folder)

            self.filename = name
            self.window.tool_bar.file_saving.status_signal.emit(self.file_4d.filename, True, True)

    def close_open_file(self, value):
        if self.states.interaction_mode == StatesHolder.InteractionMode.stem_4d and self.file_4d is not None:
            if not value:
                self.save_file(self.filename, True)
            elif not self.file_4d.is_open():
                self.file_4d.load(self.filename)
