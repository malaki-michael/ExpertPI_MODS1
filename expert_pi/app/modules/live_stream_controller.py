from functools import partial

from expert_pi import stream_clients
from expert_pi.app.states.states_holder import StatesHolder
from expert_pi.gui import main_window
from expert_pi.gui.main_window import MainWindow
from expert_pi.stream_clients import live_clients
from expert_pi.stream_processors import edx_processing, fft_processing, normalizer


class LiveProcessingThreadController:
    def __init__(
        self,
        window: main_window.MainWindow,
        states: StatesHolder,
        stem_client: live_clients.StemLiveStreamClient,
        camera_client: live_clients.CameraLiveStreamClient,
        edx_client: live_clients.EDXLiveStreamClient,
        image_normalizer: normalizer.Normalizer,
        diffraction_normalizer: normalizer.Normalizer,
    ):
        self.window = window
        self.states = states

        self.stem_client = stem_client
        self.camera_client = camera_client
        self.edx_client = edx_client

        self.emitters = {}

        self.stem_detectors = ["BF", "HAADF"]

        for detector in self.stem_detectors:
            self.emitters[detector + "_accumulator"] = stream_clients.live_clients.Emitter(
                partial(self.update_stem, channel=detector), max_fps=25
            )
            self.stem_client.add_emitter(self.emitters[detector + "_accumulator"], channel=detector)

        self.emitters["camera_accumulator"] = stream_clients.live_clients.Emitter(
            partial(self.update_camera, channel="camera"), max_fps=30
        )

        self.image_normalizer = image_normalizer

        self.diffraction_normalizer = diffraction_normalizer

        self.fft_processor = fft_processing.FFTProcessor(self.update_fft)

        self.edx_spectrum_processor = edx_processing.EDXSpectrumProcessor(self.update_spectrum_view)
        self.edx_map_processor = edx_processing.EDXMapProcessor(self.update_edx_map)

        self.edx_client.add_emitter(self.edx_spectrum_processor)
        self.edx_client.add_emitter(self.edx_map_processor)

        self.camera_client.add_emitter(self.emitters["camera_accumulator"])
        self.start()

        self._signals = self._create_signals()
        self.connect_signals(window)

    def connect_signals(self, window: MainWindow):
        self.window = window
        self._signals = self._create_signals()

        for signal, fce in self._signals.items():
            signal.connect(fce)

    def disconnect_signals(self):
        for signal, fce in self._signals.items():
            signal.disconnect(fce)

    def _create_signals(self) -> dict:
        signals = {
            self.window.stem_tools.image_type.clicked: self.set_stem_channel,
            self.window.diffraction_tools.image_type.clicked: self.set_diffraction_channel,
            self.window.stem_tools.visualisation.clicked: (
                lambda name: self.stem_emit() if name == "histogram" else None
            ),
            self.window.image_histogram.histogram_changed: lambda channel, alpha, beta: self.stem_emit(),
            self.window.image_view.request_update_signal: self.stem_emit,
            self.window.diffraction_tools.visualisation.clicked: (
                lambda name: self.camera_emit() if name == "histogram" else None
            ),
            self.window.diffraction_histogram.histogram_changed: lambda channel, alpha, beta: self.camera_emit(),
        }
        return signals

    def update_fft(self, fft_log_amplitude):
        if self.states.diffraction_view_channel == "fft":
            self.diffraction_normalizer.set_image(fft_log_amplitude)

    def update_camera(self, chunk, start, shape, frame_index, scan_id, channel=None):
        if self.states.diffraction_view_channel == channel:
            if shape != self.diffraction_normalizer.shape:
                self.diffraction_normalizer.set_shape(shape)
            self.diffraction_normalizer.set_chunk(chunk, start, frame_index, scan_id)

    def update_stem(self, chunk, start, shape, frame_index, scan_id, channel=None):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and self.states.image_view_channel == channel
        ):
            if shape != self.image_normalizer.shape:
                self.image_normalizer.set_shape(shape)
            self.image_normalizer.set_chunk(chunk, start, frame_index, scan_id)

            if self.states.live_fft_enabled:
                self.fft_processor.set_image(self.image_normalizer.raw_image)

    def update_edx_map(self, image, frame_index):
        if (
            self.states.interaction_mode == StatesHolder.InteractionMode.survey
            and self.states.image_view_channel == "EDX"
        ):
            self.image_normalizer.set_image(image, frame_index=frame_index)

    def update_spectrum_view(self, centers, histogram):
        self.window.spectrum_view.update_data(centers, histogram)

    def start(self):
        for detector in self.stem_detectors:
            self.emitters[detector + "_accumulator"].start()

        self.emitters["camera_accumulator"].start()
        self.fft_processor.start()
        self.edx_spectrum_processor.start()

    def stop(self):
        for detector in self.stem_detectors:
            self.emitters[detector + "_accumulator"].stop()
        self.emitters["camera_accumulator"].stop()
        self.fft_processor.stop()
        self.edx_spectrum_processor.stop()

    def update_normalizer(self, channel, alpha, beta):
        self.emitters[channel + "_norm"].set_alpha_beta(alpha, beta)
        self.emitters[channel + "_norm"].update()

    def set_stem_channel(self, channel):
        self.states.image_view_channel = channel
        if channel in self.stem_detectors and self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            self.emitters[channel + "_accumulator"].emit()

    def stem_emit(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            if self.states.image_view_channel == "EDX":
                self.edx_map_processor.emit()
            else:
                self.emitters[self.states.image_view_channel + "_accumulator"].emit()

    def camera_emit(self):
        if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            channel = self.states.diffraction_view_channel
            if channel == "camera":
                self.emitters[channel + "_accumulator"].emit()
            elif channel == "fft":
                self.fft_processor.emit()

    def set_diffraction_channel(self, channel):
        self.states.diffraction_view_channel = channel
        if self.states.interaction_mode == StatesHolder.InteractionMode.survey:
            if channel == "camera":
                self.states.live_fft_enabled = False
                self.emitters[channel + "_accumulator"].emit()
            elif channel == "fft":
                self.states.live_fft_enabled = True
                self.fft_processor.set_image(self.image_normalizer.raw_image)
                self.fft_processor.emit()
