import threading
import time
import traceback

from expert_pi.stream_processors import functions


class FFTProcessor:
    def __init__(self, update_fft, update_filtered_image=None, max_fps=10):
        self._min_wait_time = 1 / max_fps
        self._event = threading.Event()
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        self.image = None
        self.filter_mask = None
        self.fft_log_amplitude = None
        self.filtered_image = None

        self.update_fft = update_fft
        self.update_filtered_image = update_filtered_image

    def start(self):
        if self._running or (self._thread is not None and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run)
        self._running = True
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._running = False
            self._event.set()

    def set_image(self, image):
        with self._lock:
            self.image = image
        self._event.set()

    def set_filter_mask(self, filter_mask):
        with self._lock:
            self.filter_mask = filter_mask

    def process(self, image, filter_mask=None):
        if filter_mask is None:
            fft_log_amplitude = functions.fft_image(image)
            filtered_image = image

        else:
            filtered_image, fft_complex = functions.filter_image(image, filter_mask)
            fft_log_amplitude = functions.fft_log_image(fft_complex)

        return fft_log_amplitude, filtered_image

    def emit(self):
        self._event.set()

    def _run(self):
        while True:
            last_update = time.monotonic()
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                image = self.image
                filter_mask = self.filter_mask  # TODO faster creation of copy

            if image is None:
                time.sleep(self._min_wait_time)
                continue

            self.fft_log_amplitude, self.filtered_image = self.process(image, filter_mask)

            if self.update_fft is not None:
                try:
                    self.update_fft(self.fft_log_amplitude)
                except:
                    traceback.print_exc()

            if self.update_filtered_image is not None:
                try:
                    self.update_filtered_image(self.filtered_image)
                except:
                    traceback.print_exc()

            from_last_update = time.monotonic() - last_update
            if from_last_update < self._min_wait_time:
                time.sleep(self._min_wait_time - from_last_update)
