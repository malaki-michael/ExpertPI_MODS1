import time
import numpy as np
import cv2
from collections.abc import Callable

from expert_pi.stream_clients.live_clients import EmitterBase

ENERGY_VALUES = np.array(range(0, 2**16))


def generate_base_function(energy, mn_fwhm=130):
    mu = energy
    sigma = mn_fwhm / 2.3548 / np.sqrt(5900.3) * np.sqrt(mu)
    a = 1
    result = a / sigma * np.exp(-((ENERGY_VALUES - mu) ** 2) / (2.0 * sigma**2))

    return result / np.sum(result)


class EDXSpectrumProcessor(EmitterBase):
    _BIN_WIDTH = 10
    _MAX_ENERGY = 30_000

    def __init__(self, update_function: Callable[[np.ndarray, np.ndarray], None] = None, max_fps: int = 5):
        super().__init__(update_function, max_fps)

        self._cumulative_hist = np.zeros(self._MAX_ENERGY // self._BIN_WIDTH, dtype="int")
        self._centers = np.linspace(0, self._MAX_ENERGY, num=self._MAX_ENERGY // self._BIN_WIDTH + 1)
        self._energy_events = []

    def calibrate_spectrum(self, lines):
        with self._lock:
            _histogram = self._cumulative_hist

    def set_data(self, data: dict, *_):
        # hist, _ = np.histogram(data['energy'][0], bins=self._MAX_ENERGY//self._BIN_WIDTH, range=[0, self._MAX_ENERGY])

        # with self._lock:
        #     self._cumulative_hist += hist
        with self._lock:
            self._energy_events.append(data["energy"][0])
        self._event.set()

    def acquisition_started(self, _):
        with self._lock:
            self._cumulative_hist = np.zeros(self._MAX_ENERGY // self._BIN_WIDTH, dtype="int")
            # self._cumulative_hist[:] = 0

    def _run(self):
        while True:
            last_update = time.monotonic()
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                histogram = self._cumulative_hist
                energy_events = self._energy_events
                self._energy_events = []

            energy_events = np.concatenate(energy_events)
            hist, _ = np.histogram(energy_events, bins=self._MAX_ENERGY // self._BIN_WIDTH, range=(0, self._MAX_ENERGY))
            histogram += hist

            from_last_update = time.monotonic() - last_update
            if from_last_update < self._min_wait_time:
                time.sleep(self._min_wait_time - from_last_update)

            if self.update_function is not None:
                self.update_function(self._centers, histogram)


# TODO: refactor class, this is first slow working implementation
class EDXMapProcessor(EmitterBase):
    def __init__(self, update_function, max_fps: int = 5):
        super().__init__(None, max_fps)

        self.update_function = update_function

        self._matrix = None
        self._bases = None
        self._elements_list = []
        self.elements = {}

        self._image = None
        self._raw_images = None
        self._events = []
        self._events_cache = []
        self.max_cache_size = 100_000

        self.filter_factor = 3  # must be odd number

        self.start()

    def element_change(self, element_item):
        element_name = element_item.name
        recalculate_from_cache = False
        if element_name in self.elements:
            if not element_item.selected:
                del self.elements[element_name]
                recalculate_from_cache = True
            else:
                with self._lock:
                    self.elements[element_name]["color"] = element_item.color
                    self.elements[element_name]["active"] = element_item.active

        elif element_item.selected:
            bases = []
            for name, energy in element_item.lines.items():
                bases.append(generate_base_function(energy))

            element_base = np.sum(bases, axis=0)  # TODO weights of bases
            with self._lock:
                self.elements[element_name] = {
                    "color": element_item.color,
                    "base": element_base,
                    "active": element_item.active,
                }
            recalculate_from_cache = True

        if self._raw_images is None:
            return

        if recalculate_from_cache and self._shape is not None:
            n = self.generate_bases()
            shape = self._shape
            size = shape[0] * shape[1]
            raw_images = np.zeros((n, size), dtype=np.float64)

            with self._lock:
                self._raw_images = raw_images
            self.schedule_cache_recalculation()

        elif element_name in self._elements_list:
            self._event.set()

    def set_data(self, data: dict, start: int, pixels: int, frame_index: int):
        if self._shape is None:
            return

        event = {"data": data["energy"], "start": start, "pixels": pixels, "frame_index": frame_index}

        with self._lock:
            self._events.append(event)
            if len(self._events_cache) < self.max_cache_size:
                self._events_cache.append(event)

        self._event.set()

    def generate_bases(self):
        bases, element_list, colors = [], [], []
        with self._lock:
            for element_name, element in self.elements.items():
                element_list.append(element_name)
                colors.append(element["color"])
                bases.append(element["base"])

        bases = np.array(bases)
        n = bases.shape[0]

        m = []
        for i in range(n):
            m.append([])
            for j in range(n):
                m[i].append(np.sum(bases[i] * bases[j]))
        matrix = np.linalg.inv(m) if n > 0 else np.array([])

        with self._lock:
            self._bases = bases
            self._matrix = matrix
            self._elements_list = element_list
            self._colors = colors

        return n

    def schedule_cache_recalculation(self):
        with self._lock:
            self._events = self._events_cache

        self._event.set()

    def acquisition_started(self, shape):
        if shape == (1, 1):
            self._shape = None
            return

        self._shape = (shape[0], shape[1], 3)  # RGB 8 bit images
        size = shape[0] * shape[1]

        n = self.generate_bases()

        image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        raw_images = np.zeros((n, size), dtype=np.float64)

        with self._lock:
            self._events = []
            self._events_cache = []
            self._image = image
            self._raw_images = raw_images

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
                events = self._events
                self._events = []
                image = self._image
                raw_images = self._raw_images
                bases = self._bases
                matrix = self._matrix
                element_list = self._elements_list
                colors = self._colors

            if bases.size == 0:
                continue

            for batch in events:
                energies = batch["data"][0]
                pixels = batch["data"][1]
                probabilities = bases[:, energies]
                raw_images[:, pixels] += matrix @ probabilities
                last_frame_index = batch["frame_index"]

            last_frame_index = None

            ind = []
            for i, element in enumerate(element_list):
                if self.elements[element]["active"]:
                    ind.append(i)

            if len(ind) > 0:
                colors_selected = [colors[i] for i in ind]

                filtered_images = []

                if self.filter_factor > 0:
                    k_fact = self.filter_factor
                    kernel = np.ones((k_fact, k_fact), np.uint8)

                for i in ind:
                    r = cv2.normalize(
                        raw_images[i, :].reshape(self._shape[:2]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )

                    if self.filter_factor > 0:
                        r2 = cv2.dilate(r, kernel)
                        r3 = cv2.medianBlur(r2, k_fact)
                    else:
                        r3 = r

                    filtered_images.append(r3)

                mixed_image = np.zeros((filtered_images[0].shape[0], filtered_images[0].shape[1], 3), dtype="uint8")

                for i in range(len(filtered_images)):
                    color = colors_selected[i]
                    for j in range(3):
                        mixed_image[:, :, j] = cv2.addWeighted(
                            filtered_images[i],
                            int(color[1 + 2 * j : 3 + 2 * j], 16) / 255,
                            mixed_image[:, :, j],
                            1.0,
                            0.0,
                        )

                self.mixed_image = mixed_image

                image = mixed_image

            if self.update_function is not None:
                self.update_function((image * 256).astype("uint16"), last_frame_index)  # TODO To 16bit faster

            from_last_update = time.monotonic() - last_update
            if from_last_update < self._min_wait_time:
                time.sleep(self._min_wait_time - from_last_update)
