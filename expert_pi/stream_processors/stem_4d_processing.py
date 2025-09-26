import threading
import time
import traceback

import numba
import numpy as np
import stem_measurements.dead_pixels_map
import stem_measurements.virtual_detectors

from expert_pi.measurements.data_formats import stem4d
from expert_pi.stream_clients import cache


class FileSaver4DSTEM:
    """Saves the data from the stream to a file."""

    def __init__(self):
        """Initialize the FileSaver4DSTEM object."""
        self.thread = None
        self.running = False
        self.processed_item = None

    def start(self, total_pixels: int, scan_id, stem4d: stem4d.STEM4D):
        self.total_pixels = total_pixels
        self.scan_id = scan_id
        self.stem4d = stem4d

        self.stem4d.edx = None

        if self.thread is not None and self.running:
            self.stop()

            if self.thread.is_alive():
                self.thread.join()

        self.thread = threading.Thread(target=self._run)

        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def __call__(self, header, data):
        while self.processed_item is not None and self.running:
            time.sleep(0.01)  # block the upper thread if there is still previous item
        self.processed_item = (header, data)

    def _run(self):
        remaining = self.total_pixels
        # i = 0
        # j = 0

        while remaining > 0 and self.running:
            if self.processed_item is None:
                time.sleep(0.01)  # TODO conditional wakeup
                continue
            header, data = self.processed_item  # data is still raw
            self.processed_item = None

            if header["scanId"] != self.scan_id:
                print("wrong scan id", header["scanId"])
                continue

            if not header["cameraData"]["compressionType"] == 2:
                raise Exception(
                    "data from stream must be already compressed, found type:", header["cameraData"]["compressionType"]
                )

            self.stem4d.write_chunks(data["cameraData"], header["cameraData"]["lengths"])

            # byte_counter = 0
            # for k in range(header["pixelCount"]):
            #     L = header["cameraData"]["lengths"][k]
            #     self.stem4d.data4D.id.write_direct_chunk(
            #         [i, j, 0, 0], data["cameraData"][byte_counter : byte_counter + L]
            #     )
            #     byte_counter += L
            #     j += 1
            #     if j == header["scanDimensions"][2]:
            #         j = 0
            #         i += 1

            if "edxData" in data:
                edx_data = cache.CacheClient.parse_edx_data(header, data["edxData"])
                if self.stem4d.edx is None:
                    self.stem4d.edx = {}
                    for detector in edx_data.keys():
                        self.stem4d.edx[detector] = {
                            "energy": [],
                            "dead_times": [],
                        }

                for detector in edx_data.keys():
                    self.stem4d.edx[detector]["energy"].append(edx_data[detector]["energy"])
                    self.stem4d.edx[detector]["dead_times"].append(edx_data[detector]["dead_times"])

            remaining -= header["pixelCount"]

        if self.stem4d.edx is not None:
            for detector in self.stem4d.edx.keys():
                self.stem4d.edx[detector]["energy"] = np.hstack(self.stem4d.edx[detector]["energy"])
                self.stem4d.edx[detector]["dead_times"] = np.hstack(self.stem4d.edx[detector]["dead_times"])

        self.running = True


class Processor4DSTEMLive:
    def __init__(self):
        self.thread = None
        self.running = False
        self.processed_item = None

        self.dimensions = (0, 0, 0, 0)
        self.masks = {}  # channel:mask
        self.virtual_images = {}  # channel:image

        self.consumers = []
        self.info_consumers = []

    def set_consumers(self, consumers, info_consumers=[]):
        # consumers: list(Callable)
        self.consumers = consumers
        self.info_consumers = info_consumers

    def set_masks(self, masks):
        self.masks = masks
        self.virtual_images = {}

    def start(self, dimensions, scan_id, stem4d: stem4d.STEM4D | None):
        self.dimensions = dimensions
        self.scan_id = scan_id
        self.stem4d = stem4d

        stem_measurements.dead_pixels_map.set_dead_pixels(stem4d.dead_pixels)

        if self.thread is not None and self.running:
            self.stop()
            if self.thread.is_alive():
                self.thread.join()

        if self.stem4d is not None:
            self.stem4d.channels = [channel for channel in self.masks]
            for channel in self.masks:
                self.stem4d.virtual_images[channel] = np.zeros(dimensions[:2], dtype=np.uint64)
                self.virtual_images[channel] = self.stem4d.virtual_images[channel]
        else:
            for channel in self.masks:
                self.virtual_images[channel] = np.zeros(dimensions[:2], dtype=np.uint64)

        self.thread = threading.Thread(target=self._run)
        self.running = True
        self.thread.start()

        self.error_item = None

    def stop(self):
        self.running = False

    def __call__(self, header, data):
        while self.processed_item is not None and self.running:
            print("sleeping live processing")
            time.sleep(0.01)  # block the upper thread if there is still previous item
        self.processed_item = (header, data)

    def _run(self):
        remaining = self.dimensions[0] * self.dimensions[1]

        position = 0

        for consumer in self.info_consumers:
            consumer(0, self.dimensions[0] * self.dimensions[1])

        while remaining > 0 and self.running:
            try:
                start_tm = time.perf_counter()
                if self.processed_item is None:
                    time.sleep(0.01)  # TODO conditional wakeup
                    continue
                print("processing item", remaining, self.dimensions[0] * self.dimensions[1])
                header, data = self.processed_item  # data is still raw
                self.processed_item = None

                if header["scanId"] != self.scan_id:
                    print("wrong scan id", header["scanId"])
                    continue

                if not header["cameraData"]["compressionType"] == 2:
                    raise Exception(
                        "data from stream must be already compressed, found type:",
                        header["cameraData"]["compressionType"],
                    )

                if header["cameraData"]["bytesPerPixel"] == 1:
                    dtype = np.uint8
                elif header["cameraData"]["bytesPerPixel"] == 2:
                    dtype = np.uint16
                else:
                    dtype = np.uint32

                decompressed_data = np.empty(
                    (header["pixelCount"], header["cameraData"]["imageHeight"], header["cameraData"]["imageWidth"]),
                    dtype=dtype,
                )
                stem_measurements.bitshuffle_decompress_batch(
                    data["cameraData"], header["cameraData"]["lengths"], decompressed_data, thread_count=8
                )
                decompress_tm = time.perf_counter()

                for channel, mask in self.masks.items():
                    if mask is None:
                        stem_measurements.virtual_detectors.reset_detector_mask()
                    else:
                        stem_measurements.virtual_detectors.set_detector_mask(
                            mask
                        )  # TODO allow multiple masks now it needs to recalculate everytime
                    self.virtual_images[channel].flat[position : position + header["pixelCount"]] = (
                        stem_measurements.virtual_detectors.virtual_detector(decompressed_data, thread_count=8)
                    )
                virtual_image_tm = time.perf_counter()

                for consumer in self.consumers:
                    consumer(self.virtual_images)
                for consumer in self.info_consumers:
                    consumer(position + header["pixelCount"], self.dimensions[0] * self.dimensions[1])

                consumers_tm = time.perf_counter()
                m = header["pixelCount"]
                print(
                    f"decompress: {(decompress_tm - start_tm) / m * 1e6:5.3f} "
                    f"us mask: {(virtual_image_tm - decompress_tm) / m * 1e6:5.3f} "
                    f"us consumer: {(consumers_tm - virtual_image_tm) / m * 1e6:5.3f} us "
                )

            except:
                import traceback

                traceback.print_exc()

                self.error_item = (header, data)

                print(header, data.keys(), len(data["cameraData"]))

            remaining -= header["pixelCount"]
            position += header["pixelCount"]
        for consumer in self.info_consumers:
            consumer(-1, self.dimensions[0] * self.dimensions[1])
        self.running = False


class Processor4DSTEMFile:
    def __init__(self):
        self.threads = []
        self.running = False
        self.file_reading = False
        self.processed_item = None

        self.masks = {}  # channel:mask
        self.virtual_images = {}  # channel:image

        self.consumers = []  # virtual_images
        self.inso_consumers = []  # counter,total

        self.batch_size = 1024
        self.first_show_level = int(np.log2(self.batch_size) / 2)

        self._reset_lock = threading.Lock()
        self._reset_event = threading.Event()

        self._event = threading.Event()
        self.readout_item = None

        self.initialize()

    def initialize(self):
        self.threads = [threading.Thread(target=self._run_process_levels), threading.Thread(target=self._run_file_read)]
        self.file_reading = True
        self.running = True
        for thread in self.threads:
            thread.start()

        self.readout_item = None

    def finalize(self):
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join()

    def set_consumers(self, consumers, info_consumers=[]):
        # consumers: list(Callable)
        self.consumers = consumers
        self.info_consumers = info_consumers

    @staticmethod
    def get_indices(n, m, level):
        if level == 0:
            return np.array([0]), np.array([0])

        max_level_n = int(np.ceil(np.log(n) / np.log(2)))
        max_level_m = int(np.ceil(np.log(m) / np.log(2)))

        d = max(max_level_n, max_level_m) - level
        x, y = np.meshgrid(
            np.arange(2 ** min(level, max_level_n)), np.arange(2 ** min(level, max_level_m)), indexing="ij"
        )
        mask = ((x % 2) | (y % 2)) == 1

        x2 = x[mask] * 2**d
        y2 = y[mask] * 2**d

        mask2 = (x2 < n) & (y2 < m)
        x2 = x2[mask2]
        y2 = y2[mask2]
        i = np.arange(len(x2))
        np.random.shuffle(i)

        return x2[i], y2[i]

    def analyze(self, masks, stem4d: stem4d.STEM4D, file_access_lock):
        with self._reset_lock:
            stem_measurements.dead_pixels_map.set_dead_pixels(stem4d.dead_pixels)
            self.stem4d = stem4d
            self.file_access_lock = file_access_lock
            self.masks = masks
            self.reset = True

            for consumer in self.info_consumers:
                consumer(0, stem4d.shape[0] * stem4d.shape[1])

        self._reset_event.set()

    def stop(self):
        with self._reset_lock:
            self.reset = True

    def __call__(self, header, data):
        while self.processed_item is not None and self.running:
            print("sleeping live processing")
            time.sleep(0.01)  # block the upper thread if there is still previous item
        self.processed_item = (header, data)

    def _run_file_read(self):
        while self.file_reading:
            while not self._reset_event.wait(timeout=0.5):
                if not self.running:
                    return

            with self._reset_lock:
                self._reset_event.clear()
                self.reset = False
                file_4d = self.stem4d
                masks = self.masks

            shape = file_4d.shape
            remaining = shape[0] * shape[1]

            max_level = int(np.ceil(np.log(max(shape[:2])) / np.log(2)))

            start_total = time.perf_counter()

            total_done = 0

            for level in range(max_level + 1):
                x2, y2 = self.get_indices(shape[0], shape[1], level)

                rect_size = 2 ** (max_level - level)

                remaining_level = len(x2)
                position_level = 0

                while remaining_level > 0:
                    if not self.running:
                        return
                    if self.reset:
                        break

                    to_read = min(remaining_level, self.batch_size)

                    total_done += to_read

                    x2b = x2[position_level : position_level + to_read]
                    y2b = y2[position_level : position_level + to_read]

                    compressed_data = np.empty(
                        int(shape[2] * shape[3] * file_4d.dtype.itemsize * to_read * 1.1),
                        dtype=np.uint8,
                    )  # compressed data will never be more then 110% uncompressed size

                    position = 0
                    lengths = []
                    failed = []
                    for i in range(to_read):
                        try:
                            # with self.file_access_lock:
                            #     mask, chunk = file_4d.data4D.id.read_direct_chunk(
                            #         [x2b[i], y2b[i], 0, 0], out=compressed_data[position:]
                            #     )
                            chunk = file_4d.read_chunk((x2b[i], y2b[i]), output=compressed_data[position:])
                        except Exception as e:
                            print(f"error during reading chunk: {e}")

                            traceback.print_exc()
                            failed.append(i)
                            continue
                        position += chunk.nbytes
                        lengths.append(chunk.nbytes)

                    mask_valid = np.ones(to_read, dtype=bool)
                    if failed:
                        mask_valid[np.array(failed)] = False
                        if np.sum(mask_valid) == 0:
                            continue

                    while self.readout_item is not None:
                        print("waiting for processing")
                        if self.running is False:
                            return
                        time.sleep(0.2)
                    self.readout_item = (
                        compressed_data[:position],
                        shape,
                        file_4d.dtype,
                        lengths,
                        x2b[mask_valid],
                        y2b[mask_valid],
                        rect_size,
                        level,
                        masks,
                        total_done,
                    )
                    self._event.set()

                    position_level += to_read
                    remaining_level -= to_read

                if self.reset:
                    break
                remaining -= remaining_level

            print(
                "reading done",
                time.perf_counter() - start_total,
                "s",
                total_done,
                (time.perf_counter() - start_total) / total_done * 1e6,
                "us",
            )

        self.file_reading = False
        self._event.set()

    def _run_process_levels(self):
        counter = 0

        while True:
            while True:
                temp = self.readout_item
                if temp is None:
                    while not self._event.wait(timeout=0.5):
                        if not self.running:
                            return
                else:
                    self._event.clear()
                    self.readout_item = None
                    break

            compressed_data, shape, dtype, lengths, x2b, y2b, rect_size, level, masks, total_done = temp

            try:
                decompressed_data = np.empty((len(lengths), shape[2], shape[3]), dtype=dtype)
                stem_measurements.bitshuffle_decompress_batch(
                    compressed_data, lengths, decompressed_data, thread_count=8
                )

                self.clean_virtual_images(masks.keys(), shape)

                for channel, mask in masks.items():
                    if mask is None:
                        stem_measurements.virtual_detectors.reset_detector_mask()
                        result = stem_measurements.virtual_detectors.virtual_detector(decompressed_data, thread_count=8)
                    else:
                        # TODO allow multiple masks now it needs to recalculate everytime
                        stem_measurements.virtual_detectors.set_detector_mask(mask)
                        result = stem_measurements.virtual_detectors.virtual_detector(decompressed_data, thread_count=8)
                        result = result[0]
                    color_virtual_image(self.virtual_images[channel], x2b, y2b, result, rect_size)

                counter += len(lengths)

                if level >= self.first_show_level:
                    for consumer in self.consumers:
                        consumer(self.virtual_images)

                with self._reset_lock:
                    if self.reset:
                        total_done = -1
                    for consumer in self.info_consumers:
                        consumer(total_done, shape[0] * shape[1])
            except:
                import traceback

                traceback.print_exc()

            if not self.file_reading and self.readout_item is None:
                break

        self.running = False

    def clean_virtual_images(self, channels, shape):
        to_delete = []
        for channel in self.virtual_images:
            if channel not in channels:
                to_delete.append(channel)
        for channel in to_delete:
            del self.virtual_images[channel]

        for channel in channels:
            if (
                channel not in self.virtual_images
                or self.virtual_images[channel].shape[0] != shape[0]
                or self.virtual_images[channel].shape[1] != shape[1]
            ):
                self.virtual_images[channel] = np.empty(shape[:2], dtype=np.uint64)


@numba.njit
def color_virtual_image(img, x2, y2, values, rect_size):
    for i in range(len(x2)):
        img[x2[i] : x2[i] + rect_size, y2[i] : y2[i] + rect_size] = values[i]
