import json
import socket
import struct
import threading
import time
import traceback
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class LiveStreamError(Exception):
    pass


class EmitterBase(ABC):
    def __init__(self, update_function=None, max_fps=20):
        self.update_function = update_function

        self._min_wait_time = 1 / max_fps
        self._event = threading.Event()
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        self._shape = (0, 0)

    @property
    def shape(self):
        return self._shape

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

    @abstractmethod
    def _run(self):
        pass


class Emitter(EmitterBase):
    def __init__(
        self, update_function: Callable[[np.ndarray, int, tuple[int, int], int], None] = None, max_fps: int = 20
    ):
        super().__init__(update_function, max_fps)

        self._image = np.zeros((0,), dtype=np.uint16)
        # self._shape = (0, 0)
        self._frame_index = 0
        self._scan_index = 0

        self.minimal_scan_index = 0  # if data lower hen this scan index then processing is ignored

        self._last_frame_index = -1
        self._start = 0
        self._end = 0
        self._update_all = False

    @property
    def image(self):
        return self._image

    def emit(self, frame_index=None):
        with self._lock:
            if frame_index is not None:
                self._frame_index = frame_index
            self._update_all = True
            self._event.set()

    def acquisition_started(self, shape, scan_index=None, buffer=None, force: bool = False):
        with self._lock:
            self._start = 0
            self._end = 0
            self._update_all = False
            self._frame_index = 0
            self._last_frame_index = 0
            self._scan_index = scan_index
            if buffer is not None:
                self._image = buffer
                self._shape = shape
            elif shape != self._shape or force:
                self._image = np.zeros((shape[0] * shape[1],), dtype=np.uint16)
                self._shape = shape

    def set_chunk(self, chunk, start, frame_index, set_data=True):
        chunk_size = len(chunk)

        if set_data:
            self._image[start : start + chunk_size] = chunk[:]

        with self._lock:
            self._frame_index = frame_index
            if start == 0:
                end = 0
                if frame_index - self._last_frame_index > 1:
                    self._update_all = True
            else:
                end = self._end
            self._end = start + chunk_size
            if end < self._start < self._end:
                self._update_all = True

        self._event.set()

    def _run(self):
        while True:
            last_update = time.monotonic()

            self._event.wait()
            self._lock.acquire()
            self._event.clear()

            if not self._running:
                self._lock.release()
                break

            shape = self._shape
            frame_index = self._frame_index
            scan_index = self._scan_index

            if scan_index is not None and scan_index < self.minimal_scan_index:
                self._lock.release()
                continue

            self._last_frame_index = self._frame_index
            if self._update_all:
                chunk = self._image
                self._update_all = False
                self._start = 0 if self._end == self._image.size else self._end
                start = 0
            elif self._start > self._end:
                start = self._start
                chunk = self._image[self._start :]
                self._start = 0
                self._event.set()
            else:
                start = self._start
                chunk = self._image[start : self._end]
                self._start = 0 if self._end == self._image.size else self._end

            self._lock.release()

            if self.update_function is not None:
                try:
                    self.update_function(chunk, start, shape, frame_index, scan_index)
                except:
                    traceback.print_exc()

            from_last_update = time.monotonic() - last_update
            if from_last_update < self._min_wait_time:
                time.sleep(self._min_wait_time - from_last_update)


class LiveStreamClient(ABC):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

        self._running = False
        self._thread = None
        self._socket = None

    @property
    def is_running(self):
        return self._running

    def connect(self):
        if self._socket is not None:
            raise RuntimeError("Already connected. Disconnect first.")

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()

    def disconnect(self):
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
            except OSError:
                pass

        if self._thread is not None:
            self._running = False
            self._thread.join(5)

        self._socket = None
        self._thread = None

    def _receive_data(self, length: int) -> bytearray:
        data = self._socket.recv(length, socket.MSG_WAITALL)
        if len(data) != length:
            raise LiveStreamError("data receiving failed")

        return data

    @abstractmethod
    def _run(self):
        pass

    def __del__(self):
        self.disconnect()


class CameraLiveStreamClient(LiveStreamClient):
    def __init__(self, host: str, port: int, max_fps: int = 25):
        super().__init__(host, port)

        self._image = None
        self._buffer = None
        self._last_send = time.monotonic()
        self._min_period = 1 / max_fps
        self._consume_emitters: typing.List[Emitter] = []

        self._shape = (0, 0)
        self._last_frame_id = 0

    @property
    def shape(self):
        return self._shape

    @property
    def frame_id(self):
        return self._last_frame_id

    def get_image(self):
        return self._image.reshape(self._shape).copy()

    def get_image_reference(self):
        """
        :return: reference to flatten image
        """
        return self._image

    def add_emitter(self, emitter: Emitter):
        self._consume_emitters.append(emitter)
        emitter.acquisition_started(self.shape, buffer=self._image)

    def remove_emitter(self, emitter: Emitter):
        if emitter in self._consume_emitters:
            self._consume_emitters.remove(emitter)

    def _run(self):
        try:
            self._socket.connect((self._host, self._port))
            while self._running:
                length_data = self._receive_data(16)
                message_length, header_length = struct.unpack("<QQ", length_data)
                data_length = message_length - header_length - 8

                header_data = self._receive_data(header_length)
                header = json.loads(header_data.decode("ASCII"))
                self._last_frame_id = header["FrameId"]
                shape = (header["Height"], header["Width"])
                if shape != self._shape:
                    self._buffer = bytearray(shape[0] * shape[1] * 2)
                    self._image = np.frombuffer(self._buffer, dtype=np.uint16, count=shape[0] * shape[1])
                    self._shape = shape
                    for emitter in self._consume_emitters:
                        emitter.acquisition_started(shape, buffer=self._image)

                recv_size = self._socket.recv_into(self._buffer, data_length, socket.MSG_WAITALL)
                if recv_size != data_length:
                    raise LiveStreamError("data receiving failed")

                now = time.monotonic()
                dt = now - self._last_send
                if dt > self._min_period:
                    self._last_send = now
                    for emitter in self._consume_emitters:
                        emitter.emit(frame_index=self._last_frame_id)

                # barrier for higher camera fps
                else:
                    time.sleep(0.01)

        except (OSError, LiveStreamError):
            self._running = False

        finally:
            self._socket.close()
            self._socket = None


class StemLiveStreamClient(LiveStreamClient):
    def __init__(self, host: str, port: int):
        super().__init__(host, port)

        self._emitters: dict[str, list[Emitter]] = {}
        self._channels = 0

        self._shape = (0, 0)
        self._scan_index = None

    def connect(self):
        super().connect()
        self._channels = 0

    def add_emitter(self, emitter: Emitter, channel: str):
        if channel not in self._emitters:
            self._emitters[channel] = [emitter]
        elif emitter not in self._emitters[channel]:
            self._emitters[channel].append(emitter)
        emitter.acquisition_started(self._shape, scan_index=self._scan_index, force=True)

    def remove_emitter(self, emitter: Emitter, channel: str):
        if channel in self._emitters and emitter in self._emitters[channel]:
            self._emitters[channel].remove(emitter)

    def _run(self):
        try:
            self._socket.connect((self._host, self._port))
            while self._running:
                length_data = self._receive_data(16)
                message_length, header_length = struct.unpack("<QQ", length_data)

                header_data = self._receive_data(header_length)
                header = json.loads(header_data.decode("ASCII"))

                if "Size" in header:
                    self._channels = len(header["Detectors"])
                    self._shape = (header["Size"]["Height"], header["Size"]["Width"])
                    self._scan_index = header["ScanId"]

                    for channel in header["Detectors"]:
                        if channel in self._emitters:
                            for emitter in self._emitters[channel]:
                                emitter.acquisition_started(self._shape, scan_index=self._scan_index)

                elif self._channels != 0:
                    data_length = message_length - header_length - 8
                    data = self._receive_data(data_length)

                    chunk = np.frombuffer(data, dtype=np.uint16)
                    channel = header["Detector"]
                    if channel in self._emitters:
                        for emitter in self._emitters[channel]:
                            emitter.set_chunk(chunk, start=header["PixelOffset"], frame_index=header["FrameIndex"])

        except (OSError, LiveStreamError):
            self._running = False

        finally:
            self._socket.close()
            self._socket = None


class EDXLiveStreamClient(LiveStreamClient):
    def __init__(self, host: str, port: int, processor: "EDXProcessor" = None):
        super().__init__(host, port)

        self._emitters: list["EDXSpectrumProcessor" or "EDXProcessor"] = [] if processor is None else [processor]
        self._acquisition_started = False

    def connect(self):
        super().connect()
        self._acquisition_started = False

    def add_emitter(self, emitter: "EDXSpectrumProcessor" or "EDXProcessor"):
        self._emitters.append(emitter)

    def remove_emitter(self, emitter: "EDXSpectrumProcessor"):
        self._emitters.remove(emitter)

    def _run(self):
        try:
            self._socket.connect((self._host, self._port))
            while self._running:
                length_data = self._receive_data(16)
                message_length, header_length = struct.unpack("<QQ", length_data)

                header_data = self._receive_data(header_length)
                header = json.loads(header_data.decode("ASCII"))

                if "Size" in header:
                    self._acquisition_started = True
                    shape = (header["Size"]["Height"], header["Size"]["Width"])

                    for emitter in self._emitters:
                        emitter.acquisition_started(shape)

                elif self._acquisition_started:
                    data_length = message_length - header_length - 8
                    data = self._receive_data(data_length)

                    energy, energy_pixels, dead, dead_pixels = [], [], [], []
                    for detector_header in header["edxData"]:
                        bytes_ = detector_header["bytesPerEvent"]
                        energy_length = detector_header["numberOfEnergyEvents"]
                        dead_time_length = detector_header["numberOfDeadTimeEvents"]
                        event_data = data[
                            detector_header["byteOffset"] : detector_header["byteOffset"]
                            + (energy_length + dead_time_length) * bytes_
                        ]

                        help_array = np.frombuffer(event_data, dtype=np.uint16).reshape((-1, 3))
                        events = help_array[:, 0].flatten()
                        pixels = np.frombuffer(help_array[:, 1:].tobytes(), dtype=np.uint32).flatten()

                        energy.append(events[:energy_length])
                        energy_pixels.append(pixels[:energy_length])
                        dead.append(events[energy_length:])
                        dead_pixels.append(pixels[energy_length:])

                    result = {
                        "energy": (np.concatenate(energy), np.concatenate(energy_pixels)),
                        "dead": (np.concatenate(dead), np.concatenate(dead_pixels)),
                    }
                    for emitter in self._emitters:
                        emitter.set_data(result, header["PixelOffset"], header["PixelCount"], header["FrameIndex"])

        except (OSError, LiveStreamError):
            self._running = False

        finally:
            self._socket.close()
            self._socket = None
