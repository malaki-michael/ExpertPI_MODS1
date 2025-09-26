import json
import socket
import struct
import threading
import traceback

import numpy as np
import stem_measurements


class CacheClient:
    PACKET_SIZE = 1048576  # according to some tests
    DATA_TYPES = {1: np.uint8, 2: np.uint16, 4: np.uint32}

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._socket = None
        self._is_connected = False
        self._lock = threading.Lock()

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self, time_out=None):
        if self._socket is not None:
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(time_out)
        self._socket.connect((self._host, self._port))
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def __del__(self):
        self.disconnect()

    # def _receive_data(self, length: int) -> np.ndarray:
    #     data = np.empty(length, dtype=np.uint8)
    #     mem_data = memoryview(data)
    #     data_length = 0
    #
    #     while data_length < length:
    #         rest = length - data_length
    #         size = self.PACKET_SIZE if rest > self.PACKET_SIZE else rest
    #         get_length = self._socket.recv_into(mem_data[data_length:], size, flags=socket.MSG_WAITALL)
    #         data_length += get_length
    #
    #     return data

    def _receive_data(self, length: int) -> np.ndarray:
        data = np.empty(length, dtype=np.uint8)
        mem_data = memoryview(data)
        self._socket.recv_into(mem_data, length, flags=socket.MSG_WAITALL)

        return data

    def get_item(self, scan_id: int, pixel_count: int, raw=False, bslz4_threads=1, additional_header={}):
        with self._lock:
            disconnect = False
            try:
                if not self._is_connected:
                    disconnect = True
                    self.connect()

                # send request:
                header = {"scanId": int(scan_id), "pixelCount": int(pixel_count)}
                header.update(additional_header)
                json_string = json.dumps(header)
                length = struct.pack("Q", len(json_string))
                bytes_ = length + json_string.encode()
                self._socket.send(bytes_)

                # receive header:
                bytes_recv = self._receive_data(16)
                message_length, header_length = struct.unpack("<QQ", bytes_recv)

                header_data = self._receive_data(header_length)
                header = json.loads(header_data.tobytes().decode("ASCII"))

                message_length -= 16 + header_length
                data = {}
                if "stemData" in header:
                    byte_size = header["pixelCount"] * len(header["stemData"]) * 2
                    stem_data = self._receive_data(byte_size)
                    message_length -= byte_size
                    data["stemData"] = stem_data if raw else self.parse_stem_data(header, stem_data)

                if "edxData" in header:
                    byte_size = sum([
                        (item["numberOfEnergyEvents"] + item["numberOfDeadTimeEvents"]) * item["bytesPerEvent"]
                        for item in header["edxData"]
                    ])

                    edx_data = self._receive_data(byte_size)
                    message_length -= byte_size
                    data["edxData"] = edx_data if raw else self.parse_edx_data(header, edx_data)

                if "cameraData" in header:
                    byte_size = sum(header["cameraData"]["lengths"])
                    camera_data = self._receive_data(byte_size)
                    message_length -= byte_size
                    data["cameraData"] = (
                        camera_data if raw else self.parse_camera_data(header, camera_data, bslz4_threads)
                    )
                if "postprocessingData" in header:
                    data["postprocessingData"] = {}
                    for name, item in header["postprocessingData"].items():
                        byte_size = header["postprocessingData"][name]["byteSize"]
                        message_length -= byte_size
                        data["postprocessingData"][name] = self._receive_data(byte_size)  # TODO support non-raw data
                if message_length > 0:
                    self._receive_data(message_length)
                    raise Exception(f"still need to receive {message_length}B")

            except Exception as error:
                traceback.print_exc()
                self.disconnect()
                return False, None
            finally:
                if disconnect:
                    self.disconnect()

            return header, data

    @staticmethod
    def parse_stem_data(header, data) -> dict:
        output_data = {}
        for item in header["stemData"]:
            name = item["detector"]
            output_data[name] = np.frombuffer(
                data, dtype=np.uint16, count=header["pixelCount"], offset=item["byteOffset"]
            )

        return output_data

    @staticmethod
    def parse_edx_data(header, data) -> dict:
        output_data = {}
        byte_offset_start = min([item["byteOffset"] for item in header["edxData"]])

        for detector in header["edxData"]:
            offset = detector["byteOffset"] - byte_offset_start
            bytes_ = detector["bytesPerEvent"]
            energy_length = detector["numberOfEnergyEvents"]
            event_data = data[offset : offset + (energy_length + detector["numberOfDeadTimeEvents"]) * bytes_]

            help_array = np.frombuffer(event_data, dtype=np.uint16).reshape((-1, 3))
            events = help_array[:, 0].flatten()
            pixels = np.frombuffer(help_array[:, 1:].tobytes(), dtype=np.uint32).flatten()

            output_data[detector["detector"]] = {
                "energy": (events[:energy_length], pixels[:energy_length]),
                "dead_times": (events[energy_length:], pixels[energy_length:]),
            }

        return output_data

    @classmethod
    def parse_camera_data(cls, header, data, bslz4_threads=1) -> np.ndarray:
        dtype = cls.DATA_TYPES[header["cameraData"]["bytesPerPixel"]]

        if header["cameraData"]["compressionType"] == "None" or header["cameraData"]["compressionType"] == 0:
            buffer_data = np.frombuffer(data, dtype=dtype)
            output_data = buffer_data.reshape((
                header["pixelCount"],
                header["cameraData"]["imageWidth"],
                header["cameraData"]["imageHeight"],
            ))
        elif header["cameraData"]["compressionType"] == "Jpeg2000" or header["cameraData"]["compressionType"] == 1:
            raise NotImplementedError("Jpeg2000 is deprecated ")
        elif header["cameraData"]["compressionType"] == "Bslz4" or header["cameraData"]["compressionType"] == 2:
            output_data = np.empty(
                (header["pixelCount"], header["cameraData"]["imageWidth"], header["cameraData"]["imageHeight"]),
                dtype=dtype,
            )
            stem_measurements.bitshuffle_decompress_batch(
                data, header["cameraData"]["lengths"], output_data, thread_count=bslz4_threads
            )
        else:
            raise ValueError("unknown camera compression type")

        return output_data
