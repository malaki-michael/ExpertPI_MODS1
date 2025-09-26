from threading import Lock

import h5py
import hdf5plugin
import numpy as np
from stem_measurements.core import is_hdf5_enabled

if is_hdf5_enabled():
    from stem_measurements.core import HDF5RW

from expert_pi.measurements.data_formats import helpers

VERSION = "0.1"


class STEM4D:
    """
    h5file wrapper of 4DSTEM data NxMxPxQ with structure:

    measurement_type:string
    version:string

    channels: list[string]
    virtual_images: {<channel>:array<NxM>}
    data4D: array<NxMxPxQ>
    edx: {<detector>:{energy:array<pixel_indices,energies>,
                    deadTime:array<pixel_indices,deadtimes>}}
    parameters : {<acquisition parameters>}
    dead_pixels: [[x,y]]
    """

    @staticmethod
    def open(filename: str) -> "STEM4D":
        return STEM4D(filename, from_file=True)

    @staticmethod
    def new(filename: str, dimensions: tuple[int, ...], dtype: type = np.uint8) -> "STEM4D":
        return STEM4D(filename, dimensions=dimensions, dtype=dtype)

    def __init__(self, filename: str, dimensions: tuple[int, ...] = (), dtype=np.uint8, from_file=False):
        self.filename = filename
        self._access_lock = Lock()
        if from_file:
            self.load(filename)
        else:
            self._file = h5py.File(filename, mode="w")
            self.version = VERSION

            self.channels = []  # included also a virtual ones
            self.virtual_images = {}  # dict of channel:image

            self._file.create_dataset(
                "data4D",
                shape=dimensions,
                chunks=(1, 1, dimensions[2], dimensions[3]),
                dtype=dtype,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
            )

            if is_hdf5_enabled():
                self._file.close()
                self._rw = HDF5RW(filename, "data4D", "r+")

            self._i = 0
            self._j = 0

            self.dtype: np.dtype = np.dtype(dtype)
            self.shape = dimensions
            self.dead_pixels = np.array([[]])
            self.edx = None
            self.parameters = {}

    def is_open(self) -> bool:
        if is_hdf5_enabled():
            return self._rw.is_open()
        return bool(self._file)

    def close(self) -> None:
        if is_hdf5_enabled():
            self._rw.close()
        elif bool(self._file):
            self._file.close()

    def write_chunks(self, data: np.ndarray, lengths: list[int]) -> None:
        if not self.is_open():
            raise Exception("attempt to write to closed file")

        if is_hdf5_enabled():
            self._rw.write_chunks(data, lengths)
            return

        with self._access_lock:
            bytes_counter = 0
            for i in range(len(lengths)):
                chunk_size = lengths[i]
                data_to_write = data[bytes_counter : bytes_counter + chunk_size]
                self._file["data4D"].id.write_direct_chunk((self._i, self._j, 0, 0), data_to_write)
                bytes_counter += chunk_size
                self._j += 1
                if self._j >= self.shape[1]:
                    self._j = 0
                    self._i += 1

    def read_chunk(self, position: tuple[int, ...], output: np.ndarray | None = None) -> np.ndarray:
        if not self.is_open():
            raise Exception("attempt to write to closed file")

        if is_hdf5_enabled():
            result, chunk_size = self._rw.read_chunk(position, output)
            if output is not None:
                return result[:chunk_size]
            return result

        with self._access_lock:
            chunk: np.ndarray = self._file["data4D"].id.read_direct_chunk((*position, 0, 0), out=output)[1]
        return chunk

    def save(self) -> None:
        if not self.is_open():
            raise Exception("attempt to save already closed file")

        if is_hdf5_enabled():
            self._rw.close()
            self._file = h5py.File(self.filename, mode="r+")

        helpers.write_scalar(self._file, "measurement_type", self.__class__.__name__)
        helpers.write_scalar(self._file, "version", self.version)
        helpers.write_numpy_arrays(self._file, "channels", np.array(self.channels).astype("S"))

        if "virtual_images" not in self._file:
            self._file.create_group("virtual_images")
        for channel in self.channels:
            helpers.write_numpy_arrays(self._file["virtual_images"], channel, self.virtual_images[channel])

        for channel in self._file["virtual_images"]:
            if channel not in self.channels:
                del self._file["virtual_images"][channel]

        # data4D already written as dataset

        if len(self.dead_pixels) > 0:
            helpers.write_numpy_arrays(self._file, "dead_pixels", np.array(self.dead_pixels))

        if self.edx is not None:
            helpers.write_edx(self._file, self.edx)

        helpers.dump_parameters(self.parameters, self._file)

        self.close()

    def load(self, filename=None) -> None:
        self.close()

        if filename is not None:
            self.filename = filename

        self._file = h5py.File(self.filename, mode="r+")

        self.version = helpers.check_version_and_type(self._file, VERSION, self.__class__.__name__)

        self.channels = self._file["channels"][:].astype("str")  # copy and decode to string
        self.virtual_images = {}
        for channel in self.channels:
            self.virtual_images[channel] = self._file["virtual_images"][channel][:]

        if "dead_pixels" in self._file:
            self.dead_pixels: np.ndarray = self._file["dead_pixels"][:]  # copy to memory
        else:
            self.dead_pixels = []

        self.edx = helpers.load_edx(self._file)

        self.parameters = helpers.load_parameters(self._file["parameters"])

        data4D: h5py.Dataset = self._file["data4D"]
        self.dtype = data4D.dtype
        self.shape = data4D.shape

        if is_hdf5_enabled():
            self._file.close()
            self._rw = HDF5RW(self.filename, "data4D", "r")
