import h5py
import numpy as np

from expert_pi.measurements.data_formats import helpers

VERSION = "0.3"


class StemTiltSeries:
    """
    h5file wrapper stem tomography:
    - K-alpha size
    - N x M- image size

    measurement_type:string
    version:string

    angles : array<K> # in rads
    transforms2x2 :array<K x 2 x 2> transformations from scanning to sample plane including alpha

    channels: list[string]
    stem_images: {<channel>:array<Kx N x M>}

    shifts array<Kx2>
    aligned_images: {<channel>:array<Kx N x M>}

    edx: {<detector>:{energy:array<pixel_indices,energies>,
                    deadTime:array<pixel_indices,deadtimes>}}
    parameters : {<acquisition parameters>}
    """

    @staticmethod
    def open(filename):
        return StemTiltSeries(filename, mode="r+", from_file=True)

    @staticmethod
    def new(filename, alpha_steps, shape, channels=["BF", "HAADF"], dtype=np.uint16):
        return StemTiltSeries(
            filename,
            alpha_steps=alpha_steps,
            shape=shape,
            channels=channels,
            dtype=dtype,
            mode="w",
        )

    def __init__(
        self,
        filename,
        alpha_steps=None,
        shape=None,
        channels=["BF", "HAADF"],
        dtype=np.uint16,
        mode="r+",
        from_file=False,
    ):
        self.filename = filename
        if from_file:
            self.file = None
            self.load(filename)
        else:
            self.file = h5py.File(filename, mode=mode)
            self.version = VERSION

            self.angles = np.empty(alpha_steps, dtype=np.float64)  # rads
            self.transforms2x2 = np.empty((alpha_steps, 2, 2), dtype=np.float64)

            self.channels = channels
            self.stem_images = {}
            self.file.create_group("stem_images")

            for channel in self.channels:
                self.stem_images[channel] = self.file["stem_images"].create_dataset(
                    channel,
                    shape=(alpha_steps, *shape),
                    chunks=(1, *shape),
                    dtype=dtype,
                )

            self.shifts = None  # to be filled by processing
            self.aligned_images = None  # to be filled by processing

            self.edx = None
            self.parameters = {}

    def is_open(self):
        return self.file.__bool__()

    def save(self):
        if not self.is_open():
            raise Exception("attempt to save already closed file")

        helpers.write_scalar(self.file, "measurement_type", self.__class__.__name__)
        helpers.write_scalar(self.file, "version", self.version)

        helpers.write_numpy_arrays(self.file, "angles", self.angles)
        helpers.write_numpy_arrays(self.file, "transforms2x2", self.transforms2x2)

        helpers.write_numpy_arrays(self.file, "channels", np.array(self.channels).astype("S"))

        if "stem_images" not in self.file:
            self.file.create_group("stem_images")
        for channel in self.channels:
            helpers.write_numpy_arrays(self.file["stem_images"], channel, self.stem_images[channel])

        if self.shifts is not None:
            helpers.write_numpy_arrays(self.file, "shifts", self.shifts)

        if self.aligned_images is not None:
            if "aligned_images" not in self.file:
                self.file.create_group("aligned_images")
            for channel in self.aligned_images:
                helpers.write_numpy_arrays(self.file["aligned_images"], channel, self.aligned_images[channel])

        if self.edx is not None:
            helpers.write_edx(self.file, self.edx)

        helpers.dump_parameters(self.parameters, self.file)
        self.file.close()

    def load(self, filename=None, mode="r+"):
        if self.file is not None:
            self.file.close()
        if filename is not None:
            self.filename = filename

        self.file = h5py.File(self.filename, mode=mode)

        self.version = helpers.check_version_and_type(self.file, VERSION, self.__class__.__name__)

        self.channels = self.file["channels"][:].astype("str")  # copy and decode to string
        self.stem_images = {}
        for channel in self.channels:
            self.stem_images[channel] = self.file["stem_images"][channel]  # leave on disk

        if "shifts" in self.file:
            self.shifts = self.file["shifts"][:]
        else:
            self.shifts = None

        self.aligned_images = None
        if "aligned_images" in self.file:
            self.aligned_images = {}
            for channel in self.file["aligned_images"]:
                self.aligned_images[channel] = self.file["aligned_images"][channel]  # leave on disk

        self.edx = helpers.load_edx(self.file)

        self.parameters = helpers.load_parameters(self.file["parameters"])


# TODO update this
class StemTomography3DData:
    def __init__(self, data, parameters):
        self.data = data  # XYZ
        self.parameters = parameters

    def save(self, filename, mode="w"):
        self.file = h5py.File(filename, mode=mode)
        self.file.create_dataset("data", data=self.data)
        helpers.dump_parameters(self.parameters, self.file)
        self.file.close()

    def load(self, filename, load_to_memory=True):
        if self.file is not None:
            self.file.close()

        self.data = self.file["data"]
        self.parameters = helpers.load_parameters(self.file["parameters"])
