import h5py
import hdf5plugin
import numpy as np

from expert_pi.measurements.data_formats import helpers

VERSION = "0.2"


class DiffractionTiltSeries:
    """
    h5file wrapper 3DED (5DED):
    - K-alpha size
    - N x M- image size (or n x m for 5DSTEM rectangle)
    - P x Q diffraction size

    measurement_type:string
    version:string

    angles : array<K> # in rads
    transforms2x2 :array<K x 2 x 2> transformations from scanning to sample plane including alpha

    channels: list[string]
    stem_images: {<channel>:array<N x M>}

    diffractions array <K x P x Q> processed data
    dead_pixels: [[x,y]]
    data5D: array<n x m x P x Q> raw data
    diffraction_selections: array<K x (i,j,height,width)>

    edx: {<detector>:{energy:array<pixel_indices,energies>,
                    deadTime:array<pixel_indices,deadtimes>}}
    parameters : {<acquisition parameters>}

    """

    @staticmethod
    def open(filename):
        return DiffractionTiltSeries(filename, mode="r+", from_file=True)

    @staticmethod
    def new(filename, alpha_steps, shape_4d, channels=["BF"], dtype=np.uint16, dtype_4d=np.uint8):
        return DiffractionTiltSeries(
            filename,
            alpha_steps=alpha_steps,
            shape_4d=shape_4d,
            channels=channels,
            dtype=dtype,
            dtype_4d=dtype_4d,
            mode="w",
        )

    def __init__(
        self,
        filename,
        alpha_steps=None,
        shape_4d=None,
        channels=["BF"],
        dtype=np.uint16,
        dtype_4d=np.uint8,
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

            self.diffractions = np.empty((alpha_steps, shape_4d[2], shape_4d[3]), dtype=dtype)
            self.data5D = self.file.create_dataset(
                "data5D",
                shape=(alpha_steps, *shape_4d),
                chunks=(1, 1, 1, shape_4d[2], shape_4d[3]),
                dtype=dtype_4d,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
            self.diffraction_selections = np.empty((alpha_steps, 4), dtype=np.float64)

            self.edx = None
            self.parameters = {}
            self.dead_pixels = []

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
            if channel in self.stem_images:
                helpers.write_numpy_arrays(self.file["stem_images"], channel, self.stem_images[channel])

        for channel in self.file["stem_images"]:
            if channel not in self.channels:
                del self.file["stem_images"][channel]

        helpers.write_numpy_arrays(self.file, "diffractions", self.diffractions)
        if len(self.dead_pixels) > 0:
            helpers.write_numpy_arrays(self.file, "dead_pixels", np.array(self.dead_pixels))

        # data5D: already written
        helpers.write_numpy_arrays(self.file, "diffraction_selections", self.diffraction_selections)

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
        if self.version == "0.1":
            self.file.close()
            self.file, self.version = convert_from_v1(filename)

        self.angles = self.file["angles"][:]
        self.transforms2x2 = self.file["transforms2x2"][:]

        self.channels = self.file["channels"][:].astype("str")  # copy and decode to string
        self.stem_images = {}
        for channel in self.file["stem_images"]:
            self.stem_images[channel] = self.file["stem_images"][channel][:]

        self.diffractions = self.file["diffractions"][:]
        if "dead_pixels" in self.file:
            self.dead_pixels = self.file["dead_pixels"][:]
        else:
            self.dead_pixels = []

        self.data5D = self.file["data5D"]  # keep on disk
        self.diffraction_selections = self.file["diffraction_selections"][:]

        self.edx = helpers.load_edx(self.file)

        self.parameters = helpers.load_parameters(self.file["parameters"])


def convert_from_v1(filename):
    print("converting to new version")
    f = h5py.File(filename, "r+")
    f["stem_images"] = f["images"]
    del f["images"]

    f["data5D"] = f["diffractions"]
    del f["diffractions"]
    f.create_dataset("diffractions", data=f["data5D"][:, 0, 0, :, :])
    f["version"][()] = VERSION
    return f, VERSION
