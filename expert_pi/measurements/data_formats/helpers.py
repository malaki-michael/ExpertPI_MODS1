import h5py
import numpy as np


def dump_parameters(parameters: dict, file, root=True):
    if root:
        if "parameters" not in file:
            output = file.create_group("parameters")
        else:
            output = file["parameters"]
    else:
        output = file

    for key, value in parameters.items():
        if isinstance(value, dict):
            if key in output:
                group = output[key]
            else:
                group = output.create_group(key)
            group = dump_parameters(value, group, root=False)
        elif isinstance(value, list | np.ndarray):
            if key in output:
                output[key][:] = value
            else:
                output.create_dataset(key, data=value)

        elif key in output:
            output[key][()] = value
        else:
            output.create_dataset(key, shape=(), data=value)

    return output


def load_parameters(parameters: h5py._hl.group.Group, output=None):
    if output is None:
        output = {}
    for key, value in parameters.items():
        if isinstance(value, (h5py._hl.group.Group)):
            output[key] = {}
            output[key] = load_parameters(value, output[key])
        elif value.shape == ():
            output[key] = value[()]
        else:
            output[key] = value[:]
    return output


def write_numpy_arrays(file, name, data):
    if name in file:
        file[name][:] = data[:]
    else:
        file.create_dataset(name, data=data)


def write_scalar(file, name, scalar):
    if name in file:
        file[name][()] = scalar
    else:
        file.create_dataset(name, data=scalar)


def write_edx(file, edx_data, force_rewrite=False):
    if "edx" not in file:
        file.create_group("edx")

    for detector in edx_data:
        if detector in file["edx"] and force_rewrite:
            del file["edx"]["detector"]

        if detector not in file["edx"]:
            file["edx"].create_group(detector)
            file["edx"][detector].create_dataset("energy", data=edx_data[detector]["energy"])
            file["edx"][detector].create_dataset("dead_times", data=edx_data[detector]["dead_times"])


def load_edx(file):
    if "edx" not in file:
        return None

    edx = {}
    for channel in file["edx"].keys():
        edx[channel] = {
            "energy": file["edx"][channel]["energy"],
            "dead_times": file["edx"][channel]["dead_times"],
        }
    return edx


def load_numpy_arrays(file, name):
    return file[name][:]  # will create memory copy from h5 file


def check_version_and_type(file, required_version, required_type):
    if file["measurement_type"][()].decode() != required_type:
        raise Exception("wrong measurement_type", file["measurement_type"][()].decode())

    # ---------- version ----------
    version = file["version"][()].decode()
    if version != required_version:
        print("trying to load different version of " + required_type + " " + version + ", needed:" + required_version)
    return version
