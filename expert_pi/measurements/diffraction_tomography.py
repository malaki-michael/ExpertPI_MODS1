import datetime

import cv2
import numpy as np

from expert_pi.measurements.data_formats.diffraction_tomography import DiffractionTiltSeries
from expert_pi.measurements.orientation import bragg_fitting


def process_4dstem_darkfield(
    tilt_series: DiffractionTiltSeries, virtual_image_threshold=0.7, radius_mask=0.1, indices=None
):
    shape = tilt_series.data5D.shape

    x = np.linspace(-1, 1, num=shape[-1])
    X, Y = np.meshgrid(x, x)  # assuming diffraction square

    vs = []  # virtual images
    masks = []
    ds = []  # diffraction patterns

    if indices is None:
        indices = range(shape[0])
    for i in indices:
        print(i, "/", shape[0], end="\r")
        v = np.zeros(shape[1:3])

        data_4d = tilt_series.data5D[i]
        d_all = np.mean(data_4d, axis=(0, 1))

        s = np.sum(d_all)
        center = np.array([np.sum(d_all * X) / s, np.sum(d_all * Y) / s])

        mask2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 > radius_mask**2

        v = np.zeros(shape[1:3])
        for k in range(shape[1]):
            for m in range(shape[2]):
                v[k, m] = np.sum(data_4d[k, m] * mask2)
        # ds.append(np.mean(d,axis=0))
        mask2 = v / np.max(v) > virtual_image_threshold

        vs.append(v)
        masks.append(mask2)
        ds.append(np.mean(data_4d[mask2], axis=0))
    return vs, masks, ds


def process_4dstem_spot_count(
    tilt_series: DiffractionTiltSeries,
    minimum_number_of_spots,
    minimal_spot_distance=5,
    relative_prominence=0.001,
    absolute_prominence=200,
    indices=None,
):
    shape = tilt_series.data5D.shape

    x = np.linspace(-1, 1, num=shape[-1])
    _xg, _yg = np.meshgrid(x, x)  # assuming diffraction square

    # nice spot filteration:

    angular_fov = tilt_series.parameters["detectors"]["camera"]["angular fov (mrad)"]

    shape = tilt_series.data5D.shape

    vs = []  # virtual images (number of spots)
    ds = []  # diffraction patterns
    masks = []

    if indices is None:
        indices = range(shape[0])

    for i in indices:
        print(i, end="\r")
        v = np.zeros(shape[1:3])
        mask = np.zeros(shape[1:3], dtype="bool")
        d = []
        for k in range(shape[1]):
            for m in range(shape[2]):
                image = tilt_series.data5D[i, k, m]
                xys, _v_max, _p = bragg_fitting.fit_diffraction_patterns(
                    np.array([image]), angular_fov, minimal_spot_distance, relative_prominence, absolute_prominence
                )[0]
                v[k, m] = xys.shape[0]
                mask[k, m] = xys.shape[0] >= minimum_number_of_spots
                if mask[k, m]:
                    d.append(image)
        ds.append(np.mean(d, axis=0))
        vs.append(v)
        masks.append(mask)

    return vs, masks, ds


def export_tiffs(ds, path):
    max = np.max(ds)
    for i in range(len(ds)):
        params = [259, 1]
        cv2.imwrite(f"{path}{i:06d}.tiff", (ds[i] / max * (2**16 - 1)).astype("uint16"), params)


def export_video(tilt_series, vs, masks, ds, path, diffraction_max=1000, image_max=40, cut=0, fps=10):
    # print angle
    font = cv2.FONT_HERSHEY_PLAIN
    org = (int(512 / vs[0].shape[0] * vs[0].shape[1]), 512 - 50)
    font_scale = 2
    color = (255, 255, 0)
    thickness = 1

    i = 0
    vv = cv2.resize(vs[i], (int(512 / vs[0].shape[0] * vs[0].shape[1]), 512), interpolation=cv2.INTER_NEAREST)

    image_shape = (vv.shape[1] + 512, vv.shape[0])
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, image_shape)

    for i in range(len(vs)):
        print(i, end="\r")
        vv = cv2.resize(vs[i], (int(512 / vs[0].shape[0] * vs[0].shape[1]), 512), interpolation=cv2.INTER_NEAREST)

        vv_norm = vv / image_max
        vv_mask = cv2.resize(
            (masks[i] * 1).astype("uint8"),
            (int(512 / vs[0].shape[0] * vs[0].shape[1]), 512),
            interpolation=cv2.INTER_NEAREST,
        )

        text = f"{tilt_series.angles[i] / np.pi * 180:5.1f} deg"

        vv_rgb = (np.clip(np.dstack([(1 - vv_mask) * vv_norm, vv_norm, (1 - vv_mask) * vv_norm]), 0, 1) * 255).astype(
            "uint8"
        )

        ds_crop = cv2.resize(ds[i][cut : 512 - cut, cut : 512 - cut] / diffraction_max, (512, 512))
        ds_rgb = (np.clip(np.dstack([ds_crop, ds_crop, ds_crop]), 0, 1) * 255).astype("uint8")

        im = np.hstack([vv_rgb, ds_rgb])

        im2 = cv2.putText(im, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        video.write(cv2.resize(im2, image_shape))

    video.release()


def export_pets_input(tilt_series, path, tiff_path):
    try:
        _aperpixel = tilt_series.parameters["detectors"]["camera"]["pixel size (A^-1)"]
    except:
        tilt_series.parameters["detectors"]["camera"]["pixel size (A^-1)"] = tilt_series.parameters["detectors"][
            "camera"
        ]["pixel size (1"]["A)"]

    with open(path, "w") as f:
        f.write(
            f"""# PETS input file for Rotation Electron Diffraction generated by `Tescan Expert PI`
# {str(datetime.datetime.now())}
# For definitions of input parameters, see: 
# http://pets.fzu.cz/

geometry precession
detector default
beamstop no
lambda {tilt_series.parameters["wavelenght (pm)"] / 100}
center    AUTO
# rec space area {tilt_series.parameters["detectors"]["camera"]["angular fov (mrad)"]} mrad
Aperpixel {tilt_series.parameters["detectors"]["camera"]["pixel size (A^-1)"]}
phi {tilt_series.parameters["detectors"]["camera"]["precession angle (mrad)"] / 1000 / np.pi * 180}
omega 0
bin 1
dstarmax  1.4
dstarmaxps  1.4
i/sigma    3.00    2.00
reflectionsize 10
noiseparameters 2.5000      0

imagelist
"""
        )
        for i in range(tilt_series.angles.shape[0]):
            f.write(f"{tiff_path}{i:06d}.tiff  {tilt_series.angles[i] / np.pi * 180}  0.00 \n")

        f.write(
            f"""
endimagelist
"""
        )
