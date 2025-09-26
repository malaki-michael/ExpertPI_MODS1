import json

import cv2
import numpy as np
import scipy.constants

from expert_pi import grpc_client
from expert_pi.config import ConnectionConfig


def save_tiff_image(file_name, image):
    params = [259, 1]
    cv2.imwrite(file_name, image, params)


def save_png_image(file_name, image):
    params = None
    cv2.imwrite(file_name, image, params)


def save_metadata(file_name, metadata):
    with open(file_name + "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def get_microscope_metadata(config: ConnectionConfig):
    energy = grpc_client.gun.get_high_voltage()
    return {
        "microscope": {"type": "Tescan Tensor", "host": config.host, "api_version": grpc_client.server.get_version()},
        "optical_mode": grpc_client.microscope.get_optical_mode().name,
        "beam_energy (keV)": energy / 1e3,
        "wavelenght (pm)": get_wavelength(energy),
        "beam_current (nA)": grpc_client.illumination.get_current() * 1e9,
        "convergence_angle (mrad)": grpc_client.illumination.get_convergence_half_angle() * 1e3,
        "camera_stage_rotation (rad)": grpc_client.projection.get_camera_to_stage_rotation(),
        "detectors": {},
        "stage": {
            "rotation (º)": grpc_client.scanning.get_rotation() / np.pi * 180,
            "tilt_alpha (º)": grpc_client.stage.get_alpha() / np.pi * 180,
            "tilt_beta (º)": grpc_client.stage.get_beta() / np.pi * 180,
            "x (mm)": grpc_client.stage.get_x_y()["x"] * 1e3,
            "y (mm)": grpc_client.stage.get_x_y()["y"] * 1e3,
            "z (mm)": grpc_client.stage.get_z() * 1e3,
        },
    }


def get_wavelength(energy):
    """energy in electronvolts -> return in picometers"""
    phir = energy * (1 + scipy.constants.e * energy / (2 * scipy.constants.m_e * scipy.constants.c**2))
    g = np.sqrt(2 * scipy.constants.m_e * scipy.constants.e * phir)
    k = g / scipy.constants.hbar
    wavelength = 2 * np.pi / k
    return wavelength * 1e12  # to picometers


def get_stem_metadata(fov, pixel_time, channels=["BF", "HAADF"]):
    result = {"fov (um)": fov, "pixel_time (us)": pixel_time, "channels": {}}

    max_angles = grpc_client.projection.get_max_detector_angles()  # todo speed up by used cached value?
    HAADF_inserted = grpc_client.stem_detector.get_is_inserted(grpc_client.stem_detector.DetectorType.HAADF)

    if "BF" in channels:
        result["channels"]["BF"] = {
            "collection angles (mrad)": {
                "min": 1e3 * max_angles["bf"]["start"],
                "max": 1e3 * max_angles["bf"]["end"] if not HAADF_inserted else 1e3 * max_angles["haadf"]["start"],
            }
        }
    if "HAADF" in channels:
        result["channels"]["HAADF"] = {
            "collection angles (mrad)": {
                "min": 1e3 * max_angles["haadf"]["start"],
                "max": 1e3 * max_angles["haadf"]["end"],
            }
        }
    return result


def get_camera_metadata(exposure, precession=False, precession_angle=None, precession_frequency=None, wavelength=None):
    """Get camera metadata.

    exposure ms,
    precession_angle mrad,
    precession frequency kHz
    wavelength pm
    """
    results = {
        "angular fov (mrad)": 1e3 * grpc_client.projection.get_max_camera_angle() * 2,
        "exposure (ms)": exposure,
        "rotation (º)": grpc_client.projection.get_camera_to_stage_rotation() / np.pi * 180,
    }
    if precession:
        if precession_angle is None:
            precession_angle = grpc_client.scanning.get_precession_angle() * 1e3
        results["precession angle (mrad)"] = precession_angle

        if precession_frequency is None:
            precession_frequency = grpc_client.scanning.get_precession_frequency() / 1000

        results["precession frequency (kHz)"] = precession_frequency

    if wavelength:
        results["pixel size (A^-1)"] = (
            results["angular fov (mrad)"] * 1e-3 / (wavelength * 0.01) / 512
        )  # assuming 512 px per camera

    return results


def get_edx_metadata(channels):
    results = {"filter": grpc_client.xray.get_xray_filter_type().name, "channels": {}}
    for channel in channels:
        results["channels"][channel] = {"collection angle (srad)": 1.0}  # TODO put here exact value
    return results
