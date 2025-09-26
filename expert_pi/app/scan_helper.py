# TODO move here a scanning wrapper from scanning and camera acquisition

from expert_pi import grpc_client
import numpy as np


def stop():
    grpc_client.scanning.stop_scanning()


def set_fov(fov):
    grpc_client.scanning.set_field_width(fov)


def start_rectangle_scan(
    pixel_time: float = 300e-9,
    total_size: int = 1024,
    frames: int = 1,
    rectangle=None,
    detectors: list[grpc_client.scanning.DetectorType] | list[str] = [grpc_client.scanning.DetectorType.BF],
    is_precession_enabled=False,
    is_cross_elimination_enabled=False,
    camera_exposure=0,
    is_beam_blanked=False,
):
    """Start a rectangle scan.

    pixel_time in seconds
    rectangle:[top,left,height,width] or None, must be withing the total_size parameter
    """
    number_of_frames = frames
    scan_field_number_of_pixels = total_size
    if rectangle is None:
        width = total_size
        height = total_size
        left = 0
        top = 0
    else:
        top, left, height, width = rectangle

    enabled_detectors = []
    for detector in detectors:
        if isinstance(detector, str):
            enabled_detectors.append(grpc_client.scanning.DetectorType[detector])
        else:
            enabled_detectors.append(detector)

    scan_id = grpc_client.scanning.start_rectangle_scan(
        number_of_frames,
        width,
        height,
        left,
        top,
        pixel_time,
        scan_field_number_of_pixels,
        enabled_detectors,
        is_precession_enabled,
        is_cross_elimination_enabled,
        camera_exposure,
        is_beam_blanked,
    )
    if not scan_id["scan_id"]:
        return None
    try:
        return int(scan_id["scan_id"])
    except:
        return None


def start_multi_series(
    pixel_time: float = 300e-9,
    total_size: int = 1024,
    frames: int = 1,
    rectangle=None,
    detectors: list[grpc_client.scanning.DetectorType] = [grpc_client.scanning.DetectorType.BF],
    tilt_factors=np.array,
    camera_exposure=0,
    is_beam_blanked=False,
):
    number_of_frames = frames
    scan_field_number_of_pixels = total_size
    if rectangle is None:
        width = total_size
        height = total_size
        left = 0
        top = 0
    else:
        top, left, height, width = rectangle

    enabled_detectors = detectors

    tilt_factors_prepared = [{"x": tf[0] * 1.0, "y": tf[1] * 1.0} for tf in tilt_factors]

    scan_id = grpc_client.scanning.start_tilt_series(
        number_of_frames,
        width,
        height,
        left,
        top,
        pixel_time,
        scan_field_number_of_pixels,
        enabled_detectors,
        tilt_factors_prepared,
        camera_exposure,
        is_beam_blanked,
    )
    if not scan_id["scan_id"]:
        return None
    try:
        return int(scan_id["scan_id"])
    except:
        return None


def get_scanning_shift_and_transform():
    xy_e = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)  # TODO various types
    xy_electronic = np.array([xy_e["x"], xy_e["y"]]) * 1e6  # to um
    xy_s = grpc_client.stage.get_x_y()
    xy_stage = np.array([xy_s["x"], xy_s["y"]]) * 1e6  # to um

    rotation = grpc_client.scanning.get_rotation()
    transform2x2 = np.array(grpc_client.stage.get_optics_to_sample_projection()).reshape(3, 3)[:2, :2]

    return xy_electronic, xy_stage, rotation, transform2x2


previous_shift_type = "stage"


def set_combined_shift(
    shift_xy,
    last_electronic,
    last_stage,
    transform2x2,
    mode="combined",
    electronic_limit=1,
    use_future=True,
    stage_nowait=False,
    alternative_stage_xy_set=None,
    stage_auto_stop=True,
):
    """Set combined electronic and stage shift.

    shift_xy in scanning plane needs to be already rotated by scan rotation
    mode: electronic combined stage
    """
    global previous_shift_type

    use_stage = True
    future = None

    new_electronic = last_electronic
    new_stage = last_stage

    if np.all(np.abs(last_electronic + shift_xy) < electronic_limit) and mode in {"electronic", "combined"}:
        new_electronic = last_electronic + shift_xy
        if previous_shift_type == "stage":
            if stage_auto_stop:
                grpc_client.stage.stop()
        try:
            grpc_client.illumination.set_shift(
                {"x": new_electronic[0] * 1e-6, "y": new_electronic[1] * 1e-6},
                grpc_client.illumination.DeflectorType.Scan,
            )
            previous_shift_type = "electronic"
            use_stage = False
        except:
            pass
    if use_stage and mode != "electronic":
        xy_e_to_stage = np.dot(transform2x2, last_electronic + shift_xy)
        grpc_client.illumination.set_shift({"x": 0.0, "y": 0.0}, grpc_client.illumination.DeflectorType.Scan)
        if alternative_stage_xy_set is None:
            if use_future:
                future = grpc_client.stage.set_x_y.future(
                    (xy_e_to_stage[0] + last_stage[0]) * 1e-6,
                    (xy_e_to_stage[1] + last_stage[1]) * 1e-6,
                    stage_nowait,
                    True,
                    True,
                )
            else:
                grpc_client.stage.set_x_y(
                    (xy_e_to_stage[0] + last_stage[0]) * 1e-6,
                    (xy_e_to_stage[1] + last_stage[1]) * 1e-6,
                    stage_nowait,
                    True,
                    True,
                )
        else:
            alternative_stage_xy_set(
                (xy_e_to_stage[0] + last_stage[0]) * 1e-6, (xy_e_to_stage[1] + last_stage[1]) * 1e-6
            )
        new_electronic = np.zeros(2)
        new_stage = xy_e_to_stage + last_stage

        previous_shift_type = "stage"

    return new_stage, new_electronic, future, use_stage
