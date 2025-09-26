import numpy as np
from .. import settings
from expert_pi.app import scan_helper
# from ...stream_clients import cache_client
from ... import grpc_client
import time
from . import objects
from time import perf_counter
from . import stage_shifting


def acquire_image(fov, total_size, pixel_time, rectangle=None):
    """fov in um, pixel_time in ns"""
    grpc_client.scanning.set_field_width(fov*1e-6)

    if rectangle is None:
        width = total_size
        height = total_size
    else:
        top, left, height, width = rectangle

    scan_id = scan_helper.start_rectangle_scan(pixel_time*1e-9, total_size, frames=1, rectangle=rectangle)
    header, data = cache_client.get_item(int(scan_id), height*width)
    try:
        return data["stemData"]["BF"].reshape(height, width)
    except:
        import traceback
        traceback.print_exc()
        print(header,data)


def acquire_image_tilt_series(fov, total_size, pixel_time, tilt_factors, rectangle=None):
    """fov in um, pixel_time in ns"""
    grpc_client.scanning.set_field_width(fov*1e-6)
    if rectangle is None:
        width = total_size
        height = total_size
    else:
        top, left, height, width = rectangle

    scan_id = scan_helper.start_multi_series(pixel_time*1e-9, total_size, frames=1, tilt_factors=tilt_factors, rectangle=rectangle)
    header, data = cache_client.get_item(int(scan_id), len(tilt_factors)*height*width)

    return data["stemData"]["BF"].reshape(len(tilt_factors), height, width)


measurement_acquisition = None


def acquire_next(input: objects.NodeImage) -> objects.NodeImage:
    """error in um from previous step to choose properly fov"""
    measure_at_this_step = False
    with stage_shifting.motion_type_condition:
        if stage_shifting._motion_type == objects.MotionType.acquiring and stage_shifting._motion_type_last_id <= input.id:
            measure_at_this_step = True
    if measure_at_this_step:
        measurement_acquisition(input)
        stage_shifting.change_motion_type(objects.MotionType.acquired, input.id)

    chosen_fov = input.next_fov
    scale = settings.reference_N/settings.tracking_N
    start_time = perf_counter()

    output = objects.NodeImage(input.id + 1, input.serie_id, None, chosen_fov, settings.pixel_time)

    output.reference_N = settings.reference_N

    if settings.use_rectangle_selection:
        i = np.where(settings.allowed_fovs == input.next_fov)[0][0]

        output.rectangle_selection = True
        output.fov_total = settings.fovs_total[i]
        output.total_pixels = settings.total_pixels[i]
        output.rectangle = settings.rectangles[i]
        output.center = settings.centers[i]



    else:
        output.rectangle_selection = True
        output.fov_total = input.next_fov
        output.total_pixels = settings.reference_N
        output.rectangle = np.array([0, 0, settings.reference_N, settings.reference_N])
        output.center = np.array([0, 0])

    if scale > 1:
        w, h = output.rectangle[2:4]
        rectangle = [int(output.rectangle[0] + w//2 - w//2//scale),
                     int(output.rectangle[1] + h//2 - h//2//scale),
                     int(w//scale), int(h//scale)]
        output.rectangle = rectangle

    if settings.tilt_for_z is None:
        img = acquire_image(output.fov_total, output.total_pixels, settings.pixel_time, rectangle=output.rectangle)
    else:
        angle = np.sqrt(settings.tilt_for_z[0]**2 + settings.tilt_for_z[1]**2)
        grpc_client.scanning.set_precession_angle(angle*1e-3)
        grpc_client.scanning.set_precession_frequency(0)
        img = acquire_image_tilt_series(output.fov_total, output.total_pixels, settings.pixel_time, np.array([[0, 0], settings.tilt_for_z/angle]), rectangle=output.rectangle)

    output.image = img

    end_time = perf_counter()

    output.scale = scale

    # PID data
    output.previous_errors = input.errors
    output.previous_errors_integral = input.errors_integral
    output.previous_errors_difference = input.errors_difference

    output.acquisition_time_counters = [start_time, end_time]
    output.shift_electronic, output.stage_xy, output.rotation, output.transform2x2 = scan_helper.get_scanning_shift_and_transform()
    output.stage_z = grpc_client.stage.get_z()*1e6
    output.stage_ab = np.array([grpc_client.stage.get_alpha(), grpc_client.stage.get_beta()])

    if len(img.shape) == 3:
        output.tilt_angle = settings.tilt_for_z*1

    return output
