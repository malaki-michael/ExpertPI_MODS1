import numpy as np
from . import objects
from .. import settings
from . import stage_shifting

actual_errors = None
actual_errors_integral = None

actual_PI = None
actual_PI_type = None


def change_PI_term(PI):  # will recalculate live integral term appropriately
    global actual_errors_integral, actual_PI
    factor = np.nan_to_num(settings.PI/PI)
    actual_errors_integral = factor*actual_errors_integral  # TODO add thread lock?
    actual_PI = PI


def PID_stage_offset(input: objects.NodeImage) -> objects.NodeImage:
    global actual_errors_integral, actual_errors, actual_PI_type

    target_z_offset = settings.target_z_offset
    target_xy_offset = settings.target_xy_offset

    input.target_xy_offset = target_xy_offset
    input.target_z_offset = target_z_offset

    errors = target_xy_offset - np.nan_to_num(np.dot(input.transform2x2, input.offset_xy))
    if input.offset_z is not None:
        errors = np.concatenate([errors, [target_z_offset - np.nan_to_num(input.offset_z)]])

    input.errors = errors*1

    if actual_errors_integral is not None:
        motion_type = stage_shifting._motion_type  # move to separate variable for preventing thread issues
        if not settings.enable_regulation and (motion_type == objects.MotionType.moving or motion_type == objects.MotionType.stabilizing):
            input.errors_integral = actual_errors_integral = actual_errors_integral*1
        else:
            input.errors_integral = actual_errors_integral = errors*1 + actual_errors_integral*1
    else:
        input.errors_integral = actual_errors_integral = errors*1

    if actual_errors is not None:
        input.errors_difference = errors - actual_errors
    else:
        input.errors_difference = errors*1

    actual_errors = errors*1
    motion_type = stage_shifting._motion_type
    if actual_PI_type is None or actual_PI_type != motion_type:
        actual_PI_type = motion_type
        if motion_type == objects.MotionType.moving:
            change_PI_term(settings.PI_moving)
        else:
            change_PI_term(settings.PI)

    input.regulated_xyz_shift = settings.P*input.errors + settings.PI*input.errors_integral + settings.PD*input.errors_difference

    input.total_error = np.sqrt(np.sum(errors**2))
    return input
