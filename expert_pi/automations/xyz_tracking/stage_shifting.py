from . import objects
from .. import settings
import numpy as np
from ... import grpc_client
from expert_pi.app import scan_helper
from threading import Lock, Condition
import time
from functools import partial

data = []


def add_datapoint(input: objects.NodeImage):
    data.append(input)


next_alpha = None
last_stabilization_alpha = None
next_alpha_lock = Lock()
alpha_xyz_correction = np.zeros(3)
manual_xyz_correction = np.zeros(3)
manual_fov = None

motion_type_condition = Condition()
_motion_type = objects.MotionType.moving
_motion_type_timer = None
_motion_type_last_id = None

_use_beta_instead=False #debug


def change_motion_type(new_motion_type, id=None):
    global _motion_type, _motion_type_timer, _motion_type_last_id
    # print(new_motion_type, id)
    with motion_type_condition:
        if _motion_type != new_motion_type:
            _motion_type_timer = time.monotonic()
        if id is not None:
            _motion_type_last_id = id
        _motion_type = new_motion_type
        motion_type_condition.notify()


def schedule_alpha(alpha, dx, dy, dz, acquire_at_this_step=False):
    global next_alpha, last_stabilization_alpha
    with next_alpha_lock:
        # print("next", alpha)
        alpha_xyz_correction = np.array([dx, dy, dz])
        fast_step = False
        if acquire_at_this_step:
            if settings.stabilization_alpha_skip is not None:
                if last_stabilization_alpha is not None and np.abs(last_stabilization_alpha - alpha) < settings.stabilization_alpha_skip/180*np.pi:
                    fast_step = True
                else:
                    last_stabilization_alpha = alpha
        next_alpha = alpha, acquire_at_this_step, fast_step


def compensate_stage(input: objects.NodeImage) -> objects.NodeImage:
    global next_alpha, _motion_type

    fovs = settings.allowed_fovs[settings.allowed_fovs > np.sum(settings.fov_factor[:len(input.errors)]*np.sqrt(input.errors**2))]

    with motion_type_condition:
        motion_type_actual = _motion_type  # copy to separate variable due to multithreading
        motion_type_last_id = _motion_type_last_id
        motion_type_timer = _motion_type_timer

    if motion_type_actual == objects.MotionType.moving:
        fovs = fovs[fovs > settings.minimum_fov_for_moving]

    if len(fovs) == 0 or np.any(input.confidence_xy < 0):
        input.next_fov = settings.allowed_fovs[0]
    else:
        input.next_fov = fovs[-1]

    if manual_fov is not None:
        input.next_fov = settings.allowed_fovs[np.argmin(np.abs(settings.allowed_fovs - manual_fov))]

    input.next_xy = input.reference_object.stage_xy + input.reference_object.shift_electronic + input.regulated_xyz_shift[:2] - input.shift_electronic
    if len(input.regulated_xyz_shift) == 3:
        input.next_z = input.reference_object.stage_z + input.regulated_xyz_shift[2]

    alpha = None
    xyz_correction = manual_xyz_correction*1
    acquire_at_this_step = False
    fast_step = False
    with next_alpha_lock:
        xyz_correction += alpha_xyz_correction*1
        if next_alpha is not None:
            alpha = next_alpha[0]
            acquire_at_this_step = next_alpha[1]
            fast_step = next_alpha[2]
            change_motion_type(objects.MotionType.moving)
            motion_type_actual = objects.MotionType.moving
        next_alpha = None

    input.xyz_correction = xyz_correction

    # TODO support scan rotation
    if motion_type_actual == objects.MotionType.moving or motion_type_actual == objects.MotionType.stabilizing:
        if settings.enable_regulation:
            if input.next_z is None:
                input.next_z = input.reference_object.stage_z
            if alpha is None:
                dxy = input.next_xy[0:2] + xyz_correction[0:2] - input.stage_xy  # electronic shift already substracted

                dxy_scanning = np.linalg.solve(input.transform2x2, dxy)  # back to scanning plane

                new_stage, new_electronic, future, use_stage = scan_helper.set_combined_shift(dxy_scanning, input.shift_electronic, input.stage_xy, input.transform2x2,
                                                                                              mode=settings.shift_mode, electronic_limit=settings.electronic_shift_limit,
                                                                                              alternative_stage_xy_set=partial(grpc_client.stage.set_x_y_z, z=(input.next_z + xyz_correction[2])*1e-6),
                                                                                              stage_auto_stop=False)

                if not use_stage:
                    # TODO use electronic z as well
                    grpc_client.stage.set_z((input.next_z + xyz_correction[2])*1e-6, True, True, True)


            else:
                # print("abc", alpha)
                grpc_client.illumination.set_shift({"x": input.reference_object.shift_electronic[0]*1e-6, "y": input.reference_object.shift_electronic[1]*1e-6},
                                                   grpc_client.illumination.DeflectorType.Scan)
                tilts=[alpha,input.stage_ab[1]]
                if _use_beta_instead:
                    tilts=[input.stage_ab[0],alpha]


                grpc_client.stage.set_x_y_z_a_b((input.next_xy[0] + xyz_correction[0])*1e-6,
                                                (input.next_xy[1] + xyz_correction[1])*1e-6,
                                                (input.next_z + xyz_correction[2])*1e-6,
                                                tilts[0],
                                                tilts[1])  # will be skipped for single tilt holder
        elif alpha is not None:
            tilts=[alpha,input.stage_ab[1]]
            if _use_beta_instead:
                tilts=[input.stage_ab[0],alpha]
                
            grpc_client.illumination.set_shift({"x": input.reference_object.shift_electronic[0]*1e-6, "y": input.reference_object.shift_electronic[1]*1e-6},
                                               grpc_client.illumination.DeflectorType.Scan)
            grpc_client.stage.set_x_y_z_a_b((input.reference_object.stage_xy[0] + xyz_correction[0])*1e-6,
                                            (input.reference_object.stage_xy[1] + xyz_correction[1])*1e-6,
                                            (input.reference_object.stage_z + xyz_correction[2])*1e-6,
                                            tilts[0],
                                            tilts[1])  # will be skipped for single tilt holder

    if acquire_at_this_step:
        if fast_step:
            change_motion_type(objects.MotionType.acquiring, input.id)
        else:
            change_motion_type(objects.MotionType.stabilizing, input.id)

    if motion_type_actual == objects.MotionType.stabilizing and np.all(np.abs(input.errors) < settings.allowed_error) and \
            (motion_type_last_id is None or input.id > motion_type_last_id):
        # grpc_client.stage.stop()
        # print("changed to waiting", input.errors, settings.allowed_error, np.abs(input.errors) < settings.allowed_error)
        change_motion_type(objects.MotionType.waiting, input.id)

    elif motion_type_actual == objects.MotionType.waiting and (np.any(np.abs(input.errors) > settings.allowed_error) or np.any(input.confidence_xy < 0)):
        grpc_client.stage.stop()  # TODO make sure that alpha is on Target
        # print("changed back to stabilization", input.errors, settings.allowed_error, np.abs(input.errors) < settings.allowed_error)
        change_motion_type(objects.MotionType.stabilizing, input.id)

    elif motion_type_actual == objects.MotionType.waiting and time.perf_counter() - motion_type_timer > settings.stabilization_time and \
            (motion_type_last_id is None or input.id > motion_type_last_id):
        change_motion_type(objects.MotionType.acquiring, input.id)

    input.motion_type = motion_type_actual
    input.motion_type_timer = motion_type_timer

    return input

#
# def wait_for_motion_type(required_type, caller_name=None):
#     global motion_type
#     with motion_type_event:
#         print(caller_name, "test", required_type.name)
#         motion_type_event.wait()
#         while motion_type != required_type:
#             print(caller_name, "waiting for", required_type.name)
#             motion_type_event.wait()
