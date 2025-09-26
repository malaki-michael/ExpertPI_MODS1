import numpy as np
from scipy.optimize import minimize

from expert_pi.app import scan_helper
from expert_pi.grpc_client import illumination
from expert_pi.grpc_client.modules.scanning import DetectorType
from expert_pi.measurements import kernels
from expert_pi.stream_clients import CacheClient


def autofocus(
    contrast_function,
    set_function,
    interval,
    max_iterations=6,
    direction=1,
    n=7,
    iteration_factor=0.4,
    output=False,
    overstep=True,
    overstep_factor=2,
    contrast_function_parameters={},
):
    if output:
        print("-----", interval, direction)

    xs = np.linspace(interval[0], interval[1], num=n)[::direction]

    values = []
    for x in xs:
        set_function(x)
        c = contrast_function(**contrast_function_parameters)
        values.append(c)
        if output:
            print(f"{x:8.3f}: {c:10.7f}")

    i = np.argmax(values)
    if overstep:
        dx = xs[1] - xs[0]
        width = interval[1] - interval[0]
        max_interval = [
            np.mean(interval) - width / 2 * overstep_factor,
            np.mean(interval) + width / 2 * overstep_factor,
        ]
        while np.argmax(values) == len(values) - 1:
            x += dx
            if max_interval[0] > x or max_interval[1] < x:
                print("overstep exceed ", max_interval[0], x, max_interval[1])
                break
            set_function(x)
            xs = np.concatenate((xs, [x]))
            c = contrast_function(**contrast_function_parameters)
            values.append(c)
            if output:
                print(f"overstep {x:8.3f}: {c:10.7f}")

    data = [[xs[i], values[i]] for i in range(len(values))]
    i = np.argmax(values)

    if i == len(values) - 1:
        set_function(xs[i])
        return xs[i], data

    elif i == 0:
        if overstep:
            width = np.abs(xs[-1] - xs[0])
            new_interval = (xs[0] - width / 2, xs[0] + width / 2)
            direction *= -1
            x, data2 = autofocus(
                contrast_function,
                set_function,
                new_interval,
                max_iterations=max_iterations,
                direction=direction,
                n=n,
                output=output,
                contrast_function_parameters=contrast_function_parameters,
                overstep_factor=overstep_factor,
            )
            return x, data + data2
        else:
            set_function(xs[0])
            return xs[0], data
    else:
        width = np.abs(xs[-1] - xs[0])
        new_interval = (xs[i] - width / 2 * iteration_factor, xs[i] + width / 2 * iteration_factor)
        direction *= -1
        if max_iterations > 1:
            x, data2 = autofocus(
                contrast_function,
                set_function,
                new_interval,
                max_iterations=max_iterations - 1,
                direction=direction,
                n=n,
                output=output,
                contrast_function_parameters=contrast_function_parameters,
                overstep_factor=overstep_factor,
            )
            return x, data + data2
        else:
            return xs[i], data


class StopException(Exception):
    pass


def focus_by_C3(
    cache_client: CacheClient,
    defocus_init=10e-9,
    detector=DetectorType.BF,
    kernel="sobel",
    blur="median",
    n=256,
    rectangle=None,
    pixel_time=300e-9,
    max_iterations=6,
    steps=7,
    output=False,
    stop_callback=lambda: False,
    set_callback=lambda x: False,
    is_precession_enabled=False,
):
    initial = illumination.get_condenser_defocus(type=illumination.CondenserFocusType.C3)

    w, h = n, n
    if rectangle is not None:
        w, h = rectangle[2], rectangle[3]

    def contrast_function():
        if stop_callback():
            raise StopException("stopped")
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=pixel_time,
            total_size=n,
            rectangle=rectangle,
            detectors=[detector],
            is_precession_enabled=is_precession_enabled,
        )
        _header, data = cache_client.get_item(scan_id, int(w * h))  # use int instead of int32 for json seriability
        img = data["stemData"][detector.name].reshape(w, h)
        return kernels.contrast_function(img, kernel=kernel, blur=blur)

    def set_function(value):
        illumination.set_condenser_defocus(initial + value, type=illumination.CondenserFocusType.C3)
        set_callback(initial + value)

    result = autofocus(
        contrast_function,
        set_function,
        [-defocus_init, defocus_init],
        max_iterations=max_iterations,
        n=steps,
        output=output,
    )
    set_function(result[0])

    if np.abs(result[0]) >= defocus_init:
        print("autofocus did not converge properly", result[0], "not inside", [-defocus_init, defocus_init])
    return result


#
# def focus_by_stage(defocus_init=50, detector='BF', kernel='sobel', blur='median', N=256, rectangle=None, pixel_time=300, max_iterations=6, steps=7, output=True,
#                    continue_acquisition=True, plot=True, use_coords='stage', overstep=True, overstep_factor=2, alpha_correction=None):
#     initial = stage.get_values(requested=False)
#
#     def contrast_function():
#         img = xy_calibration.scan_stem_image(N=N, rectangle=rectangle, pixel_time=pixel_time, detector=detector, output=False)
#         return kernels.contrast_function(img, kernel=kernel, blur=blur)
#
#     if use_coords == 'stage':
#         if alpha_correction is None:
#             def set_function(value):
#                 stage.set_values([initial[0], initial[1] + value, initial[2], initial[3], initial[4]], skip_axes=[True, False, True, True, True])
#                 stage.wait_till_on_target()
#         else:
#             def set_function(value):
#                 rot_y = value*np.cos(alpha_correction)
#                 rot_z = -value*np.sin(alpha_correction)
#                 stage.set_values([initial[0], initial[1] + rot_y, initial[2] + rot_z, initial[3], initial[4]], skip_axes=[True, False, False, True, True])
#                 stage.wait_till_on_target()
#     elif use_coords == 'sample':
#         def set_function(value):
#             sample.z.set(value, wait=True)
#     else:
#         raise ValueError(f"Unknown 'use_coords' parameter: {use_coords}")
#
#     result = autofocus(contrast_function, set_function, [-defocus_init, defocus_init], max_iterations=max_iterations, N=steps, output=output, overstep=overstep, overstep_factor=overstep_factor)
#     set_function(result[0])
#
#     if np.abs(result[0]) >= defocus_init:
#         print('autofocus did not converge properly', result[0], 'not inside', [-defocus_init, defocus_init])
#
#     if continue_acquisition:
#         acquisition.start_acquisition(N, [0, 0, N, N], pixel_time, stem_detectors=[detector], frame_count=0)
#
#     if plot:
#         plt.figure()
#         plt.plot([r[0] for r in result[1]], [r[1] for r in result[1]], '-o')
#         plt.xlabel('Z stage (At)')
#
#     stage.stop()
#     sample.update_from_microscope()
#
#     return result[0]
#
#


def auto_stigmation(
    cache_client: CacheClient,
    init_step=20.0,
    detector=DetectorType.BF,
    kernel="sobel",
    blur="median",
    n=64,
    rectangle=None,
    pixel_time=300e-9,
    max_iterations=40,
    stop_callback=lambda: False,
    set_callback=lambda x: False,
    is_precession_enabled=False,
):
    # init_step in um

    stigmator = illumination.get_stigmator()
    init_astigmatism = np.array([stigmator["x"], stigmator["y"]]) * 1e3

    init_simplex = init_step * np.array([[1, 0], [1 / np.sqrt(2), -1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)]])

    w, h = n, n
    if rectangle is not None:
        w, h = rectangle[2], rectangle[3]

    def contrast_function():
        if stop_callback():
            raise StopException("stopped")
        scan_id = scan_helper.start_rectangle_scan(
            pixel_time=pixel_time,
            total_size=n,
            rectangle=rectangle,
            detectors=[detector],
            is_precession_enabled=is_precession_enabled,
        )
        _header, data = cache_client.get_item(scan_id, int(w * h))  # use int instead of int32 for json seriability

        img = data["stemData"][detector.name].reshape(w, h)
        return kernels.contrast_function(img, kernel=kernel, blur=blur)

    def set_function(value):
        set_value = init_astigmatism + value
        illumination.set_stigmator(set_value[0] * 1e-3, set_value[1] * 1e-3)
        set_callback(set_value)

    def minimize_function(x):
        set_function(x)
        contrast = contrast_function()
        # data.append([x,contrast])
        return -contrast

    result = minimize(
        minimize_function,
        np.zeros(2),
        method="Nelder-Mead",
        options={"initial_simplex": init_simplex, "maxiter": max_iterations, "maxfev": int(2.5 * max_iterations)},
    )
    set_function(result.x)

    return result.x
