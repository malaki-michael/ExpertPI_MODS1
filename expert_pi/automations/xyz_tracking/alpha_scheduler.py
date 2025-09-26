import numpy as np
from time import sleep
from . import stage_shifting, objects
from .. import settings
import matplotlib.pyplot as plt


def get_t_form_x(x, x0, v0, a):
    if a == 0:
        tm = (x - x0)/v0
    else:
        D = v0**2 - 2*a*(x0 - x)

        tm_p = (-v0 + np.sqrt(D[D >= 0]))/a
        tm_n = (-v0 - np.sqrt(D[D >= 0]))/a

        tm_minimum = np.minimum(tm_p, tm_n)
        tm = np.maximum(tm_p, tm_n)
        tm[tm_minimum >= 0] = tm_minimum[tm_minimum >= 0]

        if len(tm) == 0 and np.abs(D) < 1e-10:  # rounding error:
            tm = np.array([-v0/a])

    return tm


def move_to(x0, v0, xe, ve, v_max, a_max):
    v3 = ve
    x3 = xe

    for vm in [-v_max, v_max]:
        for a in [-a_max, a_max]:
            d = a

            t1 = (vm - v0)/a
            t2 = -x0/vm + x3/vm + 0.5*v3**2/d/vm - 0.5*vm/d + 0.5*v0**2/a/vm - 0.5*vm/a
            t3 = (vm - v3)/d

            if t1 >= 0 and t2 >= 0 and t3 >= 0:
                x1 = x0 + v0*t1 + 0.5*a*t1**2
                x2 = x1 + vm*t2

                xs = [x0, x1, x2]
                vs = [v0, vm, vm]
                accs = [a, 0, -d]
                ts = [t1, t2, t3]
                return ts, xs, vs, accs

    # no solution at maximal speed - we just accelerate and decellerate:
    v2 = ve
    x2 = xe

    a = a_max
    D = -a*x0 + a*x2 + 0.5*v0**2 + 0.5*v2**2
    if -a*x0 + a*x2 + 0.5*v0**2 + 0.5*v2**2 < 0:
        a = -a_max
        D = -a*x0 + a*x2 + 0.5*v0**2 + 0.5*v2**2
    if -a*x0 + a*x2 + 0.5*v0**2 + 0.5*v2**2 < 0:
        # no able to reach the position crank up the acceleration in that case
        print("warning increasing acceleration to", a)
        a = (0.5*v0**2 + 0.5*v2**2)/(x2 - x0)
        D = 0

    t1 = max((- v0 + np.sqrt(D))/a, (- v0 - np.sqrt(D))/a)
    t2 = max((- v2 + np.sqrt(D))/a, (- v2 - np.sqrt(D))/a)

    x1 = x0 + v0*t1 + 0.5*a*t1**2
    v1 = v0 + a*t1

    xs = [x0, x1]
    vs = [v0, v1]
    accs = [a, -a]
    ts = [t1, t2]

    return ts, xs, vs, accs


def check_validity(ts, xs, vs, accs, v_diff=1e-6, x_diff=1e-6):
    xs2 = xs + vs*ts + 0.5*accs*ts**2
    vs2 = vs + accs*ts

    mask = np.abs(np.abs(xs2[:-1] - xs[1:]) > x_diff)
    if np.any(mask):
        print("x is not smooth:")
        print("x start:", xs[1:][mask])
        print("x end:", xs2[:-1][mask])

    mask = np.abs(vs2[:-1] - vs[1:]) > v_diff
    if np.any(mask):
        print("v is not smooth:")
        print("v start:", vs[1:][mask])
        print("v end:", vs2[:-1][mask])


def generate_xs(ts, xs, vs, accs, ts_checkpoints, xs_checkpoints, dt=0.05):
    ts_result = [0]
    xs_result = [xs[0]]
    if len(ts_checkpoints) > 0:
        mask = [ts_checkpoints[0] == 0]
        if ts_checkpoints[0] == 0:
            ts_checkpoints = ts_checkpoints[1:]
    else:
        mask = False

    t = 0
    x = xs[0]
    v = vs[0]
    a = accs[0]

    ts_cum = np.cumsum(ts)

    max_t = np.sum(ts)

    i = 0

    for k, t_check in enumerate(ts_checkpoints):
        tis = np.linspace(t, t_check, num=max(2, int(np.round((t_check - t)/dt))))
        for j in range(1, len(tis)):
            ti = tis[j]
            while ti > ts_cum[i]:
                i += 1
                x = xs[i]
                v = vs[i]
                a = accs[i]

            if i == 0:
                t_rel = ti
            else:
                t_rel = ti - ts_cum[i - 1]
            ts_result.append(ti)

            if j == len(tis) - 1:
                xs_result.append(xs_checkpoints[k])  # assign directly xs_checkpoint to prevent rounding issues.
                mask.append(True)
            else:
                xs_result.append(x + v*t_rel + 0.5*a*t_rel**2)
                mask.append(False)
        t = t_check

    if max_t > t:
        tis = np.linspace(t, max_t, num=max(2, int(np.round((max_t - t)/dt))))
        for j in range(1, len(tis)):
            ti = tis[j]
            while ti > ts_cum[i]:  # limit due to rounding
                i += 1
                x = xs[i]
                v = vs[i]
                a = accs[i]

            if i == 0:
                t_rel = ti
            else:
                t_rel = ti - ts_cum[i - 1]
            ts_result.append(ti)
            xs_result.append(x + v*t_rel + 0.5*a*t_rel**2)
            mask.append(False)

    return np.array(ts_result), np.array(xs_result), np.array(mask)


def create_alpha_positions(measured_alpha_positions, dt=0.05, alpha_speed=20, alpha_acceleration=10, initial_speed=None, initial_alpha=0, plot=False):
    """measured_alpha_positions needs to be ordered from lowest to highest"""

    if initial_speed is None:
        da = measured_alpha_positions[0] - initial_alpha
        initial_speed = np.sign(da)*min(np.sqrt(np.abs(da)*2*alpha_acceleration), alpha_speed)

    t0s, x0s, v0s, acc0s = move_to(initial_alpha, initial_speed, measured_alpha_positions[0], 0, alpha_speed, alpha_acceleration)
    t1s, x1s, v1s, acc1s = move_to(measured_alpha_positions[0], 0, measured_alpha_positions[-1], 0, alpha_speed, alpha_acceleration)
    t2s, x2s, v2s, acc2s = move_to(measured_alpha_positions[-1], 0, initial_alpha, initial_speed, alpha_speed, alpha_acceleration)

    ts = np.concatenate([t0s, t1s, t2s])
    xs = np.concatenate([x0s, x1s, x2s])
    vs = np.concatenate([v0s, v1s, v2s])
    accs = np.concatenate([acc0s, acc1s, acc2s])

    check_validity(ts, xs, vs, accs)

    # assign t to alpha measured
    t1_start_total = np.sum(t0s)
    tms = []

    t1s_cumsum = np.cumsum(t1s)

    i = 0
    for x in measured_alpha_positions:
        while True:
            t = get_t_form_x(np.array([x]), x1s[i], v1s[i], acc1s[i])[0]
            if t > t1s[i]:
                t1_start_total += t1s[i]
                i += 1
            else:
                tms.append(t + t1_start_total)
                break
            if i >= len(t1s):
                raise Exception("measured position " + str(x) + " exceeded time " + str(t))

    ts, xs, mask = generate_xs(ts, xs, vs, accs, tms, measured_alpha_positions, dt=dt)

    if plot:
        plt.figure()
        plt.plot(ts, xs, "-o")
        plt.plot(ts[mask], xs[mask], "s")
        plt.show()

    return ts, xs, mask


def drive_open_cycle(amplitude, max_time, dt=0.05, repeats=1,
                     is_running_callback=None, finish_callback=None, correction_model=None):
    """amplitude in degrees"""
    a = np.linspace(0, 1, num=int(max_time/dt))
    alphas = np.repeat(-amplitude*np.sin(2*np.pi*a), repeats)

    if correction_model is not None:
        corrections = correction_model(alphas)
    else:
        corrections = np.zeros((len(alphas), 3))

    for i, alpha in enumerate(alphas):
        if is_running_callback is not None:
            running, paused = is_running_callback(alpha, i, len(alphas))
            if not running:
                break
            while paused:
                sleep(0.1)
                running, paused = is_running_callback(alpha, i, len(alphas))
                if not running:
                    break
            if not running:
                break
        alpha_rad = alpha/180*np.pi

        stage_shifting.schedule_alpha(alpha_rad, corrections[i, 0], corrections[i, 1], corrections[i, 2])
        sleep(dt)
    if finish_callback is not None:
        finish_callback()


def measure_positions(measured_alpha_positions,
                      is_running_callback=None, finish_callback=None, correction_model=None,skip_measure=False):
    ts, alphas, mask = create_alpha_positions(measured_alpha_positions, dt=settings.alpha_dt, alpha_speed=settings.alpha_speed, alpha_acceleration=settings.alpha_acceleration)

    if skip_measure:
        mask[:]=False

    if correction_model is not None:
        corrections = correction_model(alphas)
    else:
        corrections = np.zeros((len(alphas), 3))

    i = 0

    measured_i = 0
    total_measured = np.sum(mask)

    stage_shifting.last_stabilization_alpha = None

    while i < len(alphas):
        alpha = alphas[i]
        if is_running_callback is not None:
            running, paused = is_running_callback(alpha, measured_i, total_measured)
            if not running:
                break
            while paused:
                sleep(0.1)
                running, paused = is_running_callback(alpha, measured_i, total_measured)
                if not running:
                    break
            if not running:
                break
        alpha_rad = alpha/180*np.pi

        if mask[i]:
            stage_shifting.schedule_alpha(alpha_rad, corrections[i, 0], corrections[i, 1], corrections[i, 2], True)
        else:
            stage_shifting.schedule_alpha(alpha_rad, corrections[i, 0], corrections[i, 1], corrections[i, 2], False)

        if i == 0:
            dt = ts[i]
        else:
            dt = ts[i] - ts[i - 1]
        sleep(dt)
        if mask[i]:
            measured_i = np.sum(mask[:i + 1])

            def condition():
                running, paused = is_running_callback(alpha, measured_i, total_measured)
                if stage_shifting._motion_type == objects.MotionType.acquired or not running:
                    return True
                else:
                    return False

            with stage_shifting.motion_type_condition:
                stage_shifting.motion_type_condition.wait_for(condition)
                running, paused = is_running_callback(alpha, measured_i, total_measured)
                if not running:
                    break
                if paused:
                    i -= 1  # set the same alpha step one more time when we unpause
                stage_shifting._motion_type = objects.MotionType.moving
        i += 1
    if finish_callback is not None:
        finish_callback()
