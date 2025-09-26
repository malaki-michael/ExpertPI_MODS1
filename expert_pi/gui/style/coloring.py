import numpy as np
from numba import njit


# use this for real time - vizualization of the shift of two images
@njit
def get_colored_differences(img0, img1, a=1.7, b=0.6):
    # TODO optimilize this further:
    s = 0.5 * (img1 * 1.0 + img0)

    s_max = np.max(s)
    if s_max != 0:
        s = (s / np.max(s) * b + (1 - b) / 2) * 255
    else:
        s += (1 - b) / 2 * 255
    dp = np.maximum(0, (img0 * 1.0 - img1 * 1.0))

    dp_max = np.max(dp)
    if dp_max != 0:
        dp = np.clip(1 - a * dp / dp_max, 0, 1)

    dm = np.maximum(0, (img1 * 1.0 - img0 * 1.0))
    dm_max = np.max(dm)
    if dm_max != 0:
        dm = np.clip(1 - a * dm / dm_max, 0, 1)

    result = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            result[i, j, 0] = s[i, j] * dp[i, j]
            result[i, j, 1] = s[i, j] * dp[i, j] * dm[i, j]
            result[i, j, 2] = s[i, j] * dm[i, j]
    return result


# used in colored slider
color_done = "#009900"
color_moving = "#cc0000"
color_acquiring = "#cc8800"
color_todo = "#555555"
