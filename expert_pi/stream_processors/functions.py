import cv2
import numpy as np
import scipy.signal
import cv2 as cv
import numba

WINDOWS = {
    "hanning": np.hanning,
    "hamming": np.hamming,
    "bartlett": np.bartlett,
    "blackman": np.blackman,
    "barthann": scipy.signal.windows.barthann,
    "bohman": scipy.signal.windows.bohman,
    "boxcar": scipy.signal.windows.boxcar,
}

precalculating_windows = {window_name: {} for window_name in WINDOWS.keys()}


def fft_image(image, window=None):
    if window is not None and window in WINDOWS:
        if image.shape in precalculating_windows[window]:
            window_array = precalculating_windows[window][image.shape]
        else:
            win_fce = WINDOWS[window]
            wy = np.clip(win_fce(image.shape[0]), 0, 1)
            wx = np.clip(win_fce(image.shape[1]), 0, 1)
            window_array = np.sqrt(np.outer(wy, wx)).astype(np.float32)
            precalculating_windows[window][image.shape] = window_array

        image = cv.multiply(image, window_array, dtype=cv2.CV_32F)

    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    mag = cv.magnitude(dft[:, :, 0], dft[:, :, 1])
    result = np.fft.fftshift(mag)

    cv.log(result, result)
    result = cv.addWeighted(result, alpha=65535 / result.max(), src2=result, beta=0, gamma=0, dtype=cv.CV_16U)

    return result


def filter_image(image, fft_mask):
    fft_image0 = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    fft_image1 = np.fft.fftshift(fft_image0)
    fft_image1[:, :, 0] *= fft_mask
    fft_image1[:, :, 1] *= fft_mask
    image2 = cv.idft(np.fft.ifftshift(fft_image1), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    return image2, fft_image0


def fft_log_image(fft_image):
    result = cv.magnitude(fft_image[:, :, 0], fft_image[:, :, 1])
    cv.log(result, result)
    result = cv.addWeighted(result, alpha=65535 / result.max(), src2=result, beta=0, gamma=0, dtype=cv.CV_16U)

    return result


@numba.njit(nogil=True)
def calc_hist_numba(img):
    size = img.shape[0]
    histogram = np.zeros((1024,), dtype=np.float32)

    for i in range(size):
        histogram[img[i] // 64] += 1.0

    return histogram


@numba.njit(nogil=True)
def calc_hist_numba_amplify(img, amplify=1):
    # amplify must be power of 2
    size = img.shape[0]

    if amplify <= 64:
        histogram = np.zeros((1024,), dtype=np.float32)
        factor = 64 // amplify
        for i in range(size):
            index = img[i] // factor
            if index >= 1024:
                index = 1023
            histogram[index] += 1.0
    else:
        factor = amplify // 64
        max_bin = 1024 // factor
        histogram = np.zeros((max_bin,), dtype=np.float32)
        for i in range(size):
            index = img[i]
            if index >= max_bin:
                index = max_bin - 1
            histogram[index] += 1.0
        histogram = np.repeat(histogram, factor)
    return histogram


@numba.njit(nogil=True)
def calc_hist_cast_numba(img, alpha, beta):
    size = img.size
    cast = np.zeros((size,), dtype=np.uint8)
    histogram = np.zeros((1024,), dtype=np.float32)

    for i in range(size):
        value = img[i] * alpha + beta
        if value > 255.0:
            value = 255.0
        elif value < 0.0:
            value = 0.0
        cast[i] = value + 0.5
        histogram[img[i] // 64] += 1.0

    return histogram, cast


@numba.njit(nogil=True)
def calc_cast_numba(img, alpha, beta):
    size = img.size
    cast = np.zeros((size,), dtype=np.uint8)

    for i in range(size):
        value = img[i] * alpha + beta
        if value > 255.0:
            value = 255.0
        elif value < 0.0:
            value = 0.0
        cast[i] = value + 0.5

    return cast


def count_rate_calibration(image, hv=100):
    """Guessed test function  y = -0.0319x4 + 0.5351x3 - 2.9334x2 + 5.6926x - 0.112   shows current in pixel in pA."""
    p = np.poly1d([-0.0319, 0.5351, -2.9334, 5.6926, -0.112])
    calibrated_image = p(image)

    return calibrated_image


def get_calibrate_function(parameters):
    if "calibrated_parameters" and "voltage" not in parameters:
        return None

    p = np.poly1d(parameters["calibrated_parameters"])
    hv = parameters["voltage"]

    def fce(image):
        calibrated_image = p(image)

        return calibrated_image

    return fce
