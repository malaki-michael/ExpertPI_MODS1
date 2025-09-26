import cv2
import numpy as np
from scipy.signal import medfilt2d

from expert_pi.measurements.gabor_variance import GaborKernel, GaborKernelSettings

# Use for high FOV or Stage:
kernel_mdct = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

kernel_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def _median_filter(image, ksize):
    """Median filter with opencv or scipy fallback for unsupported dtypes.

    Args:
        image (np.ndarray): Image to filter
        ksize (int): Kernel size

    Returns:
        np.ndarray: Filtered image
    """
    try:
        return cv2.medianBlur(image, ksize=ksize)
    except cv2.error:
        return medfilt2d(image, kernel_size=ksize)


def image_function(
    image, kernel=None, ksize=5, sigma=3, crop=False, gabor_kernel_settings: GaborKernelSettings | None = None
):
    if kernel is None or kernel == "None":
        return image

    crop_size = ksize
    if kernel == "sobel":
        img2 = _median_filter(image, ksize)
        img = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=ksize)
    elif kernel == "sobel2D":
        # img2 = _median_filter(image, ksize=ksize)
        img2 = cv2.GaussianBlur(image, (0, 0), sigma, sigma)
        grad_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, scale=8, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, scale=8, borderType=cv2.BORDER_DEFAULT)
        img = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        crop_size = sigma
    elif kernel == "scharr2D":
        # img2 = _median_filter(image, ksize=ksize)
        img2 = cv2.GaussianBlur(image, (0, 0), sigma, sigma)
        grad_x = cv2.Scharr(img2, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Scharr(img2, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        img = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, -1)
        crop_size = sigma
    elif kernel == "laplacian":
        img = cv2.GaussianBlur(image, (0, 0), sigma, sigma)
        dtype = cv2.CV_32F if img.dtype == np.float32 else cv2.CV_64F
        img = cv2.Laplacian(img, dtype, ksize=3, scale=1)
        img = np.absolute(img)
        crop_size = sigma
    elif kernel == "simple_laplacian":
        dtype = cv2.CV_32F if image.dtype == np.float32 else cv2.CV_64F
        img2 = cv2.Laplacian(image, dtype)
        img = np.absolute(img2)
    elif kernel == "median":
        img = _median_filter(image, ksize=ksize)
    elif kernel == "gaussian":
        img = cv2.GaussianBlur(image, (ksize, ksize), sigma, sigma)
    elif kernel == "gabor":
        img = GaborKernel(gabor_kernel_settings=gabor_kernel_settings)(image, device="cpu").mean(dim=0)
        img = img.detach().cpu().numpy()
    elif isinstance(kernel, np.ndarray):
        img = cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if crop is False:
        return img

    c = crop if isinstance(crop, int) else crop_size
    return np.ascontiguousarray(img[c:-c, c:-c])


def contrast_function(
    image, normalize=True, kernel=None, blur=None, blur_size=5, gabor_kernel_settings: GaborKernelSettings | None = None
):
    mean = np.mean(image)

    if blur == "median":
        image = _median_filter(image, blur_size)
    elif blur == "gaussian":
        image = cv2.GaussianBlur(image, (0, 0), blur_size)

    if mean != 0 and normalize:
        image = (image - mean) / mean

    if kernel is None:
        return np.var(image)
    elif kernel == "sobel":
        return np.var(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
    elif kernel == "gabor":
        return GaborKernel(gabor_kernel_settings=gabor_kernel_settings).variance(image, device="cpu")
    elif np.all((kernel - kernel.T) == 0):
        return np.var(
            cv2.filter2D(image, -1, kernel)
        )  # note: filter2d is correlation not convolution! 5-10x faster with opencv
    else:
        return np.var(cv2.filter2D(image, -1, kernel)) + np.var(cv2.filter2D(image, -1, kernel.T))
