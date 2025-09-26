import inspect
from collections.abc import Callable
from enum import Flag

import cv2
import matplotlib.pyplot as plt
import numpy as np
import stem_measurements
from scipy.ndimage import shift as scipy_shift
from serving_manager.api import registration_model
from skimage.registration import phase_cross_correlation

from expert_pi.measurements import kernels
from expert_pi.config import get_config

_custom_methods = {}


def register_custom_method(name: str, method: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    _custom_methods[name] = method


def remove_custom_method(name: str):
    if name in _custom_methods:
        _custom_methods.pop(name)


class Method(Flag):
    NONE = 2**0
    Templates = 2**1  # faster then Patches
    PatchesPass1 = 2**2  # quit slow
    PatchesPass2 = 2**3  # quit slow
    CrossCorr = 2**4  # fast but inaccurate
    ECCMaximization = 2**5  # relatively speed and good accuracy
    Orb = 2**6  # can be useful for repeated patterns
    OpticalFlow = 2**7  # not tested
    Custom = 2**8

    TEMRegistration = 2**9  # A.I. image registration method
    TEMRegistrationMedium = 2**10
    TEMRegistrationTiny = 2**11

    TEMRegistrationAll = TEMRegistration | TEMRegistrationMedium | TEMRegistrationTiny

    # TODO add TEM registration back when it is installed on
    Normal = Templates | CrossCorr | ECCMaximization | Custom
    Stable = Templates | CrossCorr | ECCMaximization | Orb | Custom
    All = Templates | PatchesPass1 | PatchesPass2 | CrossCorr | ECCMaximization | Orb | Custom


def get_offset_of_pictures(
    img0,
    img1,
    fov: float = 0,
    kernel: str | None = None,
    method: Method = Method.CrossCorr,
    upsample_factor=1,
    correlation_threshold=1.0,
    plot: bool = False,
    output: bool = False,
    coordinate_type: str = "sample",
    get_corr_coeff=False,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Calculates the offset between two images.

    Args:
        img0: The reference image.
        img1: The image with offset.
        fov: The field of view in micrometers.
        kernel: The kernel for image processing.
        method: The method used for calculating the offset. Can be a combination of methods using the '|' operator.
        upsample_factor: The precision of shift measuring.
        correlation_threshold: The correlation threshold.
        plot: If True, plots the result graph.
        output: If True, prints additional information.
        coordinate_type: The coordinate type. Can be 'sample' or 'image'. For image, the y-coordinate has the opposite
                         sign.
        get_corr_coeff: If True, returns the correlation coefficient along with the offset.
        **kwargs: Additional parameters for the methods.

    Returns:
        The offset between the two images in micrometers as a numpy array. If get_corr_coeff is True, returns a tuple
        containing the offset and the correlation coefficient.

    Raises:
        ShiftMeasurementException: If no method was selected.
    """
    shifts, used_methods = [], []
    corr_coeffs: list[float] = []

    used_kernels = {}

    def get_images(img_kernel):
        if isinstance(img_kernel, np.ndarray):
            return kernels.image_function(img0, img_kernel, crop=True), kernels.image_function(
                img1, img_kernel, crop=True
            )
        if img_kernel in used_kernels:
            return used_kernels[img_kernel]
        im0 = kernels.image_function(img0, img_kernel, crop=True)
        im1 = kernels.image_function(img1, img_kernel, crop=True)
        used_kernels[img_kernel] = (im0, im1)
        return im0, im1

    def method_end(method_name):
        nonlocal method
        used_methods.append(method_name)
        if corr_coeffs[-1] > correlation_threshold:
            method = Method.NONE

    ref_for_coeff, img_for_coeff = get_images("gaussian" if kernel is None else kernel)

    if Method.Templates in method:
        reference, image = get_images("gaussian" if kernel is None else kernel)
        add_kwargs = extract_parameters_for_function(method_matching_templates, kwargs)
        shifts.append(method_matching_templates(reference, image, upsample_factor=upsample_factor, **add_kwargs))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("Templates")

    if Method.PatchesPass1 in method:
        reference, image = get_images("gaussian" if kernel is None else kernel)
        shifts.append(method_matching_patches(image, image, upsample_factor=upsample_factor))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("PatchesPass1")

    if Method.PatchesPass2 in method:
        reference, image = get_images("gaussian" if kernel is None else kernel)
        shifts.append(method_matching_patches(reference, image, upsample_factor=upsample_factor, n=7))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("PatchesPass2")

    if Method.CrossCorr in method:
        reference, image = get_images("sobel2D" if kernel is None else kernel)
        shifts.append(method_cross_correlation(reference, image, upsample_factor=upsample_factor))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("CrossCorr")

    if Method.Custom in method:
        for method_name, custom_method in _custom_methods.items():
            reference, image = get_images("None" if kernel is None else kernel)
            add_kwargs = extract_parameters_for_function(custom_method, kwargs)
            shifts.append(custom_method(reference, image, **add_kwargs))
            corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
            method_end(method_name)

    tem_registrations_methods = Method.TEMRegistrationAll & method
    if tem_registrations_methods:
        reference, image = get_images(None)
        for reg_method in (Method.TEMRegistration, Method.TEMRegistrationMedium, Method.TEMRegistrationTiny):
            if reg_method in tem_registrations_methods:
                add_kwargs = extract_parameters_for_function(method_tem_registration, kwargs)
                shifts.append(method_tem_registration(reference, image, model=reg_method.name, **add_kwargs))
                add_kwargs2 = extract_parameters_for_function(get_correlation_coefficient, kwargs)
                corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1], **add_kwargs2))
                method_end(reg_method.name)

    if Method.ECCMaximization in method:
        filter_ = 5 if kernel is None else 1
        reference, image = get_images(kernel)
        estimation = shifts[np.nanargmax(corr_coeffs)] if corr_coeffs else (0, 0)
        shifts.append(method_ecc_maximization(reference, image, estimation=estimation, gauss_filter_size=filter_))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("ECCMaximization")

    if Method.Orb in method:
        reference, image = get_images("sobel2D")  # force to use sobel2D kernel
        shifts.append(method_orb_features_matching(reference, image))
        try:
            coeff = get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1])
        except cv2.error:
            coeff = -1
        corr_coeffs.append(coeff)

        method_end("Orb")

    if Method.OpticalFlow in method:
        reference, image = get_images("gaussian" if kernel is None else kernel)
        shifts.append(method_optical_flow(reference, image, threshold=1000))
        corr_coeffs.append(get_correlation_coefficient(ref_for_coeff, img_for_coeff, shifts[-1]))
        method_end("OpticalFlow")

    if not shifts:
        raise ShiftMeasurementError("No method was selected.")

    index = np.nanargmax(np.nan_to_num(corr_coeffs, nan=-1))
    im_shift = shifts[index]
    calculated_shift = im_shift / img0.shape * fov

    if output:
        print("best method:", used_methods[index])
        print("correlation:", corr_coeffs[index])
        print("-" * 10)
        print("correlations", "methods")
        for m, c in zip(used_methods, corr_coeffs):
            print(c, m)
        print("-" * 10)

        print("image size:", img0.shape, fov, "um")
        print("pixel shift:", im_shift)

        rotation = np.arctan2(im_shift[0], -im_shift[1])
        print("rotation:", rotation / np.pi * 180, "deg", rotation, "rad")

    if plot:
        y_factor = img0.shape[1] / img0.shape[0]
        extent = np.array([-fov / 2, fov / 2, -fov / 2 / y_factor, fov / 2 / y_factor])
        extent2 = (
            extent
            - np.array([-calculated_shift[1], -calculated_shift[1], calculated_shift[0], calculated_shift[0]]).tolist()
        )
        extent = extent.tolist()

        plt.figure(figsize=(10, 3))
        ax1 = plt.subplot(1, 4, 1)
        ax2 = plt.subplot(1, 4, 2)
        ax3 = plt.subplot(1, 4, 3)
        ax4 = plt.subplot(1, 4, 4)

        ax1.imshow(img0, extent=extent, cmap="Blues")
        ax1.set_title("reference image")

        ax2.imshow(img1, extent=extent, cmap="Reds")
        ax2.set_title("offset image")

        ax3.imshow(img0.real, extent=extent, cmap="Blues", alpha=0.5)
        ax3.imshow(img1.real, extent=extent2, cmap="Reds", alpha=0.5)
        ax3.set_xlim(min(extent[0], extent2[0]), max(extent[1], extent2[1]))
        ax3.set_ylim(min(extent[2], extent2[2]), max(extent[3], extent2[3]))
        ax3.set_title("Both")

        image_product = np.fft.fft2(img0) * np.fft.fft2(img1).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        ax4.imshow(np.abs(cc_image), extent=extent)
        ax4.plot([calculated_shift[1]], [-calculated_shift[0]], "o")

        plt.show()

    if coordinate_type == "sample":
        calculated_shift = np.array([calculated_shift[1], -calculated_shift[0]])  # To XY coordinates + change of sign
    elif coordinate_type == "image":
        calculated_shift = np.array([calculated_shift[1], calculated_shift[0]])  # To XY coordinates + change of sign

    if get_corr_coeff:
        return calculated_shift, corr_coeffs[index]

    return calculated_shift


def extract_parameters_for_function(function, parameters):
    """Extracts parameters for a function from a dictionary.

    If you have lots of parameters in **kwargs and you would like to propagate just the ones related to the function use:
    function(**extract_parameters_for_function(function,kwargs))
    """
    signature = inspect.signature(function)
    new_parameters = {k: parameters[k] for k in signature.parameters.keys() & parameters.keys()}
    return new_parameters


class ShiftMeasurementError(Exception):
    """Exaption for shift measurement."""

    pass


def _old_method_features_matching(reference, image, method="orb", match="bf", k=0.7):
    """Method use features matching.

    Args:
        reference: reference image, uint16
        image: moving image, uint16
        method: 'orb', 'sift'
        match: 'bf' - brutal force, 'flann'
        k: matching ratio

    Returns:
        shift [dy, dx]
    """
    ref = cv2.convertScaleAbs(reference, alpha=255 / 65535)
    img = cv2.convertScaleAbs(image, alpha=255 / 65535)

    if method == "sift":
        ft = cv2.SIFT_create()
        norm = cv2.NORM_L2
        search_params = dict(checks=50)
        index_params = dict(algorithm=1, trees=5)
    elif method == "orb":
        ft = cv2.ORB_create(nlevels=2)
        norm = cv2.NORM_HAMMING
        search_params = dict(checks=50)
        index_params = dict(
            algorithm=6,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,
        )  # 2
    else:
        raise ValueError(f"Unknown method: {method}")

    if match == "bf":
        matcher = cv2.BFMatcher(normType=norm)
    else:
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        kp1, des1 = ft.detectAndCompute(ref, None)
        kp2, des2 = ft.detectAndCompute(img, None)

        matches = matcher.knnMatch(des1, des2, k=2)

        good = [ma[0] for ma in matches if len(ma) == 2 and ma[0].distance < k * ma[1].distance]

        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #
        # h, w = ref.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, mat)
        # result = -np.array([dst[0, 0, 1], dst[0, 0, 0]], dtype=np.float64)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        diff = src_pts - dst_pts

        xm = np.median(diff[:, 0])
        x_new = [num for num in diff[:, 0] if xm - 1 < num < xm + 1]
        ym = np.median(diff[:, 1])
        y_new = [num for num in diff[:, 1] if ym - 1 < num < ym + 1]

        result = np.array([np.median(y_new), np.median(x_new)])

    except Exception as _:
        result = np.asarray((np.nan, np.nan))

    return result


def method_tem_registration(
    reference, image, model="TEMRegistration", return_value="shift", left_masks=None, right_masks=None
):
    """TESCAN A.I. registration method.

    Args:
        reference: reference image
        image: shifted image
        model: possible values: ['TEMRegistration' - normal model,
                                'TEMRegistrationMedium' - faster, lower precision,
                                'TEMRegistrationTiny' - superfast, very raw precision]
        return_value: possible values:['shift', 'homography', 'full']
        left_masks: masks for left image
        right_masks: masks for right image

    Returns:
        shift: array of shape (2,) representing the shift [dy, dx]
    """
    if reference.dtype != np.uint8:
        reference = (reference / np.max(reference) * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        image = (image / np.max(image) * 255).astype(np.uint8)

    x = cv2.hconcat([reference, image])
    config = get_config().inference

    try:
        model_output = registration_model(
            x,
            host=config.host,
            port=config.plugin_port,
            model_name=model,
            return_only_homography=False,
            left_masks=left_masks,
            right_masks=right_masks,
        )[0]  # newer version allows batches
        if not model_output["success"]:
            return np.array([np.nan, np.nan])

    except Exception as _:
        import traceback

        traceback.print_exc()
        return np.array([np.nan, np.nan])

    if return_value == "shift":
        xy = np.array(model_output["translation"]) * reference.shape[0]
        return np.array([xy[1], xy[0]])

    return model_output


def method_optical_flow(reference, image, estimation=None, threshold=None, plot=False, use_affine=False):
    """Computes a dense optical flow using the Gunnar Farneback's algorithm.

    Args:
        reference: Reference image.
        image: Shifted image.
        estimation: Shift estimation.
        threshold: Threshold value for images with vacuum. None - no vacuum, 0 - use Ostu threshold.
        plot: If True, plot shifts maps.
        use_affine: If True, result shift is calculated from affine coefficients.

    Returns:
        shift: Array of shape (2,) representing the shift [dy, dx].
    """
    if estimation is not None:
        flow = np.zeros((reference.shape[0], reference.shape[1], 2), dtype=np.float32)
        flow[:, :, 0] = -estimation[1]
        flow[:, :, 1] = -estimation[0]
        flag = cv2.OPTFLOW_USE_INITIAL_FLOW
    else:
        flow = None
        flag = 0

    flow = cv2.calcOpticalFlowFarneback(
        reference, image, flow, pyr_scale=0.5, levels=3, winsize=40, iterations=15, poly_n=7, poly_sigma=1.5, flags=flag
    )

    if plot:
        plt.figure()
        plt.imshow(flow[:, :, 0], "gray")
        plt.figure()
        plt.imshow(flow[:, :, 1], "gray")

    mask = None
    if threshold is None:
        x_shifts = flow[:, :, 0].flatten()
        y_shifts = flow[:, :, 1].flatten()
    else:
        if threshold == 0:
            type_ = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        else:
            type_ = cv2.THRESH_BINARY

        r, mask = cv2.threshold(reference, threshold, reference.max(), type_)
        mask = mask.astype(np.bool)

        x_shifts = flow[:, :, 0][mask]
        y_shifts = flow[:, :, 1][mask]

    if use_affine:
        ry = range(reference.shape[0])
        rx = range(reference.shape[1])
        cx, cy = np.meshgrid(rx, ry)

        if mask is None:
            cx = cx.flatten()
            cy = cy.flatten()
        else:
            cx = cx[mask]
            cy = cy[mask]

        dx = cx + x_shifts
        dy = cy + y_shifts

        src_pts = np.array([[[x, y]] for x, y in zip(cx, cy)])
        dst_pts = np.array([[[x, y]] for x, y in zip(dx, dy)])

        m = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

        return -np.array((m[1, 2], m[0, 2]))

    return -np.asarray((np.median(y_shifts), np.median(x_shifts)))


def method_orb_features_matching(reference, image):
    """Computes the shift between a reference image and a moving image.

    Args:
        reference: The reference image.
        image: The moving image.

    Returns:
        The shift [dy, dx].
    """
    # TODO: test proper parameters for cv functions
    alpha = 255 / max(reference.max(), image.max())
    ref = cv2.convertScaleAbs(reference, alpha=alpha)
    img = cv2.convertScaleAbs(image, alpha=alpha)

    max_feature = max(1000, int(0.005 * ref.size))
    scale_factor = 1.2
    edge_threshold = 0
    n_levels = 5

    orb = cv2.ORB_create(max_feature, scale_factor, n_levels, edge_threshold)

    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

    matches = matcher.knnMatch(des1, des2, k=2)

    good = [ma[0] for ma in matches if len(ma) == 2 and ma[0].distance < 0.7 * ma[1].distance]
    # good = [ma[0] for ma in matches]

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)

    if len(src_pts) == 0:
        return np.array((np.nan, np.nan))

    m = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

    if m is None:
        return np.zeros(2)

    return -np.array((m[1, 2], m[0, 2]))


def method_cross_correlation(reference_image, image, upsample_factor=1):
    """Method that uses phase cross correlation to estimate the shift between a reference image and a shifted image.

    Args:
        reference_image: The reference image.
        image: The shifted image.
        upsample_factor: The subpixel precision.

    Returns:
        The shift as a tuple (y, x).
    """
    shift = phase_cross_correlation(reference_image, image, upsample_factor=upsample_factor)[0]
    return shift


def method_ecc_maximization(reference_image, image, estimation=(0, 0), gauss_filter_size=1):
    """Estimate shift by Enhanced Correlation Coefficient maximization.

    Args:
        reference_image: The reference image.
        image: The shifted image.
        estimation: The initial estimation of the shift. Default is (0, 0).
        gauss_filter_size: The size of the Gaussian filter. Default is 1, which means it is turned off.

    Returns:
        The shift as a tuple (y, x).
    """
    ref = reference_image.astype(np.float32) if reference_image.dtype != np.float32 else reference_image
    img = image.astype(np.float32) if image.dtype != np.float32 else image

    matrix = np.array(((1, 0, estimation[1]), (0, 1, estimation[0])), dtype=np.float32)

    try:
        cv2.findTransformECC(
            img,
            ref,
            matrix,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-10),
            inputMask=None,
            gaussFiltSize=gauss_filter_size,
        )
    except cv2.error:
        return np.array((np.nan, np.nan))

    return np.asarray((matrix[1, 2], matrix[0, 2]))


def method_matching_templates(reference_image, image, upsample_factor=1, rectangle=None, overlap=0.2):
    """Method use template matching, first for center area of image and after for best places in each quadrant.

    Args:
        reference_image: The reference image.
        image: The shifted image.
        upsample_factor: The subpixel precision.
        rectangle: A list of four integers representing the coordinates and size of the template to select from the
                   image. The format is ['left', 'top', 'width', 'height'].
        overlap: The overlap between quadrants.

    Returns:
        The shift as a tuple (y, x).
    """
    ref = reference_image.astype(np.float32) if reference_image.dtype != np.float32 else reference_image
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    shape = np.asarray(image.shape)

    if rectangle is None:
        # image center
        sizes = [np.floor(shape / 3).astype(np.uint32)]
        steps = [np.floor(sizes[0] / 2).astype(np.uint32)]

        size = int(np.min(shape) / 3)
        size -= size % 2
        cuts_1 = np.round(shape * (1 + overlap) / 2).astype(np.uint32).astype(np.uint32)
        cuts_2 = shape - cuts_1

        selections = (
            (0, cuts_1[0], 0, cuts_1[1]),
            (cuts_2[0], shape[0], 0, cuts_1[1]),
            (0, cuts_1[0], cuts_2[1], shape[1]),
            (cuts_2[0], shape[0], cuts_2[1], shape[1]),
        )

        for s in selections:
            coord = get_place_for_measurement(image[s[0] : s[1], s[2] : s[3]], size, sub_variance_factor=2, plot=False)
            sizes.append(np.asarray((size, size)))
            steps.append(np.asarray((coord[0] + s[0], coord[1] + s[2])))

    else:
        sizes = [(rectangle[3], rectangle[2])]
        steps = [(rectangle[1], rectangle[0])]

    shifts, coeffs = [], []
    for step, size in zip(steps, sizes):
        template = img[step[0] : step[0] + size[0], step[1] : step[1] + size[1]]

        match = cv2.matchTemplate(ref, template, method=cv2.TM_CCOEFF_NORMED)
        shift = cv2.minMaxLoc(match)[3]

        sub_reference = ref[shift[1] : shift[1] + size[0], shift[0] : shift[0] + size[1]]
        if upsample_factor != 1:
            sub_pixel_shift, max_coeff = _sub_pixel_shift_correlation(sub_reference, template, upsample_factor)
            shift = np.asarray(shift) + sub_pixel_shift
            coeffs.append(max_coeff)
        else:
            coeffs.append(get_correlation_coefficient(sub_reference, template, (0, 0)))

        shift = np.asarray((shift[1] - step[0], shift[0] - step[1]))
        shifts.append(shift)

    shift = shifts[np.argmax(coeffs)]

    return shift


def method_matching_patches(reference_image, image, upsample_factor=1, n=5, overlap=0.25, use_statistic=True):
    """Method that uses template matching for each sub-image from a grid of sub-images.

    Args:
        reference_image: The reference image.
        image: The shifted image.
        upsample_factor: The subpixel precision.
        n: The size of the grid.
        overlap: The overlap between quadrants.
        use_statistic: Whether to calculate the mean shift for each grouped shifts.

    Returns:
        The shift as a tuple (y, x).
    """
    sizes = np.floor((
        image.shape[0] / (overlap + n - overlap * n),
        image.shape[1] / (overlap + n - overlap * n),
    )).astype(np.uint32)
    steps = np.floor((sizes[0] * (1 - overlap), sizes[1] * (1 - overlap))).astype(np.uint32)

    ref = reference_image.astype(np.float32) if reference_image.dtype != np.float32 else reference_image
    img = image.astype(np.float32) if image.dtype != np.float32 else image

    shifts, coeffs = [], []
    for i in range(n):
        for k in range(n):
            template = img[i * steps[0] : i * steps[0] + sizes[0], k * steps[1] : k * steps[1] + sizes[1]]
            if cv2.meanStdDev(template)[1][0, 0] ** 2 < 100:  # TODO: estimate this threshold
                continue
            match = cv2.matchTemplate(ref, template, method=cv2.TM_CCOEFF_NORMED)
            shift = cv2.minMaxLoc(match)[3]
            sub_reference = ref[shift[1] : shift[1] + sizes[0], shift[0] : shift[0] + sizes[1]]
            if upsample_factor != 1:
                sub_pixel_shift, max_coeff = _sub_pixel_shift_correlation(sub_reference, template, upsample_factor)
                shift = np.asarray(shift) + sub_pixel_shift
                coeffs.append(max_coeff)
            else:
                coeffs.append(get_correlation_coefficient(sub_reference, template, (0, 0)))

            shift = np.asarray((shift[1] - i * steps[0], shift[0] - k * steps[1]))
            shifts.append(shift)

    shift = shifts[np.argmax(coeffs)]

    if not use_statistic:
        return shift

    groups_shift = [shift]
    groups_coeff = [get_correlation_coefficient(ref, img, shift)]

    while len(shifts) > 0:
        actual_shift = shifts.pop(0)
        actual_coeff = coeffs.pop(0)

        group_shift = [actual_shift]
        group_coeff = [actual_coeff]

        for i, s in enumerate(shifts):
            if np.linalg.norm(actual_shift - s) < 1.5:
                group_shift.append(s)
                group_coeff.append(coeffs[i])
                shifts.pop(i)
                coeffs.pop(i)

        shift_mean = np.mean(group_shift, axis=0)
        coeff_mean = np.mean(group_coeff, axis=0)

        if coeff_mean >= 0.9:
            groups_shift.append(shift_mean)
            groups_coeff.append(get_correlation_coefficient(ref, img, shift_mean))

    return groups_shift[np.argmax(groups_coeff)]


def _sub_pixel_shift_correlation(reference, image, upsample_factor, shift=(0, 0)):
    scale = 0.5
    final_scale = 1 / upsample_factor
    max_coeff = get_correlation_coefficient(reference, image, (0, 0))
    shift = np.array(shift)
    directions = (np.array((-1, 1)), np.array((1, 1)), np.array((-1, -1)), np.array((1, -1)))
    while scale >= final_scale:
        shifts, coeffs = [], []
        for d in directions:
            s = shift + d * scale
            shifts.append(s)
            coeffs.append(get_correlation_coefficient(reference, image, s))

        if max_coeff > max(coeffs):
            scale *= 0.5
            continue

        shift = shifts[np.argmax(coeffs)]
        scale *= 0.5
        max_coeff = max(coeffs)

    return shift, max_coeff


def get_correlation_coefficient(img_0, img_1, shift, inner_rectangle_0=None, inner_rectangle_1=None) -> float:
    """Calculation of Pearson correlation coefficient between two shifted images.

    Args:
        img_0: The original image.
        img_1: The shifted image.
        shift: The shift as a tuple (y, x).
        inner_rectangle_0: Optional. Selects the inner rectangle of img_0 [top, left, height, width].
        inner_rectangle_1: Optional. Selects the inner rectangle of img_1 [top, left, height, width].

    Returns:
        The Pearson correlation coefficient.
    """
    if np.isnan(shift).any():
        return -1

    return stem_measurements.correl_coeff(img_0, img_1, shift, inner_rectangle_0, inner_rectangle_1)


def get_correlation_coefficient_python(
    img_0, img_1, shift, inner_rectangle_0=None, inner_rectangle_1=None, plot=False
) -> float:
    """Calculates the Pearson correlation coefficient between two shifted images.

    Args:
        img_0: The original image.
        img_1: The shifted image.
        shift: The shift as a tuple (y, x).
        inner_rectangle_0: Optional. Selects the inner rectangle of img_0 [top, left, height, width].
        inner_rectangle_1: Optional. Selects the inner rectangle of img_1 [top, left, height, width].
        plot: Whether to plot the result.

    Returns:
        The Pearson correlation coefficient.
    """
    if np.isnan(shift).any():
        return -1

    pixel_shift = np.ceil(shift).astype(np.int32)
    sub_shift = np.asarray(shift) - pixel_shift
    # print(pixel_shift, sub_shift)
    if sub_shift.any():
        # matrix = np.array(((1, 0, sub_shift[1]), (0, 1, sub_shift[0])))
        # img_1_shift = cv2.warpAffine(
        #     img_1, matrix, (img_1.shape[1], img_1.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        # )
        img_1_shift = scipy_shift(img_1, sub_shift, order=1, cval=65535)  # slow but more precise than opencv
    else:
        img_1_shift = img_1

    if inner_rectangle_0 is None:
        inner_rectangle_0 = [0, 0, img_0.shape[0], img_0.shape[1]]
    if inner_rectangle_1 is None:
        inner_rectangle_1 = [0, 0, img_1.shape[0], img_1.shape[1]]

    pixel_shift[0] += inner_rectangle_1[0] - inner_rectangle_0[0]
    pixel_shift[1] += inner_rectangle_1[1] - inner_rectangle_0[1]

    if pixel_shift[0] > 0:
        inner_rectangle_0[0] += pixel_shift[0]
        inner_rectangle_0[2] = min(img_0.shape[0] - inner_rectangle_0[0], inner_rectangle_0[2] - pixel_shift[0])
    else:
        inner_rectangle_1[0] -= pixel_shift[0]
        inner_rectangle_1[2] = min(img_1.shape[0] - inner_rectangle_1[0], inner_rectangle_1[2] + pixel_shift[0])

    if pixel_shift[1] > 0:
        inner_rectangle_0[1] += pixel_shift[1]
        inner_rectangle_0[3] = min(img_0.shape[1] - inner_rectangle_0[1], inner_rectangle_0[3] - pixel_shift[1])
    else:
        inner_rectangle_1[1] -= pixel_shift[1]
        inner_rectangle_1[3] = min(img_1.shape[1] - inner_rectangle_1[1], inner_rectangle_1[3] + pixel_shift[1])

    inner_rectangle_0[2] = inner_rectangle_1[2] = max(0, min(inner_rectangle_0[2], inner_rectangle_1[2] - 1))
    inner_rectangle_0[3] = inner_rectangle_1[3] = max(0, min(inner_rectangle_0[3], inner_rectangle_1[3] - 1))

    if inner_rectangle_0[2] * inner_rectangle_0[3] == 0:
        return -1.0

    # print(inner_rectangle_0)
    # print(inner_rectangle_1)
    sub_img_0 = img_0[
        inner_rectangle_0[0] : inner_rectangle_0[0] + inner_rectangle_0[2],
        inner_rectangle_0[1] : inner_rectangle_0[1] + inner_rectangle_0[3],
    ]
    sub_img_1 = img_1_shift[
        inner_rectangle_1[0] : inner_rectangle_1[0] + inner_rectangle_1[2],
        inner_rectangle_1[1] : inner_rectangle_1[1] + inner_rectangle_1[3],
    ]

    if plot:
        plt.figure()
        plt.imshow(np.hstack([img_0, img_1]), cmap="gray")
        ir0 = inner_rectangle_0
        ir1 = inner_rectangle_1
        ir1[1] += img_0.shape[1]

        for ir in [ir0, ir1]:
            plt.plot(
                [ir[1], ir[1], ir[1] + ir[3], ir[1] + ir[3], ir[1]], [ir[0], ir[0] + ir[2], ir[0] + ir[2], ir[0], ir[0]]
            )

        plt.figure()
        plt.imshow(np.hstack([sub_img_0, sub_img_1]), cmap="gray")

    return cv2.computeECC(sub_img_0, sub_img_1)
    # return (cv2.computeECC(sub_img_0, sub_img_1), _correlation_coefficient(sub_img_0, sub_img_1))


def _correlation_coefficient(img0, img1):
    """Testing function for calculation of Pearson correlation coefficient."""
    img0 = img0.astype(np.float64)
    img1 = img1.astype(np.float64)

    # return (img0 * img1).sum()

    img0_mean = img0 - np.mean(img0)
    img1_mean = img1 - np.mean(img1)

    return np.sum(img0_mean * img1_mean) / np.sqrt(np.sum(img0_mean**2) * np.sum(img1_mean**2))


def get_place_for_measurement(image, size=256, sub_variance_factor=1, plot=False):
    if image.shape[0] < size or image.shape[1] < size:
        return [0, 0]

    img = image / np.sum(image)

    img = np.block([[np.zeros((1, 1)), np.zeros((1, img.shape[1]))], [np.zeros((img.shape[0], 1)), img]])

    x = np.cumsum(np.cumsum(img, axis=0), axis=1)
    x2 = np.cumsum(np.cumsum(img**2, axis=0), axis=1)

    # variance of entire half
    xb = x[size:, size:] + x[0:-size, 0:-size] - x[size:, 0:-size] - x[0:-size, size:]
    x2b = x2[size:, size:] + x2[0:-size, 0:-size] - x2[size:, 0:-size] - x2[0:-size, size:]

    mean = xb / size**2
    var = x2b / size**2 - mean**2
    # variance of subparts:
    if sub_variance_factor != 0:
        size_h = size // 2  # N has to be dividable of two
        xbh = x[size_h:, size_h:] + x[0:-size_h, 0:-size_h] - x[size_h:, 0:-size_h] - x[0:-size_h, size_h:]
        x2bh = x2[size_h:, size_h:] + x2[0:-size_h, 0:-size_h] - x2[size_h:, 0:-size_h] - x2[0:-size_h, size_h:]

        mean_h = xbh / size_h**2
        var_h_part = x2bh / size_h**2 - mean_h**2
        var_h = (
            var_h_part[size_h:, size_h:]
            + var_h_part[size_h:, :-size_h]
            + var_h_part[:-size_h, size_h:]
            + var_h_part[:-size_h, :-size_h]
        )
        var2 = var + sub_variance_factor * var_h
    else:
        var2 = var
        var_h = var * 0

    ind = np.unravel_index(np.argmax(var2), var.shape)

    if plot:
        _, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        ax[0, 0].imshow(img, cmap="gray")
        ax[0, 0].plot(
            [ind[1], ind[1] + size, ind[1] + size, ind[1], ind[1]],
            [ind[0], ind[0], ind[0] + size, ind[0] + size, ind[0]],
            color="red",
        )
        ax[0, 0].set_title("image")

        ax[0, 1].imshow(var)
        ax[0, 1].set_title("variance")

        ax[1, 0].imshow(var_h)
        ax[1, 0].set_title("mean of variances")

        ax[1, 1].imshow(var2)
        ax[1, 1].plot([ind[1]], [ind[0]], "o", color="red")
        ax[1, 1].set_title("total variance")

        plt.show()

    return ind
