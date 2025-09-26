import warnings

import cv2
import numpy as np

from expert_pi.measurements import shift_measurements

try:
    import astra
except (ImportError, ModuleNotFoundError):
    warnings.warn("tomography reconstruction requires astra-toolbox to be installed")
    astra = None


def transform_image(image, from_transform, to_transform):
    """Transform 2x2 matrix, return transformed image with the same center and same dimensions."""
    transform = np.linalg.solve(to_transform.T, from_transform.T).T
    warp_b = np.dot(np.eye(2) - transform, np.array(image.shape).reshape(2, 1) / 2)
    warp_mat = np.hstack([transform, warp_b])
    warped_image = cv2.warpAffine(image, warp_mat, image.shape)
    return warped_image


def transform_positions(r, from_transform, to_transform):
    """Transform 2x2 matrix, return transformed positions [[x0,x1,...],[y0,y1,...]]."""
    mr = from_transform
    mt = to_transform

    return np.dot(mr, np.linalg.solve(mt, r))


def mult_along_axis(a, b, axis):
    shape = np.swapaxes(a, a.ndim - 1, axis).shape
    b_brc = np.broadcast_to(b, shape)
    b_brc = np.swapaxes(b_brc, a.ndim - 1, axis)
    return a * b_brc


def align_neighbours_images(
    sinograms,
    transforms2x2=None,
    method=shift_measurements.Method.CrossCorr,
    neighbours=10,
    filter_type="mean",
    progress_callback=None,
):
    """Filter_type mean or median."""
    N = sinograms.shape[0]

    shifts_px = []
    for i in range(N):
        shifts = []
        for j in range(-neighbours, neighbours + 1):
            if j + i < 0 or j == 0 or j + i > N - 1:
                continue
            if transforms2x2 is not None:
                ref_im = transform_image(sinograms[j + i], transforms2x2[j + i], transforms2x2[i])
            else:
                ref_im = sinograms[j + i]

            shift = shift_measurements.get_offset_of_pictures(
                ref_im, sinograms[i], fov=1, upsample_factor=2, method=method, coordinate_type="image", plot=False
            )
            shifts.append(shift)

        if filter_type == "mean":
            shift = np.mean(shifts, axis=0)
        else:
            shift = np.median(shifts, axis=0)
        shift_px = shift * ref_im.shape
        shifts_px.append(shift_px)

        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

    return np.array(shifts_px)


def align_pairs_images(
    sinograms,
    sinograms_reference,
    method=shift_measurements.Method.CrossCorr,
    progress_callback=None,
    kernel="None",
    upsample_factor=4,
):
    n = sinograms.shape[0]

    shifts_px = []
    for i in range(n):
        shift = shift_measurements.get_offset_of_pictures(
            sinograms_reference[i],
            sinograms[i],
            fov=1,
            upsample_factor=upsample_factor,
            method=method,
            coordinate_type="image",
            plot=False,
            kernel=kernel,
        )

        shift_px = shift * sinograms[i].shape
        shifts_px.append(shift_px)

        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

    if progress_callback is not None:
        running = progress_callback(-1)

    return np.array(shifts_px)


def align_by_rectangle(
    sinograms,
    reference_index,
    reference_rectangle,
    transforms2x2,
    alpha_y=None,
    indices=None,
    method=shift_measurements.Method.CrossCorr,
    progress_callback=None,
    kernel=None,
    upsample_factor=4,
):
    """Reference_rectangle: [x,y,x2,y2] in px."""
    print(method)
    if indices is None:
        indices = range(sinograms.shape[2])

    if alpha_y is None:
        alpha_y = sinograms.shape[1] // 2

    R = np.array([
        [reference_rectangle[0], reference_rectangle[2]],
        [reference_rectangle[1] - alpha_y, reference_rectangle[3] - alpha_y],
    ])

    shifts_px = []
    for k in indices:
        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break
        ref_im = transform_image(sinograms[reference_index], transforms2x2[reference_index], transforms2x2[k])
        Rt = transform_positions(R, transforms2x2[reference_index], transforms2x2[k])
        Rt[1] += alpha_y

        i, i2 = Rt[1, :].astype("int")
        j, j2 = Rt[0, :].astype("int")

        shift = shift_measurements.get_offset_of_pictures(
            ref_im[i:i2, j:j2],
            sinograms[k][i:i2, j:j2],
            fov=1,
            upsample_factor=upsample_factor,
            method=method,
            coordinate_type="image",
            plot=False,
        )
        shift_px = shift * ref_im[i:i2, j:j2].shape
        shifts_px.append(shift_px)

    if progress_callback is not None:
        running = progress_callback(-1)
    return np.array(shifts_px)


def shift_sinograms(
    sinograms, output_sinograms, shifts, transforms2x2=None, angles=None, indices=None, progress_callback=None
):
    if indices is None:
        indices = range(sinograms.shape[0])

    for i in indices:
        matrix = np.array(((0, 0, shifts[i, 0]), (0, 0, shifts[i, 1])), dtype=np.float64)
        if transforms2x2 is not None:
            transforms2x2_no_angle = np.dot(np.array([[1, 0], [0, np.cos(angles[i])]]), transforms2x2[i])
            transform = np.linalg.inv(transforms2x2_no_angle)
            warp_b = np.dot(np.eye(2) - transform, np.array(sinograms.shape[1:]).reshape(2, 1) / 2)
            warp_mat = np.hstack([transform, warp_b])
            matrix += warp_mat
        else:
            matrix += np.array(((1, 0, 0), (0, 1, 0)), dtype=np.float64)

        image = sinograms[i].astype(np.float32)
        new_image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        output_sinograms[i] = np.clip(new_image, 0.0, 2**16 - 1).astype(np.uint16)

        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break
    if progress_callback is not None:
        running = progress_callback(-1)


def reconstruct_2d(
    sinograms,
    volume,
    angles,
    indices=None,
    method="FBP",
    projector="strip",
    iterations=10,
    det_width=1.0,
    constr=None,
    progress_callback=None,
    use_gpu=False,
):
    if astra is None:
        raise ImportError("tomography reconstruction requires astra-toolbox to be installed")

    # sinograms axes: alpha,Y,X
    # volume axes: Z,Y,X
    if method == "SIRT_serial":
        serial = True
        method = "SIRT"
    else:
        serial = False

    if use_gpu:
        method += "_CUDA"
        projector = "cuda"

    vol_geom = astra.create_vol_geom(volume.shape[0], volume.shape[1])
    proj_geom = astra.create_proj_geom("parallel", det_width, sinograms.shape[1], np.array(angles))

    vol_id = astra.data2d.create("-vol", vol_geom, 0.0)
    sin_id = astra.data2d.create("-sino", proj_geom)
    proj_id = astra.create_projector(projector, proj_geom, vol_geom)

    cfg = astra.astra_dict(method)
    cfg["ReconstructionDataId"] = vol_id
    cfg["ProjectionDataId"] = sin_id
    cfg["ProjectorId"] = proj_id
    if constr is not None and "SIRT" in method:
        cfg["option"] = {}
        cfg["option"]["MinConstraint"] = constr[0]
        cfg["option"]["MaxConstraint"] = constr[1]

    alg_id = astra.algorithm.create(cfg)

    if indices is None:
        indices = range(sinograms.shape[2])

    for i in indices:
        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

        astra.data2d.store(sin_id, sinograms[:, :, i])
        if not serial:
            astra.data2d.store(vol_id, volume[:, :, i])
        astra.algorithm.run(alg_id, iterations)
        volume[:, :, i] = astra.data2d.get(vol_id)
        if not serial:
            astra.data2d.store(vol_id, 0.0)

    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()

    if progress_callback is not None:
        running = progress_callback(-1)

    return volume


def forward_projection(
    sinograms, volume, angles, indices=None, projector="strip", det_width=1.0, progress_callback=None
):
    if astra is None:
        raise ImportError("tomography reconstruction requires astra-toolbox to be installed")

    vol_geom = astra.create_vol_geom(volume.shape[0], volume.shape[1])
    proj_geom = astra.create_proj_geom("parallel", det_width, sinograms.shape[1], np.array(angles))

    vol_id = astra.data2d.create("-vol", vol_geom, 0.0)
    sin_id = astra.data2d.create("-sino", proj_geom)
    proj_id = astra.create_projector(projector, proj_geom, vol_geom)

    cfg = astra.astra_dict("FP")
    cfg["VolumeDataId"] = vol_id
    cfg["ProjectionDataId"] = sin_id
    cfg["ProjectorId"] = proj_id

    alg_id = astra.algorithm.create(cfg)

    if indices is None:
        indices = range(sinograms.shape[2])

    for i in indices:
        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

        astra.data2d.store(vol_id, volume[:, :, i])
        astra.algorithm.run(alg_id)
        sinograms[:, :, i] = astra.data2d.get(sin_id)

    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()

    if progress_callback is not None:
        running = progress_callback(-1)

    return sinograms


def get_simulated_sinograms(
    sinograms,
    sinograms_simulated,
    volume,
    angles,
    indices=None,
    method="FBP",
    projector="strip",
    iterations=10,
    det_width=1.0,
    constr=None,
    progress_callback=None,
    use_gpu=False,
    sphere_radius=1,
):
    if astra is None:
        raise ImportError("tomography reconstruction requires astra-toolbox to be installed")

    # sinograms axes: alpha,Y,X
    # volume axes: Z,Y,X
    # sinograms already filtered by x mask
    if method == "SIRT_serial":
        serial = True
        method = "SIRT"
    else:
        serial = False

    if use_gpu:
        method += "_CUDA"
        projector = "cuda"

    vol_geom = astra.create_vol_geom(volume.shape[0], volume.shape[1])  # shoul be same
    vol_id = astra.data2d.create("-vol", vol_geom, 0.0)

    X, Y = np.meshgrid(np.linspace(-1, 1, num=volume.shape[0]), np.linspace(-1, 1, num=volume.shape[1]))
    maskR = X**2 + Y**2 < sphere_radius**2

    if indices is None:
        indices = range(sinograms.shape[2])

    for i in indices:
        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

        # first we generate the entire simulation
        proj_geom = astra.create_proj_geom("parallel", det_width, sinograms.shape[1], np.array(angles))
        sin_id = astra.data2d.create("-sino", proj_geom)
        proj_id = astra.create_projector(projector, proj_geom, vol_geom)

        cfg = astra.astra_dict(method)
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = sin_id
        cfg["ProjectorId"] = proj_id
        if constr is not None and "SIRT" in method:
            cfg["option"] = {}
            cfg["option"]["MinConstraint"] = constr[0]
            cfg["option"]["MaxConstraint"] = constr[1]

        alg_id = astra.algorithm.create(cfg)

        astra.data2d.store(sin_id, sinograms[:, :, i])
        astra.data2d.store(vol_id, 0.0)
        astra.algorithm.run(alg_id, iterations)
        volume[:, :, i] = astra.data2d.get(vol_id)

        astra.data2d.delete(sin_id)
        astra.projector.delete(proj_id)

        for j in range(angles.shape[0]):
            # backward projection
            proj_geom = astra.create_proj_geom("parallel", det_width, sinograms.shape[1], np.array(angles)[j : j + 1])
            sin_id = astra.data2d.create("-sino", proj_geom)
            proj_id = astra.create_projector(projector, proj_geom, vol_geom)

            cfg = astra.astra_dict(method)
            cfg["ReconstructionDataId"] = vol_id
            cfg["ProjectionDataId"] = sin_id
            cfg["ProjectorId"] = proj_id
            if constr is not None and "SIRT" in method:
                cfg["option"] = {}
                cfg["option"]["MinConstraint"] = constr[0]
                cfg["option"]["MaxConstraint"] = constr[1]

            alg_id = astra.algorithm.create(cfg)

            astra.data2d.store(sin_id, sinograms[j : j + 1, :, i])  # substract
            astra.data2d.store(vol_id, 0.0)  # need to create a copy
            astra.algorithm.run(alg_id, iterations)
            volume_single = astra.data2d.get(vol_id)

            # forward projection

            cfg = astra.astra_dict("FP")
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sin_id
            cfg["ProjectorId"] = proj_id

            alg_id = astra.algorithm.create(cfg)
            astra.data2d.store(vol_id, (volume[:, :, i] - volume_single / np.array(angles).shape[0]) * maskR)

            astra.algorithm.run(alg_id)

            sinograms_simulated[j : j + 1, :, i] = astra.data2d.get(sin_id)

            astra.data2d.delete(sin_id)
            astra.projector.delete(proj_id)

    astra.data2d.clear()
    astra.projector.clear()
    astra.algorithm.clear()

    if progress_callback is not None:
        running = progress_callback(-1)

    return sinograms_simulated


def apply_tube_correction_to_sinograms(sinograms, sphere_radius=1):
    x = np.linspace(-1, 1, num=sinograms.shape[1])
    filter_x = np.nan_to_num(np.sqrt(1 - (x / sphere_radius) ** 2))
    sinograms[:] = mult_along_axis(sinograms[:], filter_x, 1)
    return sinograms, filter_x


def filter_1d(sinograms):
    result = np.zeros(sinograms.shape)
    result[:, 1:] = (sinograms[:, 1:] - sinograms[:, :-1]) ** 2
    return result


def iterative_alignment(
    sinograms_raw,
    shifts,
    binning,
    angles,
    method="FBP",
    projector="strip",
    iterations=2,
    backpropagation_iterations=10,
    det_width=1.0,
    constr=None,
    use_gpu=False,
    main_progress_callback=None,
    inner_progress_callback=None,
):
    sinograms_binned = (
        sinograms_raw.reshape(
            sinograms_raw.shape[0],
            sinograms_raw.shape[1] // binning,
            binning,
            sinograms_raw.shape[2] // binning,
            binning,
        )
        .mean(axis=2)
        .mean(axis=-1)
    )

    volume = np.zeros((sinograms_binned.shape[1], sinograms_binned.shape[1], sinograms_binned.shape[2]))

    shifts_progress = [shifts * 1]

    for i in range(iterations):
        if main_progress_callback is not None:
            running = main_progress_callback(i)
            if not running:
                break
        sinograms_shifted = sinograms_binned * 1
        shift_sinograms(
            sinograms_binned, sinograms_shifted, shifts // binning, progress_callback=inner_progress_callback
        )

        cut_sphere_radius = 1 - 2 * np.max(np.abs(shifts[:, 1])) / sinograms_raw.shape[1]

        apply_tube_correction_to_sinograms(sinograms_shifted, cut_sphere_radius)

        sinograms_simulated = sinograms_shifted * 0

        get_simulated_sinograms(
            sinograms_shifted,
            sinograms_simulated,
            volume,
            angles,
            indices=[60],
            method=method,
            projector=projector,
            iterations=backpropagation_iterations,
            det_width=det_width,
            constr=constr,
            use_gpu=use_gpu,
            progress_callback=inner_progress_callback,
            sphere_radius=cut_sphere_radius,
        )

        sinograms_binned_filtered, filter = apply_tube_correction_to_sinograms(
            filter_1d(sinograms_binned), cut_sphere_radius
        )
        sinograms_simulated_filtered, filter = apply_tube_correction_to_sinograms(
            filter_1d(sinograms_simulated), cut_sphere_radius
        )

        shifts = align_pairs_images(
            sinograms_binned_filtered,
            sinograms_simulated_filtered,
            method=shift_measurements.Method.CrossCorr,
            progress_callback=inner_progress_callback,
            upsample_factor=4,
        )
        shifts_progress.append(shifts * binning)
    return volume, shifts_progress


def get_convolution_core(
    shape,
    angles,
    method="FBP",
    projector="strip",
    iterations=10,
    det_width=1.0,
    constr=None,
    progress_callback=None,
    use_gpu=False,
):
    sinograms = np.zeros((len(angles), shape[0], 1))
    if shape[0] % 2 == 0:
        sinograms[:, shape[0] // 2 : shape[0] // 2 + 2] = 0.5
    else:
        sinograms[:, shape[0] // 2] = 1

    volume = np.zeros((shape[0], shape[1], 1))

    reconstruct_2d(
        sinograms,
        volume,
        angles,
        indices=[0],
        method=method,
        projector=projector,
        iterations=iterations,
        det_width=det_width,
        constr=constr,
        progress_callback=progress_callback,
        use_gpu=use_gpu,
    )
    return volume


def deconvolute_volume(convolution_core, volume, indices=None, R_limit=0.2, a=0.01, b=0.2, progress_callback=None):
    X, Y = np.meshgrid(
        np.linspace(-1, 1, num=convolution_core.shape[0]), np.linspace(-1, 1, num=convolution_core.shape[0])
    )
    R = X**2 + Y**2
    mask = np.cos(R**2 / R_limit**2 * np.pi / 2) ** 2
    mask[R > R_limit] = 0

    probe = mask * convolution_core
    fprobe = np.fft.fftshift(np.fft.fft2(probe))

    if indices is None:
        indices = range(volume.shape[2])

    volume2 = volume * 1  # TODO might want to do it in-place

    for i in indices:
        if progress_callback is not None:
            running = progress_callback(i)
            if not running:
                break

        dump = np.conjugate(fprobe) / (np.abs(fprobe) ** 2 + a + b * R)
        fimg = np.fft.fftshift(np.fft.fft2(volume[:, :, i]))

        volume2[:, :, i] = np.abs(np.fft.fftshift(np.fft.ifft2(fimg * dump)))

    return volume2


def cut_tube(volume, sphere_radius=1):
    x = np.linspace(-1, 1, num=volume.shape[1])
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2
    mask = R2 > sphere_radius**2
    for i in range(volume.shape[2]):
        volume[:, :, i][mask] = 0
    return volume
