import numpy as np
from numba import njit

from expert_pi.measurements.orientation import point_group_mesh_generator as mesher


@njit
def find_local_peaks(image, M, K, IJ_max, IJ_min, V_max, V_min, P, mask):
    """

    :param image: input data (NxN)    square
    :param M: half distance of squares to find min/max
    :param K: N//2
    :param IJ_max: cache array of IJ indices of max values
    :param IJ_min: cache array of IJ indices of min values
    :param V_max: cache array of max values
    :param V_min: cache array of min values
    :param P: cache array of prominence values
    :param mask: cache array for masking true local maxima
    :return:
    """

    # construct min/max for each MxM grid
    for i in range(K):
        for j in range(K):
            ij = np.argmax(image[i*M:(i + 1)*M, j*M:(j + 1)*M])
            IJ_max[i, j, 0] = ij//M + i*M
            IJ_max[i, j, 1] = ij%M + j*M
            V_max[i, j] = image[IJ_max[i, j, 0], IJ_max[i, j, 1]]

            ij = np.argmin(image[i*M:(i + 1)*M, j*M:(j + 1)*M])
            IJ_min[i, j, 0] = ij//M + i*M
            IJ_min[i, j, 1] = ij%M + j*M
            V_min[i, j] = image[IJ_min[i, j, 0], IJ_min[i, j, 1]]

    # filter true local min max by comparing neighbours
    for i in range(1, K - 1):
        for j in range(1, K - 1):
            v_max = np.max(V_max[i - 1:i + 2, j - 1:j + 2])
            v_min = np.min(V_min[i - 1:i + 2, j - 1:j + 2])
            P[i, j] = (v_max - v_min)
            if V_max[i, j] == v_max:
                mask[i, j] = True

    # filter local maxima which are too close
    for i in range(1, K - 1):
        for j in range(1, K - 1):
            if mask[i, j] and np.sum(mask[i - 1:i + 2, j - 1:j + 2]) > 1:
                mask[i - 1:i + 2, j - 1:j + 2] = False
                mask[i, j] = True


def fit_diffraction_patterns(images, angular_fov, minimal_spot_distances, relative_prominence=0.01, absolute_prominence=100):
    """

    :param images: 3d numpy array: (num_images,N,N) #16bit
    :param angular_fov: full angluar fov of images in mrads
    :param minimal_spot_distances: in mrads
    :param prominence: relative value to search maxima
    :return: indices of brag peaks, absolute and relative values
    """

    N = images.shape[1]  # assuming square
    M = int(max(1, np.round(minimal_spot_distances/2/angular_fov*N)))
    K = N//M

    # allocate memory for the calculation:
    IJ_max = np.zeros((K, K, 2), dtype="int64")
    IJ_min = np.zeros((K, K, 2), dtype="int64")
    V_max = np.zeros((K, K), dtype="float")
    V_min = np.zeros((K, K), dtype="float")
    mask = np.zeros((K, K), dtype="bool")
    P = np.zeros((K, K), dtype="float")

    # cache arrays initialization:

    results = []

    # test speed:
    # start = time()

    for image in images:  # TODO speed up by paralerizing + compiling
        find_local_peaks(image, M, K, IJ_max, IJ_min, V_max, V_min, P, mask)  # inplace calculation
        P_relative = P[mask]
        V_max_masked = V_max[mask]
        P_relative[V_max_masked > 0] /= V_max_masked[V_max_masked > 0]
        mask_filter = (P_relative > relative_prominence) & (P[mask] > absolute_prominence)
        IJs = (IJ_max[mask, :])[mask_filter, :]
        P_masked = P_relative[mask_filter]
        V_max_masked = (V_max[mask])[mask_filter]

        xy = (IJs - N//2)/N*angular_fov

        results.append([xy[:, [1, 0]], P_masked, V_max_masked])

    # t = time() - start
    # print(f"time to calculate per image: {t/len(images)*1000:6.2f} ms ({len(images)}x)")

    return results


def to_orientations(points):
    proj_z = points
    proj_x = points*0 + np.array([0, 0, -1])
    mask = np.abs(proj_z[:, 2]) > 1 - 1e-6
    proj_x[mask] = np.cross(np.array([0, 1, 0]), proj_z[mask, :])
    proj_y = np.cross(proj_z, proj_x)
    proj_x = np.cross(proj_y, proj_z)
    proj_x = proj_x/np.linalg.norm(proj_x, axis=1).reshape(-1, 1)
    proj_y = proj_y/np.linalg.norm(proj_y, axis=1).reshape(-1, 1)
    proj_z = proj_z/np.linalg.norm(proj_z, axis=1).reshape(-1, 1)

    orientations = np.dstack([proj_x, proj_y, proj_z])
    return orientations


@njit
def highlight_diffraction_by_fitting(qx, qy, qI, px, py, pI, k_error):
    a2 = k_error**2
    qI2 = qI*0
    pI2 = pI*0
    for j in range(px.shape[0]):
        for i in range(qx.shape[0]):
            d2 = (qx[i] - px[j])**2 + (qy[i] - py[j])**2
            f = np.sqrt(d2/a2)
            if d2 < a2:
                qI2[i] += qI[i]*pI[j]*(1 - f**0.5)
                pI2[j] += qI[i]*pI[j]*(1 - f**0.5)
    return qI2, pI2


@njit
def calculate_correlation_fast(qx, qy, qI, px, py, pI, rotations, k_error):
    # assuming qI and pI already normalized
    k_error2 = k_error**2
    result = 0*rotations[0, 0, :]
    for i in range(qx.shape[0]):
        for j in range(px.shape[0]):
            r2 = px[j]**2 + py[j]**2
            d2 = (rotations[0, 0, :]*qx[i] + rotations[0, 1, :]*qy[i] - px[j])**2 + (rotations[1, 0, :]*qx[i] + rotations[1, 1, :]*qy[i] - py[j])**2
            mask = d2 < k_error2
            result[mask] += qI[i]*pI[j]*(1 - d2[mask]/k_error2)
    i = np.argmax(result)
    return i, result


def generate_all_diffraction_patterns(structure, group_name, generator_function, angular_step=10/180*np.pi):
    vectors = mesher.orientation_ranges[group_name]
    points, edges, triangles, colors = mesher.get_mesh([mesher.normalize(v) for v in vectors], angular_step=angular_step)

    orientations = to_orientations(points)

    qxs = []
    qys = []
    qIs = []

    rotations = np.linspace(0, 2*np.pi, num=int(2*np.pi/angular_step), endpoint=False)
    rotations_matrix = np.array([[np.cos(rotations), np.sin(rotations)],
                                 [-np.sin(rotations), np.cos(rotations)]])

    for i, M in enumerate(orientations):
        bragg_peaks = generator_function(M)
        intensity = bragg_peaks.data["intensity"]

        bragg_peaks.data["qx"]

        center = np.argwhere((bragg_peaks.data["h"] == 0) & (bragg_peaks.data["k"] == 0) & (bragg_peaks.data["l"] == 0))
        bragg_peaks.data["intensity"][center] = 0
        qxs.append(bragg_peaks.data["qx"])
        qys.append(bragg_peaks.data["qy"])
        qIs.append(bragg_peaks.data["intensity"])
        # qIs[-1]=qIs[-1]/np.sqrt(np.sum(qIs[-1]**2))

    return qxs, qys, qIs, vectors, points, edges, triangles, colors, rotations, rotations_matrix


def get_correlation_map(qxs, qys, qIs, px, py, pI, rotations_matrix, k_error):
    coefs_all = []
    coefs_zone = []

    for i in range(len(qxs)):
        rot_i, results = calculate_correlation_fast(qxs[i], qys[i], qIs[i], px, py, pI, rotations_matrix, k_error)
        coefs_all.append(results)
        coefs_zone.append(results[rot_i])

    return coefs_zone, coefs_all


def get_orientations_py4DSTEM(structure, qxy, intensities):
    import py4DSTEM  # lazy load
    dtype = [
        ('qx', 'float64'),
        ('qy', 'float64'),
        ('intensity', 'float64')
    ]
    data = np.array([(qxy[i][0], qxy[i][1], intensities[i]) for i in range(len(qxy))], dtype=dtype)
    bragg_peaks = py4DSTEM.io.datastructure.emd.pointlistarray.PointList(data)

    return structure.match_single_pattern(bragg_peaks)


def fit_orientation(image, angular_fov, structure, minimal_spot_distance=3, relative_prominence=0.01, absolute_prominence=100, method="py4DSTEM"):
    N = image.shape[0]  # assume 512x512 camera image
    xys, V_max, P = fit_diffraction_patterns(np.array([image]), angular_fov, minimal_spot_distance, relative_prominence, absolute_prominence)[0]

    qxy = xys/1000/structure.wavelength  # to reciprocal A
    if method == "py4DSTEM":
        return get_orientations_py4DSTEM(structure, qxy, V_max), xys, V_max, P
