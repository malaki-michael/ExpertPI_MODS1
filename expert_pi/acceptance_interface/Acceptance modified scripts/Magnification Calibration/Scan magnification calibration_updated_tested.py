import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage, scipy.signal, scipy.stats as st
import time
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import cv2 as cv2
#from expert_pi.grpc_client.modules.scanning import DetectorType as DT
from expert_pi.controllers import scan_helper  ################test this line for new ExpertPi version
from expert_pi.grpc_client.modules.scanning import DetectorType as DT
#from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
#from expert_pi import grpc_client, scan_helper
#from expert_pi.controllers import scan_helper  ################test this line for new ExpertPi version

def find_angles(img, fov, min_dist_for_peri=0.2, max_dist_for_peri=0.65, med_filt_size=3, expected_grid_period=0.4629, min_line_separ=0.05, ignored_lines=1, show_img=True,
                show_distances_histogram=False, n_thetas=300):
    img = scipy.signal.medfilt2d(img.astype('float64'), kernel_size=med_filt_size)

    if fov > 6:
        edges = canny(img, sigma=3, low_threshold=0.75, high_threshold=0.81, use_quantiles=True)
    else:
        edges = canny(img, sigma=3, low_threshold=0.77, high_threshold=0.82, use_quantiles=True)

    edges[0:4, :] = False
    edges[-4:, :] = False
    edges[:, 0:4] = False
    edges[:, -4:] = False

    dangle = 10*np.pi/180
    abins = np.arange(-np.pi/2, np.pi/2, dangle)
    abins = np.concatenate((abins, np.array([np.pi/2])), axis=None)

    hspace_orig, hangles, distances = hough_line(edges, theta=np.linspace(-np.pi/2, np.pi/2, 360))
    accum, angles2, distances2 = hough_line_peaks(hspace_orig, hangles, distances, num_peaks=4*np.floor(fov/expected_grid_period).astype(np.int16), threshold=0.1*np.amax(hspace_orig))

    for _ in range(ignored_lines):
        i_max = np.argmax(accum)
        mask = np.ones(len(accum), dtype=bool)
        mask[i_max] = False
        accum = accum[mask]
        angles2 = angles2[mask]

    first_accum, bins = np.histogram(angles2, bins=abins, weights=accum)
    center_angles = 0.5*bins[0:-1] + 0.5*bins[1:]
    peaks_locs, prop = scipy.signal.find_peaks(np.concatenate(([min(first_accum)], first_accum, [min(first_accum)])),
                                               distance=2, height=1)
    peaks_locs = peaks_locs - 1

    isort = np.flip(np.argsort(prop["peak_heights"]))
    first_accum = first_accum*np.cos(2*(center_angles - center_angles[peaks_locs[isort[0]]]))**2
    peaks_locs, prop = scipy.signal.find_peaks(np.concatenate(([min(first_accum)], first_accum, [min(first_accum)])),
                                               distance=2, height=1)
    peaks_locs = peaks_locs - 1

    if len(peaks_locs) < 2:
        print("Fewer than 2 angle intervals detected.")

    isort = np.flip(np.argsort(prop["peak_heights"]))
    if np.abs(center_angles[peaks_locs[isort[1]]] - center_angles[peaks_locs[isort[0]]]) > 169*np.pi/180 and len(peaks_locs) > 2:
        isort[1] = isort[2]
    if center_angles[peaks_locs[isort[0]]] > center_angles[peaks_locs[isort[1]]]:
        isort_old = isort.copy()
        isort[0] = isort_old[1]
        isort[1] = isort_old[0]

    angles_approx = np.array([center_angles[peaks_locs[isort[0]]], center_angles[peaks_locs[isort[1]]]])

    angles = np.zeros(2)
    periods = np.zeros(2)
    cia = np.zeros(2)  # confidence interval of angles
    cir = np.zeros(2)  # of periods

    if show_img:
        fig, ax = plt.subplots()
        fig.set_figwidth(6.4)
        fig.set_figheight(7)
        ax.imshow(img, extent=[0, img.shape[0], 0, img.shape[1]], origin='lower', vmin=np.mean(img) - 2*np.std(img), vmax=np.mean(img) + 2*np.std(img), cmap='gray')
        x = np.linspace(0, img.shape[0], 10)

    for ipeak in range(2):
        thetas_forhl = np.linspace(angles_approx[ipeak] - dangle, angles_approx[ipeak] + dangle, n_thetas)
        hspace, hangles, distances = hough_line(edges, theta=thetas_forhl)
        accumA, anglesA, distancesA = hough_line_peaks(hspace, hangles, distances, num_peaks=np.floor(fov/expected_grid_period).astype(np.int16),
                                                       threshold=0.1*np.amax(hspace), min_distance=np.ceil(min_line_separ*img.shape[0]/fov).astype(np.int16),
                                                       min_angle=np.ceil(10*np.pi/(180*(thetas_forhl[2] - thetas_forhl[1]))).astype(np.int16))

        angles[ipeak] = np.median(anglesA)  # np.average(anglesA, weights=accumA)
        sda = np.sqrt(np.cov(anglesA, bias=True, aweights=accumA))
        cia[ipeak] = st.t.interval(confidence=0.95, df=len(anglesA) - 1, loc=0, scale=1)[1]*sda/np.sqrt(len(anglesA))

        mask = np.abs(anglesA - angles[ipeak]) < 0.2*np.pi/180
        distancesA_masked = distancesA[mask]
        accumA_masked = accumA[mask]
        anglesA_masked = anglesA[mask]

        dist_diffsA = []
        rweightA = []
        for i in range(len(distancesA_masked)):
            for j in range(i + 1, len(distancesA_masked)):
                distdiff = np.abs(distancesA_masked[i] - distancesA_masked[j])*fov/len(img[:, 0])
                if min_dist_for_peri < distdiff and distdiff < max_dist_for_peri:
                    dist_diffsA.append(distdiff)
                    rweightA.append(accumA_masked[i]*accumA_masked[j])
        dist_diffsA = np.array(dist_diffsA)
        rweightA = np.array(rweightA)

        periods[ipeak] = np.average(dist_diffsA, weights=rweightA)  # np.median(dist_diffsA)  # np.average(dist_diffsA, weights=rweightA)
        sdr = np.sqrt(np.cov(dist_diffsA, bias=True, aweights=rweightA))
        cir[ipeak] = st.t.interval(confidence=0.95, df=len(dist_diffsA) - 1, loc=0, scale=1)[1]*sdr/np.sqrt(len(dist_diffsA))

        if show_distances_histogram:
            dist_diffsA = []
            rweightA = []
            for i in range(len(distancesA_masked)):
                for j in range(i + 1, len(distancesA_masked)):
                    distdiff = np.abs(distancesA_masked[i] - distancesA_masked[j])*fov/len(img[:, 0])
                    dist_diffsA.append(distdiff)
                    rweightA.append(accumA_masked[i]*accumA_masked[j])
            dist_diffsA = np.array(dist_diffsA)
            rweightA = np.array(rweightA)

            rbins = np.arange(0, fov, 0.1)
            plt.figure()
            plt.hist(dist_diffsA, bins=rbins, weights=rweightA)
            plt.xlabel("Distance of lines (um)")
            plt.ylabel("Weighted counts")

        if show_img:
            for j in range(0, len(anglesA_masked)):
                rho = distancesA_masked[j]
                theta = anglesA_masked[j]
                c = np.cos(theta)
                s = np.sin(theta)
                if s == 0:
                    s = 0.1e-10
                y = -c*x/s + rho/s
                if ipeak == 0:
                    ax.plot(x, y, color='red')
                else:
                    ax.plot(x, y, color='magenta')

    if show_img:
        ax.set_xlim(-50, img.shape[0] + 50)
        ax.set_ylim(-50, img.shape[1] + 50)
        ax.set_aspect(1)
        tit = f'norm_angle1 = {float(angles[0]*180/np.pi):.3f} +- {float(cia[0]*180/np.pi):.3f} deg\nnorm_angle2 = {float(angles[1]*180/np.pi):.3f} +- {float(cia[1]*180/np.pi):.3f} deg\nperi1 = {float(periods[0]):.4f} +- {float(cir[0]):.4f} um\nperi2 = {float(periods[1]):.4f}+- {float(cir[1]):.4f} um'
        ax.set_title(tit)
        plt.show()

    return angles, periods,cir


def scan_mag_test(fov=None, n_pixels=1024, pixel_time=500, real_grid_period=0.4629, min_dist_for_peri=0.2,
				  max_dist_for_peri=0.65, show_distances_histogram=True, return_values=False):
	from expert_pi.grpc_client.modules.scanning import DetectorType as DT
	from expert_pi.stream_clients import cache_client
	from expert_pi.controllers import scan_helper
    from expert_pi import grpc_client
	if fov is None:
		fov = 12 * 1e-6  # FOV in units of meters
	grpc_client.scanning.set_field_width(fov * 1e-6)  # units of meters
	scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time * 1e-9, total_size=n_pixels, frames=1,
											   detectors=[DT.BF])
	header, data = cache_client.get_item(scan_id, n_pixels ** 2)
	img = data["stemData"]["BF"]
	img = np.reshape(img, (n_pixels, n_pixels))
	angles, periods, cir = find_angles(img, fov=fov, min_dist_for_peri=min_dist_for_peri,
									   max_dist_for_peri=max_dist_for_peri, expected_grid_period=real_grid_period,
									   show_distances_histogram=show_distances_histogram, show_img=True)

	if return_values:
		return angles, periods, cir
	else:
		print("Angle diff is {0:.4f} deg".format(np.abs(angles[0] - angles[1]) * 180 / np.pi))
		print("Peri1 is {0:.4f} um".format(periods[0]))
		print("Peri2 is {0:.4f} um".format(periods[1]))
		perc_1, perc_2 = cir[0] / periods[0], cir[1] / periods[1]
		print("error percentage 1 is {0:.2f}%".format(perc_1 * 100))
		print("error percentage 2 is {0:.2f}%".format(perc_2 * 100))


scan_mag_test(fov=10, n_pixels=1024, real_grid_period=0.4629)