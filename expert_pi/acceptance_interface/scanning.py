
import multiprocessing as mp
import cv2
import piexif
from tkinter import simpledialog
import tkinter as tk
import numpy as np
import scipy.signal,scipy.ndimage
import scipy.stats as st
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt

from expert_pi.acceptance_interface.specification_criteria import scan_calibration_error
from expert_pi.acceptance_interface import specification_criteria

def askfloat_helper(*args,**kwargs):
    root = tk.Tk()
    root.withdraw()
    result_queue = kwargs.pop('result_queue')
    kwargs['parent'] = root
    result_queue.put(simpledialog.askfloat(prompt="FOV is missing from metadata, please add the scan x-axis FOV in microns",title="Scan distortion test",*args,**kwargs))
    root.destroy()


def ask_float(*args, **kwargs):
    result_queue = mp.Queue()
    kwargs['result_queue'] = result_queue
    askfloat_thread = mp.Process(target=askfloat_helper,args=args,kwargs=kwargs)
    askfloat_thread.start()
    result = result_queue.get()
    if askfloat_thread.is_alive():
        askfloat_thread.join()
    return result


def check_calibration(perc_1, perc_2, scan_calibration_error):

    spec_pass = False  # Default to False unless conditions are met
    if perc_1 < scan_calibration_error:
        if perc_2 < scan_calibration_error:
            spec_pass = True  # Both conditions passed

    return spec_pass


def find_angles(img, fov, min_dist_for_peri=0.2, max_dist_for_peri=0.65, med_filt_size=3,
                expected_grid_period=0.4629, min_line_separ=0.05, ignored_lines=1, n_thetas=300):

    shared_directory = specification_criteria.shared_directory
    # Apply median filter to the image
    img_original = img
    img = scipy.signal.medfilt2d(img.astype('float64'), kernel_size=med_filt_size)

    # Normalize the image for display as grayscale
    img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
    img_normalized = img_normalized.astype(np.uint8)  # Convert to 8-bit for Plotly compatibility

    # Canny edge detection
    if fov > 6:
        edges = canny(img, sigma=3, low_threshold=0.75, high_threshold=0.81, use_quantiles=True)
    else:
        edges = canny(img, sigma=3, low_threshold=0.77, high_threshold=0.82, use_quantiles=True)

    # Suppress edges at the borders
    edges[0:4, :] = False
    edges[-4:, :] = False
    edges[:, 0:4] = False
    edges[:, -4:] = False

    # Angle bins and hough transform
    dangle = 10 * np.pi / 180
    abins = np.arange(-np.pi / 2, np.pi / 2, dangle)
    abins = np.concatenate((abins, np.array([np.pi / 2])), axis=None)
    hspace_orig, hangles, distances = hough_line(edges, theta=np.linspace(-np.pi / 2, np.pi / 2, 360))
    accum, angles2, distances2 = hough_line_peaks(hspace_orig, hangles, distances,
                                                  num_peaks=4 * np.floor(fov / expected_grid_period).astype(int),
                                                  threshold=0.1 * np.amax(hspace_orig))

    # Ignoring specified lines
    for _ in range(ignored_lines):
        i_max = np.argmax(accum)
        mask = np.ones(len(accum), dtype=bool)
        mask[i_max] = False
        accum = accum[mask]
        angles2 = angles2[mask]

    first_accum, bins = np.histogram(angles2, bins=abins, weights=accum)
    center_angles = 0.5 * bins[:-1] + 0.5 * bins[1:]
    peaks_locs, prop = scipy.signal.find_peaks(np.concatenate(([min(first_accum)], first_accum, [min(first_accum)])),
                                               distance=2, height=1)
    peaks_locs = peaks_locs - 1
    isort = np.flip(np.argsort(prop["peak_heights"]))
    first_accum = first_accum * np.cos(2 * (center_angles - center_angles[peaks_locs[isort[0]]])) ** 2
    peaks_locs, prop = scipy.signal.find_peaks(np.concatenate(([min(first_accum)], first_accum, [min(first_accum)])),
                                               distance=2, height=1)
    peaks_locs = peaks_locs - 1

    # Determine the two main angles
    if len(peaks_locs) < 2:
        print("Fewer than 2 angle intervals detected.")

    isort = np.flip(np.argsort(prop["peak_heights"]))
    if np.abs(center_angles[peaks_locs[isort[1]]] - center_angles[peaks_locs[isort[0]]]) > 169 * np.pi / 180 and len(
            peaks_locs) > 2:
        isort[1] = isort[2]
    if center_angles[peaks_locs[isort[0]]] > center_angles[peaks_locs[isort[1]]]:
        isort[0], isort[1] = isort[1], isort[0]

    angles_approx = np.array([center_angles[peaks_locs[isort[0]]], center_angles[peaks_locs[isort[1]]]])

    # Initialize arrays for results
    angles = np.zeros(2)
    periods = np.zeros(2)
    cir = np.zeros(2)
    cia = np.zeros(2)

    fig,ax = plt.subplots()
    ax.imshow(img_original,cmap="grey")

    x = np.linspace(0, img.shape[0], 10)
    for ipeak in range(2):
        thetas_forhl = np.linspace(angles_approx[ipeak] - dangle, angles_approx[ipeak] + dangle, n_thetas)
        hspace, hangles, distances = hough_line(edges, theta=thetas_forhl)
        accumA, anglesA, distancesA = hough_line_peaks(hspace, hangles, distances,
                                                       num_peaks=np.floor(fov / expected_grid_period).astype(int),
                                                       threshold=0.1 * np.amax(hspace),
                                                       min_distance=np.ceil(min_line_separ * img.shape[0] / fov).astype(
                                                           int),
                                                       min_angle=np.ceil(10 * np.pi / (
                                                                   180 * (thetas_forhl[2] - thetas_forhl[1]))).astype(
                                                           int))

        # Calculate angle, confidence interval, and period
        angles[ipeak] = np.median(anglesA)
        sda = np.sqrt(np.cov(anglesA, bias=True, aweights=accumA))
        cia[ipeak] = st.t.interval(confidence=0.95, df=len(anglesA) - 1, loc=0, scale=1)[1] * sda / np.sqrt(
            len(anglesA))

        # Calculate distances between detected lines for period estimation
        dist_diffsA = []
        rweightA = []
        for i in range(len(distancesA)):
            for j in range(i + 1, len(distancesA)):
                distdiff = np.abs(distancesA[i] - distancesA[j]) * fov / len(img[:, 0])
                if min_dist_for_peri < distdiff < max_dist_for_peri:
                    dist_diffsA.append(distdiff)
                    rweightA.append(accumA[i] * accumA[j])

        dist_diffsA = np.array(dist_diffsA)
        rweightA = np.array(rweightA)

        periods[ipeak] = np.average(dist_diffsA, weights=rweightA)
        sdr = np.sqrt(np.cov(dist_diffsA, bias=True, aweights=rweightA))
        cir[ipeak] = st.t.interval(confidence=0.95, df=len(dist_diffsA) - 1, loc=0, scale=1)[1] * sdr / np.sqrt(
            len(dist_diffsA))

        # Draw lines corresponding to each peak
        for j in range(len(anglesA)):
            rho = distancesA[j]
            theta = anglesA[j]
            c = np.cos(theta)
            s = np.sin(theta)
            if s == 0:
                s = 1e-10  # Avoid division by zero
            y = -c * x / s + rho / s
            color = 'red' if ipeak == 0 else 'blue'
            ax.plot(x,y,color)


    perc_1, perc_2 = cir[0] / periods[0], cir[1] / periods[1]
    ax.set_xlim(-50, img.shape[0] + 50)
    ax.set_ylim(-50, img.shape[1] + 50)
    ax.set_aspect(1)

    ax.set_title(f"Scan magnification calibration test \n Scan error x = {np.round((perc_1*100),2)}%, scan error y = {np.round((perc_2*100),2)}%, specification is {scan_calibration_error*100}%")
    fig.savefig((specification_criteria.shared_directory+"/Scan magnification test.png"), format="png", transparent=False)

    return angles, periods, cir

def scan_mag_test(filepath, fov=None, real_grid_period=0.4629, min_dist_for_peri=0.2,
                  max_dist_for_peri=0.65):
    # Load the image
    img = cv2.imread(filepath, -1)

    # Try to get FOV from metadata if FOV is not provided
    if fov is None:
        try:
            metadata_object = piexif.load(filepath)
            pixel_size = metadata_object["0th"][piexif.ImageIFD.XResolution][1] / metadata_object["0th"][piexif.ImageIFD.XResolution][0]
            fov_meters = pixel_size * img.shape[0]
            fov = fov_meters * 1e6  # Convert to microns
        except (KeyError, piexif.InvalidImageDataError):
            print("Metadata retrieval failed, proceeding to user input.")

    # Fallback to `simpledialog.askfloat()` only if FOV is still None
    if fov is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        fov = simpledialog.askfloat(
            title="Scan distortion test",
            prompt="FOV is missing from metadata, please add the scan x-axis FOV in microns",
            parent=root
        )
        root.destroy()

    # Proceed with angle, period, and calibration error processing
    try:
        angles, periods, cir = find_angles(img, fov=fov, min_dist_for_peri=min_dist_for_peri,
                                                  max_dist_for_peri=max_dist_for_peri, expected_grid_period=real_grid_period)


        perc_1, perc_2 = cir[0] / periods[0], cir[1] / periods[1]

        # Call the calibration check function
        spec_pass = check_calibration(perc_1, perc_2, scan_calibration_error)
        return spec_pass , (perc_1,perc_2)
    except ZeroDivisionError:
        print("STEM image cannot be processed, try a new region and ensure there is sufficient contrast")




#img = r"C:\Users\robert.hooley\Desktop\Rob acceptance\\10um STEM.tiff"
#result = scan_mag_test(filepath = img)
