import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xraydb as xdb
from matplotlib import colors
from PIL import Image
from scipy.optimize import curve_fit
from skimage import exposure


def get_edx_filters(elements=[], lines=["Ka1", "La1", "Ma"]):
    filters = {}
    for element in elements:
        ls = element.split(" ")
        if len(ls) > 1:
            actual_lines = [ls[1]]
            element = ls[0]
        else:
            actual_lines = lines
        for line in actual_lines:
            try:
                xdb.xray_lines(element)[line]
            except:
                print(element, line, "not exist")
            else:
                energy = xdb.xray_lines(element)[line].energy
                if energy <= 30000:
                    filters[element + " " + line] = energy
                    print(element, line, energy, filters[element + " " + line])
                else:
                    print(element, line, energy, "too high energy")
    return filters


def fit_gaussian(energies, histogram, range_, plot=False):
    # Define model function to be used to fit to the data above:
    def gauss(x, *p):
        a, mu, sigma = p
        return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    mask = (energies > range_[0]) & (energies <= range_[1])

    x = energies[mask]
    y = histogram[mask]

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [np.max(y), np.mean(range_), 130]

    coeff, _var_matrix = curve_fit(gauss, x, y, p0=p0)

    # Get the fitted curve
    y_fitted = gauss(x, *coeff)

    mn_fwhm = 2.3548 * coeff[2] / np.sqrt(coeff[1]) * np.sqrt(5900.3)

    if plot:
        _f, ax = plt.subplots()
        ax.plot(energies, histogram)
        ax.plot(x, y_fitted)
        ax.set_title(f"energy: {coeff[1]:6.1f}eV MnFWHM {mn_fwhm:6.1f}eV")

    return mn_fwhm, coeff


# def generate_element_spectrum(element, energies, MnFWHM=130):
#     lines = xdb.xray_lines(element)
#
#     def gauss(x, *p):
#         A, mu, sigma = p
#         return A/sigma*np.exp(-(x - mu)**2/(2.*sigma**2))
#
#     result = energies*0
#     for line in lines.values():
#         mu = line.energy
#         sigma = MnFWHM/2.3548/np.sqrt(5900.3)*np.sqrt(mu)
#         result += gauss(energies, line.intensity, mu, sigma)
#
#     return result/np.sum(result)


def auto_elements(energies, histogram, n=10):
    d_e = energies[1] - energies[0]
    peaks = []
    prominence = 0.5

    while len(peaks) < n:
        prominence *= 0.8
        peaks = (
            scipy.signal.find_peaks(histogram / np.max(histogram), distance=60 / d_e, prominence=prominence)[0] * d_e
        )

    return find_closest(peaks)


def find_closest(energies):
    energies_lines, names = get_all_lines_sorted()
    result = []
    for energy in energies:
        result.append(names[np.argmin((energies_lines - energy) ** 2)])
    return result


def get_all_lines_sorted(energy_range=[1000, 20000], line_names=["Ka1", "La1", "Ma"]):
    energies = []
    names = []
    for i in range(1, 98):
        lines = xdb.xray_lines(i)
        element = xdb.atomic_symbol(i)
        for name, line in lines.items():
            if line.energy > energy_range[0] and line.energy < energy_range[1]:
                if name in line_names:
                    names.append(element + " " + name)
                    energies.append(line.energy)

    energies, names = zip(*sorted(zip(energies, names)))
    return energies, names


def create_histogram(header, data, bins=1024, range=[0, None], detectors=("EDX0", "EDX1")):
    energies = []

    for name, item in data["edxData"].items():
        if name not in detectors:
            continue
        energies.append(item["energy"][0])

    energies = np.concatenate(energies)

    if range[1] is None:
        sorted_events = np.sort(energies)
        range[1] = sorted_events[int(len(sorted_events) * 0.95)]

    hist, bin_edges = np.histogram(energies, bins=bins, range=range, density=True)

    width = 1 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, center, width


def show_edx_histogram(header, data, elements=[], bins=1024, range=[0, None], detectors=["EDX0", "EDX1"]):
    _f = plt.figure()

    hist, center, width = create_histogram(header, data, bins, range, detectors)

    plt.bar(center, hist, align="center", width=width)

    max_count = np.max(hist)

    if len(elements) > 0:
        filters = get_edx_filters(elements)
        for name, energy in filters.items():
            index = np.argmin(np.abs(center - energy))
            plt.text(
                energy,
                hist[index] + 0.05 * max_count,
                name + f"\n {int(energy)} eV",
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            plt.plot([energy, energy], [0, hist[index] + 0.02 * max_count], color="black")

    plt.xlim(range[0], range[1])
    plt.xlabel("Energy (eV)")
    plt.ylabel("Counts")


def show_edx_deadtime_map(header, data, frame=None):
    # over all frames

    frame_pixels = header["scanDimensions"][1] * header["scanDimensions"][2]
    _f, ax = plt.subplots(1, len(data["edxData"].keys()))
    i = 0
    for name, item in data["edxData"].items():
        _values = item["dead_time"][0]
        pixels = item["dead_time"][1] % frame_pixels
        if frame is not None:
            pixel_mask = (item["dead_time"][1] > frame * frame_pixels) & (
                item["dead_time"][1] < (frame + 1) * frame_pixels
            )
        mask = (pixels == pixels) & pixel_mask
        desc = "dead_time"

        pixels2, counts = np.unique(pixels[mask], return_counts=True)
        img = np.zeros((header["scanDimensions"][1], header["scanDimensions"][2]))
        img.flat[pixels2] += counts
        ax[i].imshow(img)
        ax[i].title.set_text(name + desc)

        i += 1


def show_edx_map(
    header, data, elements=[], frame=None, save_folder=None, detector="EDX0", energy_resolution=140, median_filter=0
):
    color_list = ["w", "b", "g", "r", "c", "m", "y", "b", "g", "r", "c", "m", "y"]
    cm = []
    n_bin = 100
    for color in color_list:
        colors_ = [colors.to_rgb("black"), colors.to_rgb(color)]  #
        cmap_name = "black_" + color
        cm.append(colors.LinearSegmentedColormap.from_list(cmap_name, colors_, N=n_bin))

    filters = {"All": [0, np.inf]}

    if len(elements) > 0:
        energies = get_edx_filters(elements)
        for name, energy in energies.items():
            filters[name] = [energy - energy_resolution, energy + energy_resolution]
    elements = ["All"] + [el.split(" ")[0] for el in elements]

    frame_pixels = header["scanDimensions"][1] * header["scanDimensions"][2]
    size = int(len(filters) / 2 + 0.5)
    dim = 2
    if len(filters) == 1:
        f, ax = plt.subplots(1, 1, figsize=(2 * dim, 2 * dim))
        ax = [[ax]]
    elif size == 1:
        f, ax = plt.subplots(2, size, figsize=(size * dim, 2 * dim))
        ax = [[a] for a in ax]
    else:
        f, ax = plt.subplots(2, size, figsize=(size * dim, 2 * dim))

    i = 0
    for name, filter in filters.items():
        item = data["edxData"][detector]

        values = item["energy"][0]
        pixels = item["energy"][1] % frame_pixels
        if frame is not None:
            pixel_mask = (item["energy"][1] > frame * frame_pixels) & (item["energy"][1] < (frame + 1) * frame_pixels)
        else:
            pixel_mask = pixels == pixels
        mask = (values > filter[0]) & (values < filter[1]) & pixel_mask
        _desc = name + f" ({filter[0]:.0f} - {filter[1]:.0f} eV)"

        pixels2, counts = np.unique(pixels[mask], return_counts=True)
        img = np.zeros((header["scanDimensions"][1], header["scanDimensions"][2]))
        img.flat[pixels2] += counts

        if median_filter > 0:
            img = scipy.signal.medfilt2d(img, median_filter)

        img_eq = exposure.equalize_hist(img)

        if save_folder is not None:
            im = Image.fromarray((img / np.max(img) * (2**16 - 1)).astype("uint16"))
            im.save(save_folder + "/" + str(name) + ".tif")

        ax[i // size][i % size].imshow(img_eq, cm[elements.index(name.split()[0])])
        if name == "All":
            label = name
        else:
            label = name + " " + str(energies[name]) + " eV"
        ax[i // size][i % size].title.set_text(label)
        ax[i // size][i % size].set_axis_off()

        i += 1

    plt.tight_layout()
    if "stemData" in data:
        for signal in data["stemData"]:
            plt.figure()
            item = data["stemData"][signal]
            img = item.reshape(header["scanDimensions"][1], header["scanDimensions"][2])
            plt.imshow(img, "Greys_r")
            if save_folder is not None:
                im = Image.fromarray(img)
                im.save(save_folder + "/" + signal + ".tif")
