import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from PIL import Image
import numba
import scipy.signal

from sklearn.linear_model import LinearRegression
from stem_measurements import edx_processing, shift_measurements


def generate_base_function(energy, energies, MnFWHM=130):
    def gauss(x, *p):
        A, mu, sigma = p
        return A/sigma*np.exp(-(x - mu)**2/(2.*sigma**2))

    mu = energy
    sigma = max(resolution_min, MnFWHM/np.sqrt(5900.3)*np.sqrt(mu))/2.3548
    result = gauss(energies, 1, mu, sigma)

    return result/np.sum(result)


# linear regression
def fit_intensities(energies, histogram, intensities, resolution, gainR, offset, background):
    gain = (1 + gainR/100)
    line_energies_calibrated = line_energies*gain + offset

    bases = []
    for i in range(len(line_energies_calibrated)):
        bases.append(generate_base_function(line_energies_calibrated[i], energies, MnFWHM=resolution))

    for j in range(len(background_factors)):
        bases.append(energies**background_factors[j])

    X = np.array(bases)
    y = histogram
    model = LinearRegression(fit_intercept=False).fit(X.T, y)

    background = model.coef_[-len(background):]
    intensities = model.coef_[:-len(background)]

    return intensities, resolution, gainR, offset, background


def generate_model_spectrum(energies, histogram, intensities, resolution, gainR, offset, background):
    bb = energies*0
    for i, f in enumerate(background_factors):
        bb += background[i]*energies**f
    result = bb

    gain = (1 + gainR/100)
    for i in range(len(line_energies)):
        result += intensities[i]*generate_base_function(line_energies[i]*gain + offset, energies, MnFWHM=resolution)
    return result


def tune_spectrum_parameters(energies, histogram, intensities, resolution, gainR, offset, background):
    def curve(x):
        # nonlocal intensities, energies, histogram
        resolution, gainR, offset, *background = x

        bb = energies*0
        for i, f in enumerate(background_factors):
            bb += background[i]*energies**f
        result = bb

        for i in range(len(line_energies)):
            result += intensities[i]*generate_base_function(line_energies[i]*(1 + gainR/100) + offset, energies, MnFWHM=resolution)
        return np.sum((result - histogram)**2)

    result = scipy.optimize.minimize(curve, [resolution, gainR, offset] + list(background), method="Powell")
    resolution, gainR, offset, *background = result.x
    return intensities, resolution, gainR, offset, background


def plot_fit(energies, histogram, intensities, resolution, gain, offset, background):
    predicted = generate_model_spectrum(energies, histogram, intensities, resolution, gain, offset, background)

    plt.figure()
    plt.plot(energies, histogram)
    plt.plot(energies, predicted)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Counts')
    plt.show()


def correct_data(energy_events, elements, plot=True):
    bins = 4000
    range_ = [200, 20000]
    hist, bin_edges = np.histogram(energy_events[0], bins=bins, range=range_, density=True)
    width = 1*(bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:])/2

    filters = edx_processing.get_edx_filters(elements)
    line_energies = np.array(list(filters.values()))

    background_factors = [0]

    energies = center
    histogram = hist
    intensities = []
    resolution = 130
    resolution_min = 100
    gainR = 0
    offset = 0
    background = [0]*len(background_factors)
    intensities, resolution, gainR, offset, background = fit_intensities(energies, histogram, intensities, resolution, gainR, offset, background)

    # iterative correct spectrum properties:
    for i in range(2):
        intensities, resolution, gainR, offset, background = tune_spectrum_parameters(energies, histogram, intensities, resolution, gainR, offset, background)
        print(f"resolution:{resolution} rel gain:{gainR} offset: {offset} background:{background}")
        intensities, resolution, gainR, offset, background = fit_intensities(energies, histogram, intensities, resolution, gainR, offset, background)

    if plot:
        plot_fit(energies, histogram, intensities, resolution, gainR, offset, background)

        f = plt.figure()
        plt.bar(center, hist, align='center', width=width)

        max_count = np.max(hist)

        if len(elements) > 0:
            filters = edx_processing.get_edx_filters(elements)
            for name, energy in filters.items():
                index = np.argmin(np.abs(center - energy))
                plt.text(energy, hist[index] + 0.05*max_count, name + f'\n {int(energy)} eV', horizontalalignment='center', verticalalignment='bottom')
                plt.plot([energy, energy], [0, hist[index] + 0.02*max_count], color='black')

        plt.xlim(range_[0], range_[1])
        plt.xlabel('Energy (eV)')
        plt.ylabel('Counts')

    energy_events_corrected = energy_events*1
    energy_events_corrected[0]*(1 + gainR) + offset
    return energy_events_corrected, resolution


data_folder = "c:/temp/NAND/"
save_folder = "c:/temp/NAND/"
file_name = "edx_map"

elements = ["W", "Cu", "O", "N", "Si", "Al"]

total_frames = 50

edx_data0 = []
edx_data1 = []
BFs = []
print("loading data")
for i in range(total_frames):
    with open(f"{data_folder}{file_name}_{i}.pdat", "rb") as f:
        (header, data, s0) = pickle.load(f)
    edx_data0.append(data["edxData"]["EDX0"]["Energy"])
    edx_data1.append(data["edxData"]["EDX1"]["Energy"])
    BFs.append(data["stemData"]["BF"].reshape(1024, 1024))
    print(i, "/", total_frames, end="\r")

edx_data0 = np.hstack(edx_data0)
edx_data1 = np.hstack(edx_data1)

edx0, resolution0 = correct_data(edx_data0, elements, plot=True)
edx1, resolution1 = correct_data(edx_data1, elements, plot=True)

edx = np.hstack([edx0, edx1])
resolution = (resolution0*edx0.shape[1] + resolution1*edx1.shape[1])/(edx0.shape[1] + edx1.shape[1])

# to show first summed and last image:
f, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].imshow(BFs[0])
ax[1].imshow(np.sum(BFs, axis=0))
ax[2].imshow(BFs[-1])
plt.show()

energies = edx[0]
pixels = edx[1]

# generate gaussians for each line of elements:
lines = edx_processing.get_edx_filters(elements)
E = np.array(range(0, 2**16))  # max energy 2**16 kV
bases = []
for name, value in lines.items():
    bases.append(generate_base_function(value, E))
N = len(bases)
bases = np.array(bases)

# generate basis matrix:
M = []
for i in range(N):
    M.append([])
    for j in range(N):
        M[i].append(np.sum(bases[i]*bases[j]))
Mi = np.linalg.inv(M)

# now we remap the energy events to the probabilities of the element (it is same as linear fitting):
batch = 400_000
channels = np.zeros((N, 1024*1024))

i = 0
while i < len(energies):
    probabilities = bases[:, energies[i:i + batch]]
    channels[:, pixels[i:i + batch]] += np.dot(Mi, probabilities)
    i += batch
    print(f"energies->elements {i/len(energies)*100:6.3f}%", end="\r")
channels = channels.reshape(N, 1024, 1024)

# filter images


median_filter = 9
dilation = 3
kernel = np.ones((3, 3), np.uint8)

channels_filtered = channels*0
print("filtering")
for i in range(N):
    channels_filtered[i, :, :] = cv2.dilate(channels[i, :, :], kernel, iterations=dilation)
    channels_filtered[i, :, :] = scipy.signal.medfilt2d(channels_filtered[i, :, :], median_filter)
    print(i, end="\r")

imgs_all = {}
for i in range(N):
    imgs_all[list(lines.keys())[i]] = channels_filtered[i, :, :]

# plot them

K = 3
L = 3
f, ax = plt.subplots(K, L, sharex=True, sharey=True)
i = 0
for name in imgs_all.keys():
    img = imgs_all[name]
    ax[i%K, i//K].set_title(name)
    ax[i%K, i//K].imshow(img, cmap="gray")
    i += 1

# saving:

for name, img in imgs_all.items():
    img = img.astype("float")
    img[img < 0] = 0
    im = Image.fromarray((img/np.max(img)*(2**15 - 1)).astype('uint16'))
    im.save(save_folder + '/' + str(name) + '.tif')

with open(f"{data_folder}{file_name}_{i}.pdat", "rb") as f:
    (header, data, s0) = pickle.load(f)

im = Image.fromarray(data["stemData"]["BF"].reshape(1024, 1024))
im.save(save_folder + '/BF.tif')

BF_summed = (np.sum(BFs, axis=0).reshape(1024, 1024)/total_frames).astype("uint16")
im = Image.fromarray(BF_summed)
im.save(save_folder + '/BF_summed.tif')

im = Image.fromarray(data["stemData"]["HAADF"].reshape(1024, 1024))
im.save(save_folder + '/HAADF.tif')

colored = np.zeros((BFs[0].shape[0], BFs[0].shape[1], 3))

colors = {
    'O Ka1': [1, 0, 0],
    'N Ka1': [0, 1, 0],
    'Al Ka1': [0, 0, 1],
}

for name, v in colors.items():
    img = imgs_all[name]*1
    img = img/np.max(img)
    colored += np.dstack([img*v[0], img*v[1], img*v[2]])

colored = (colored*255).astype("uint8")

im = Image.fromarray(colored)
im.save(save_folder + '/colored.tif')

ax[i%K, i//K].set_title("O:red N:green Al:blue")
ax[i%K, i//K].imshow(colored)

plt.show()
