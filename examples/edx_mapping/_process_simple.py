import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from PIL import Image
import numba
import scipy.signal
from stem_measurements import edx_processing

data_folder = "c:/temp/GaN/GaN_0.5um/raw_data/"
save_folder = "c:/temp/GaN//GaN_0.5um"
file_name = "LamellaGaN_0.5um"

elements = ["Ga", "Cu", "Al", "N", "O", "Au", "Mo", "Ti"]

total_frames = 720

edx_data = []
BFs = []

# -------------------------------
print("loading data")
for i in range(total_frames):
    with open(f"{data_folder}{file_name}_{i}.pdat", "rb") as f:
        (header, data, s0) = pickle.load(f)
    edx_data.append(data["edxData"]["EDX0"]["Energy"])
    edx_data.append(data["edxData"]["EDX1"]["Energy"])
    BFs.append(data["stemData"]["BF"].reshape(1024, 1024))

# to show first summed and last image:
# f, ax = plt.subplots(1, 3, sharex=True, sharey=True)
# ax[0].imshow(BFs[0])
# ax[1].imshow(np.sum(BFs, axis=0))
# ax[2].imshow(BFs[-1])
# plt.show()

energies = np.concatenate([e[0] for e in edx_data])
pixels = np.concatenate([e[1] for e in edx_data])


def generate_base_function(energy, energies, MnFWHM=130):
    def gauss(x, *p):
        A, mu, sigma = p
        return A/sigma*np.exp(-(x - mu)**2/(2.*sigma**2))

    mu = energy
    sigma = MnFWHM/2.3548/np.sqrt(5900.3)*np.sqrt(mu)
    result = gauss(energies, 1, mu, sigma)

    return result/np.sum(result)


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
median_filter = 5
dilation = 2
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

f, ax = plt.subplots(4, 5, sharex=True, sharey=True)
i = 0
for name in imgs_all.keys():
    img = imgs_all[name]
    ax[i%4, i//4].set_title(name)
    ax[i%4, i//4].imshow(img)
    i += 1
plt.show()

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
