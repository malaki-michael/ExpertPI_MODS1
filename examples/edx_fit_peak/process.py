import os
import pickle
import matplotlib.pyplot as plt
from stem_measurements import edx_processing

folder = "c:/temp/"
file_name = "edx_spectra.pdat"

with open(folder + file_name, "rb") as f:
    header, data = pickle.load(f)

# Cu 8046V
range_ = [8046 - 3*140, 8046 + 3*140]
for detector in ["EDX0", "EDX1"]:
    histogram, energies, width = edx_processing.create_histogram(header, data, bins=100, range=range_, detectors=[detector])
    edx_processing.fit_gaussian(energies, histogram, range_, plot=True)

plt.show()
