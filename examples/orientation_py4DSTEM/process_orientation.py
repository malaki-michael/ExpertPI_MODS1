from os.path import isfile, join

from pymatgen.io.cif import CifParser
import numpy as np
import h5py
import matplotlib.pyplot as plt
from stem_measurements.orientation import bragg_fitting
from stem_measurements.orientation import point_group_mesh_generator as mesher
from time import time
import pickle
from multiprocessing import Process

print("importing py4DSTEM")
import py4DSTEM


cif_path = "C:/Tescan/expert_pi/data/cifs/"
file = "9008492_Co.cif"
path = join(cif_path, file)

energy = 100_000
angular_fov = 300e-3  # rad
minimal_spot_distance = 4.5e-3  # rad
relative_prominence = 0.01
absolute_prominence = 50
file = "C:/data/2022_11_29_NiCo_P150/series3.h5"
output_folder = "C:/data/2022_11_29_NiCo_P150/series3_processed/"

try:
    with open(output_folder + "cache_structure.pdat", "rb") as f:
        structure = pickle.load(f)
except:
    print("loading structure")
    structure = py4DSTEM.process.diffraction.Crystal.from_CIF(path)
    structure.setup_diffraction(energy)
    k_max = angular_fov/structure.wavelength
    structure.calculate_structure_factors(k_max, tol_structure_factor=-1.0)
    print("generate orientation plan")
    structure.orientation_plan(zone_axis_range="auto", accel_voltage=energy)

    with open(output_folder + "cache_structure.pdat", "wb") as f:
        pickle.dump(structure, f)

cache = h5py.File(file, 'r')

orientations = {}
start = time()


def calculate(id, i_range):
    for i in i_range:
        print(id, "fitting", i, f"{time() - start:6.2f} s")
        image = cache["data"][i].reshape(512, 512)
        orientation, xys, V_max, P = bragg_fitting.fit_orientation(image, angular_fov*1e3, structure, minimal_spot_distance*1e3, relative_prominence, absolute_prominence, method="py4DSTEM")
        orientations[i] = orientation.matrix[0]
        print(len(V_max), orientation.matrix[0])
        with open(output_folder + f"{id:04}.pdat", "wb") as f:
            pickle.dump(orientations, f)



n_threads = 8
total_size = len(cache["data"])

ps = []
for i in range(n_threads):
    i_range = range(i*total_size//n_threads, min(total_size, (i + 1)*total_size//n_threads))
    print(i, i_range)
    p = Process(target=calculate, args=(i, i_range))
    ps.append(p)

if __name__ == '__main__':

    for p in ps:
        p.start()

    for p in ps:
        p.join()
