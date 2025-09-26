import pickle
import numpy as np
from stem_measurements.orientation import point_group_mesh_generator as mesher
from expert_pi.widgets.tools import point_group_visualizer
import matplotlib.pyplot as plt
from PIL import Image

n_threads = 8
output_folder = "C:/data/2022_11_29_NiCo_P150/series3_processed/"

orientations = {}
for i in range(n_threads):
    with open(output_folder + f"{i:04}.pdat", "rb") as f:
        orientations.update(pickle.load(f))

indices = []
points = []
for i, v in orientations.items():
    indices.append(i)
    points.append(v[:, 2])
points = np.array(points)
indices = np.array(indices)

d = point_group_visualizer.PointGroupVisualizer()
d.setGeometry(100, 100, 400, 400)
d.show()

group_name = "6/mmm"
d.select_group(group_name)

vectors = mesher.orientation_ranges[group_name]

mask = np.sum(np.abs(points), axis=1) > 0

reduced_points = point_group_visualizer.reduce_points_to_group(points[mask, :], d.point_group)
colors = mesher.map_points_to_colors(reduced_points, [mesher.normalize(v) for v in vectors])

image = np.zeros((128*512, 3))
image[indices[mask], :] = colors

im = Image.fromarray(image.reshape(128, 512, 3).astype("uint8"))
im.save(output_folder + "orientation_map2.png")

pix_map = d.glview.grab()
pix_map.save(output_folder + "legend.png")
