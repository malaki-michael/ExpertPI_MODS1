# plot needs to be created in main thread:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# to update run this:
fig.canvas.draw()
fig.canvas.flush_events()
# for background acquisition copy paste to console:


%%threaded
from stem_measurements import shift_measurements
import pickle
import time
import numpy as np

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.stream_clients import cache_client
from time import sleep
from matplotlib import cm

stage_abs_shift = 200  # um
file = f"C:/temp/repeatibility_measurements_{stage_abs_shift}um.pdat"
settle_time = 2  # s
fov = 4  # um
N = 1024
repeats = 40
pixel_time = 1000e-9  # s
grpc_client.scanning.set_field_width(fov*1e-6)


def plot_data(phis, offsets):
    # fig.show()
    ax.clear()

    ax.set_title(f"repeatibility {stage_abs_shift} um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    maxo = np.max(np.abs(offsets))*1.5

    phis2 = phis
    x_previous = (np.cos(phis2[1:]) - np.cos(phis2[:-1]) + 1)/2

    ax.plot([-maxo, maxo], [0, 0], "-", color="black")
    ax.plot([0, 0], [-maxo, maxo], "-", color="black")
    ax.scatter([o[0] for o in offsets[1:]], [o[1] for o in offsets[1:]], c=x_previous, cmap="brg")


def get_image():
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1)
    header, data = cache_client.get_item(scan_id, N**2)
    return data["stemData"]["BF"].reshape(N, N)


phis = [0]
offsets = [np.zeros(2)]
imgs = []

xy0 = grpc_client.stage.get_x_y()
reference = get_image()
imgs.append(reference)

for i in range(repeats):
    phi = np.random.rand()*np.pi*2
    dxy = stage_abs_shift*np.array([np.cos(phi), np.sin(phi)])
    scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=256, frames=0)
    print(i, "/", repeats, "move", stage_abs_shift, "um", phi/np.pi*180, "deg")
    grpc_client.stage.set_x_y(xy0["x"] + dxy[0]*1e-6, xy0["y"] + dxy[0]*1e-6, fixed_beta=True, fixed_alpha=True)
    grpc_client.stage.set_x_y(xy0["x"], xy0["y"], fixed_beta=True, fixed_alpha=True)
    sleep(settle_time)
    img = get_image()
    offset = shift_measurements.get_offset_of_pictures(imgs[-1], img, fov=fov, method=shift_measurements.Method.PatchesPass2)

    phis.append(phi)
    imgs.append(img)
    offsets.append(offset)

    with open(file, 'wb') as f:
        pickle.dump((phis, imgs, offsets), f)

    plot_data(phis, offsets)


