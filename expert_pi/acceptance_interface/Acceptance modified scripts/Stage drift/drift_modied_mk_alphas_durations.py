# plot needs to be created in main thread:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 8))

# for background acquisition copy paste to console:
from stem_measurements import shift_measurements
import pickle
import time
import numpy as np


# to update run this:
fig.canvas.draw()
fig.canvas.flush_events()
fig.suptitle(f'Plot updated at: {time.ctime()}')

from expert_pi.console_threads import threaded
# %%threaded

# from expert_pi import grpc_client, scan_helper
from expert_pi.controllers import scan_helper  ################test this line for new ExpertPi version
from expert_pi.stream_clients import cache_client


def get_rate(ts, cumulative, di=1):
    di = min(len(cumulative) - 1, di)
    dx = (cumulative[di:, 0] - cumulative[:-di, 0])
    dy = (cumulative[di:, 1] - cumulative[:-di, 1])
    rate = np.sqrt(dx**2 + dy**2)/((ts[di:] - ts[:-di])/1)
    return ts[di:], rate


def plot_data(ts, offsets):
    ts = np.array(ts)
    cumulative = np.cumsum(offsets, axis=0)
    total = np.sqrt(cumulative[:, 1]**2 + cumulative[:, 0]**2)
    tr, rate = get_rate(ts, cumulative, di=1)
    tsr, smooth_rate = get_rate(ts, cumulative, di=1)

    # fig.show()
    ax[0].clear()
    ax[1].clear()

    ax[0].set_xlabel("t(s)")
    ax[0].set_ylabel("offset (nm)")
    ax[1].set_ylabel("1-min avaraged drift rate (nm/min)")

    ax[0].plot(ts, cumulative[:, 0], '--', color='red', label="x")
    ax[0].plot(ts, cumulative[:, 1], '--', color='blue', label="y")
    ax[0].plot(ts, total, color='black', label="total")

    # ax[1].plot(tr,rate,":")
    ax[1].plot(tsr, smooth_rate, "-")

    # ax[1].plot([settle_time, settle_time], [0, rate_limit*5], 'k--', linewidth=0.5)  # plot settle time objective
    ax[1].plot([0, max(tr)], [rate_limit, rate_limit], 'k--', linewidth=0.5)
    ax[1].plot([0, max(tr)], [1,1], 'k--', linewidth=0.5)
    ax[1].set_ylim([0, rate_limit*5])
    # ax[1].set_ylim([0, rate_limit*15])
    ax[0].legend()

    # fig.canvas.draw()
    # fig.canvas.flush_events()


file = "C:/temp/drift_measurements.pdat"
# moves = [[7,7],[0.2,0.2],[0, 0]]  # pre-drift x,y movement, um
# moves = [[0, 0]]  # pre-drift x,y movement, um
# alphas = [0,-2,-4,-2,0,2,4,2,0]
alphas = [0,-2,0]

rate_limit = 3  # limit drift, nm/min
cycle_duration = 50  # Duration for each cycle in seconds
# nTimes = 10

fov = 0.5  # um
N = 512
pixel_time = 1000e-9  # s


def get_image():
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1)
    header, data = cache_client.get_item(scan_id, N**2)
    return data["stemData"]["BF"].reshape(N, N)

ts = []
offsets = []
imgs = []

# start collecting drift data
start = time.time()

# Record the start time of the entire process
process_start_time = time.time()

for idx, alpha in enumerate(alphas):
    # Calculate the target start time for this cycle
    target_start_time = process_start_time + idx * cycle_duration

    # Perform the operations for the current alpha value
    print(f'Cycle {idx + 1}/{len(alphas)} - alpha is: {alpha}')
    xy0 = grpc_client.stage.get_x_y()
    print(grpc_client.stage.get_x_y())

    # Start higher faster scan to see the transitions:
    scan_id = scan_helper.start_rectangle_scan(pixel_time=300e-9, total_size=256, frames=1)

    # Alpha movement
    grpc_client.stage.set_alpha(alpha * np.pi / 180)
    grpc_client.stage.stop()
    time.sleep(1)

    grpc_client.scanning.set_field_width(fov * 1e-6)

    t = time.time() - start
    img = get_image()

    ts.append(t)
    imgs.append(img)
    offsets.append(np.zeros(2))

    img = get_image()
    offset = shift_measurements.get_offset_of_pictures(imgs[-1], img, fov=fov, method=shift_measurements.Method.PatchesPass2)

    ts.append(t)
    imgs.append(img)
    offsets.append(offset * 1000)  # to nm

    plot_data(ts, offsets)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f'Plot plotted for alpha {alpha}')

    # Calculate how much time has passed since the start of this cycle
    elapsed_time = time.time() - target_start_time

    # If there's time left before the next cycle should start, sleep for the remaining time
    if elapsed_time < cycle_duration:
        time.sleep(cycle_duration - elapsed_time)
    
    print('plot plotted')
    print(time.ctime())
    # fig.suptitle(f'Plot updated at: {time.ctime()}')
    # fig.suptitle(f'Plot updated at: {time.ctime()}\n angles: {alphas}, time: {cycle_duration}')
    fig.suptitle(f'Plot updated at: {time.ctime()}\n angles: {alphas}, time: {cycle_duration}\n P1, free bellows')
    
# actualTime = time.strftime("%m%d_%H%m")
# resultsLoc = "T:/tem/MIKROSKOP GEN1/TESTY/microscope-sessions-2024/F3/08-07-DriftTesting/ST-Clamped/secondMEas/"
# fig.savefig(resultsLoc + 'drifts' + str(actualTime) + '.png', dpi=150)   # save the figure to file

# with open(file, 'rb') as f:
#     (ts, imgs, offsets) = pickle.load(f)
