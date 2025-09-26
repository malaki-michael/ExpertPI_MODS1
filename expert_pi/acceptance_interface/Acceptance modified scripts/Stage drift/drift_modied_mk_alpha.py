# plot needs to be created in main thread:
import matplotlib.pyplot as plt
from stem_measurements import shift_measurements
import pickle
import time
import numpy as np


# for background acquisition copy paste to console:
# file = "C:/temp/drift_measurements_Nonclamped.pdat"
# location = "T:/tem/MIKROSKOP GEN1/TESTY/microscope-sessions-2024/F3/"
# file = location + time.strftime("%y-%m-%d_%H-%m") +".png"
from expert_pi.console_threads import threaded



for i in range(1):
    print(i)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 8))

    # to update run this:
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.suptitle(f'Plot updated at: {time.ctime()}')

    # %%threaded

    from expert_pi import grpc_client, scan_helper
    #from expert_pi.controllers import scan_helper  ################test this line for new ExpertPi version
    from expert_pi.stream_clients import cache_client


    def get_rate(ts, cumulative, di=1):
        di = min(len(cumulative) - 1, di)
        dx = (cumulative[di:, 0] - cumulative[:-di, 0])
        dy = (cumulative[di:, 1] - cumulative[:-di, 1])
        rate = np.sqrt(dx**2 + dy**2)/((ts[di:] - ts[:-di])/60)
        return ts[di:], rate


    def plot_data(ts, offsets):
        ts = np.array(ts)
        cumulative = np.cumsum(offsets, axis=0)
        total = np.sqrt(cumulative[:, 1]**2 + cumulative[:, 0]**2)
        tr, rate = get_rate(ts, cumulative, di=1)
        tsr, smooth_rate = get_rate(ts, cumulative, di=40)

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
        # ax[1].set_ylim([0, rate_limit*5])
        ax[0].legend()

        # fig.canvas.draw()
        # fig.canvas.flush_events()

    fig.suptitle(f'Plot updated at: {time.ctime()}')
    moves = [[0.2],[0]]  # pre-drift alpha movement, rads

    rate_limit = 3  # limit drift, nm/min
    settle_time = 180  # time to 'settled'
    max_duration = 120  # 7200 is 2 hours
    timeBetweenScans = 0 #prolonged time between scans for long-time measurements
    fov = 0.15  # um
    N = 128
    pixel_time = 1000e-9  # s


    def get_image():
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1)
        header, data = cache_client.get_item(scan_id, N**2)
        return data["stemData"]["BF"].reshape(N, N)


    ts = []
    offsets = []
    imgs = []

    alpha = grpc_client.stage.get_alpha()
    print(grpc_client.stage.get_alpha())

    # start higher faster scan to see the transitions:
    grpc_client.scanning.set_field_width(40e-6)
    scan_id = scan_helper.start_rectangle_scan(pixel_time=300e-9, total_size=256, frames=1)

    # stage pre-movement
    for move in moves:
        print("moving ", move, "rads")
        grpc_client.stage.set_alpha(alpha + move[0],fixed_beta=True)

    print(grpc_client.stage.get_alpha())

    grpc_client.stage.stop()
    time.sleep(1)

    # start collecting drift data
    start = time.time()
    grpc_client.scanning.set_field_width(fov*1e-6)

    t = time.time() - start
    img = get_image()

    ts.append(t)
    imgs.append(img)
    offsets.append(np.zeros(2))

    while t < max_duration:
        t = time.time() - start
        img = get_image()
        offset = shift_measurements.get_offset_of_pictures(imgs[-1], img, fov=fov, method=shift_measurements.Method.PatchesPass2)

        ts.append(t)
        imgs.append(img)
        offsets.append(offset*1000)  # to nm

        # with open(file, 'wb') as f:
        #     pickle.dump((ts, imgs, offsets), f)

        plot_data(ts, offsets)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(timeBetweenScans)        



    print('plot plotted')
    print(time.ctime())
    # print("Waiting for another time")
    # time.sleep(3600 - max_duration)
