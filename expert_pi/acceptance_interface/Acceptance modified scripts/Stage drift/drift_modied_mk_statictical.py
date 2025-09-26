# plot needs to be created in main thread:
import matplotlib.pyplot as plt
from stem_measurements import shift_measurements
import pickle
import json
import time, datetime
import numpy as np
import pandas as pd
import os

# for background acquisition copy paste to console:
# file = "C:/temp/drift_measurements_Nonclamped.pdat"
# location = "T:/tem/MIKROSKOP GEN1/TESTY/microscope-sessions-2024/F3/"
# file = location + time.strftime("%y-%m-%d_%H-%m") +".png"
from expert_pi.console_threads import threaded

#################################################################### MEASUREMENT SETTINGS !!!
time_of_measurements = 600#s
times_between_measurements = 600#s
numberOfMeasurements = 2 #how many times to measure

loadingTime = datetime.datetime(2024,8,13,16,23) #YEAR, MONTH, DAY, HOUR, MINUTE
rate_limit = 5  # limit drift, nm/min
settle_time = 240  # time to 'settled'
saveLocally = False


filepath = "T:/tem/MIKROSKOP GEN1/TESTY/microscope-sessions-2024/" #this should remain the same
currentFolderName = "F3/08-14-DriftTestings/" #Scope name, month-day-Project
holderName = "DT07"

if saveLocally:
    filepath = "D:/stageSpeed/"

##Check if folder exsist or create
path = filepath + currentFolderName
if not os.path.exists(path):
    print("Foder does not exists, creating one")
    os.makedirs(path)

for i in range(numberOfMeasurements): #
    print(i)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 8),gridspec_kw={'height_ratios': [1, 2]})

    # to update run this:
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.suptitle(f'Plot updated at: {time.ctime()}')

    # %%threaded

    from expert_pi import grpc_client, scan_helper
    # from expert_pi.controllers import scan_helper  ################test this line for new ExpertPi version
    from expert_pi.stream_clients import cache_client


    def get_rate(ts, cumulative, di=1):
        di = min(len(cumulative) - 1, di)
        dx = (cumulative[di:, 0] - cumulative[:-di, 0])
        dy = (cumulative[di:, 1] - cumulative[:-di, 1])
        rate = np.sqrt(dx**2 + dy**2)/((ts[di:] - ts[:-di])/60)
        return ts[di:], rate


    def plot_data(ts, offsets, returnData = False):
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

        ax[1].plot([settle_time, settle_time], [0, rate_limit*5], 'k--', linewidth=0.5)  # plot settle time objective
        ax[1].plot([0, max(tr)], [rate_limit, rate_limit], 'k--', linewidth=0.5)
        ax[1].set_ylim([0, rate_limit*5])
        ax[0].legend()

        if returnData:
            return tsr, smooth_rate
        # fig.canvas.draw()
        # fig.canvas.flush_events()

    fig.suptitle(f'Plot updated at: {time.ctime()}')
    moves = [[7,7],[0.2,0.2],[0, 0]]  # pre-drift x,y movement, um

    max_duration = time_of_measurements  # 7200 is 2 hours
    timeBetweenScans = 0 #prolonged time between scans for long-time measurements
    fov = 0.5  # um
    N = 1024
    pixel_time = 1000e-9  # s


    def get_image():
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1)
        header, data = cache_client.get_item(scan_id, N**2)
        return data["stemData"]["BF"].reshape(N, N)


    ts = []
    offsets = []
    imgs = []

    xy0 = grpc_client.stage.get_x_y()
    print(grpc_client.stage.get_x_y())

    # start higher faster scan to see the transitions:
    grpc_client.scanning.set_field_width(40e-6)
    scan_id = scan_helper.start_rectangle_scan(pixel_time=300e-9, total_size=256, frames=1)

    # stage pre-movement
    for move in moves:
        print("moving ", move, "um...")
        grpc_client.stage.set_x_y(xy0["x"] + move[0]*1e-6, xy0["y"] + move[1]*1e-6, fixed_beta=True, fixed_alpha=True)

    print(grpc_client.stage.get_x_y())

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
    
    ts, driftSpeed = plot_data(ts, offsets, True)
    data = {
            'time':list(ts),
            'speeds':list(driftSpeed)
    }

    timeOfMeasurement = datetime.datetime.now()
    elapsedFromLoading = str(timeOfMeasurement - loadingTime).split(".")[0]
    filename=filepath + currentFolderName + holderName + '_drift_' + str(elapsedFromLoading).replace(":","-") #file name of data to be saved

    with open(filename+".json",'w') as f:
        json.dump(data, f, indent=4)

    ### Percentile evaluation
    try:
        firstCrossIndex = np.where(driftSpeed<rate_limit)[0][0]
        timeThresholdCrossed = f'{ts[firstCrossIndex]:{1}.{5}} s' #time that driftspeed first crosses settle limit
        ax[1].scatter([ts[firstCrossIndex]],[driftSpeed[firstCrossIndex]], c="r")

    except:
        timeThresholdCrossed = "Not reached"

    TimeThreshold = np.where(ts>=settle_time)[0][0] #index of value that first crosses settle time
    Tpd = pd.DataFrame(ts[TimeThreshold:])
    Dpd = pd.DataFrame(driftSpeed[TimeThreshold:])
    Stats = Dpd.describe(percentiles=[.9,.95,.99])



    ax[1].plot([Tpd[0].iloc[0], Tpd[0].iloc[-1]], [Stats[0]["99%"], Stats[0]["99%"]], 'r--', linewidth=0.5)
    ax[1].plot([Tpd[0].iloc[0], Tpd[0].iloc[-1]], [Stats[0]["95%"], Stats[0]["95%"]], 'b--', linewidth=0.5)
    ax[1].plot([Tpd[0].iloc[0], Tpd[0].iloc[-1]], [Stats[0]["90%"], Stats[0]["90%"]], 'g--', linewidth=0.5)
    ax[1].plot([Tpd[0].iloc[0], Tpd[0].iloc[-1]], [Stats[0]["mean"], Stats[0]["mean"]], 'y--', linewidth=0.5)



    ax[1].annotate("Percentile 99%", xy = (Tpd[0].iloc[-1], Stats[0]["99%"]))
    ax[1].annotate("Percentile 95%", xy = (Tpd[0].iloc[-1], Stats[0]["95%"]))
    ax[1].annotate("Percentile 90%", xy = (Tpd[0].iloc[-1], Stats[0]["90%"]))
    ax[1].annotate("Mean", xy = (Tpd[0].iloc[-1], Stats[0]["mean"]))

    fig.suptitle(f'Plot updated at: {time.ctime()} \nTime from loading holder: {elapsedFromLoading}, Time of limit crossed: {timeThresholdCrossed}, \nPerc after settle: 90%: {Stats[0]["90%"]:{5}.{4}}, 95%: {Stats[0]["95%"]:{5}.{4}}, 99%: {Stats[0]["99%"]:{5}.{4}}, Mean:  {Stats[0]["mean"]:{5}.{4}}')
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.savefig(filename+".png", dpi=200)

    print('plot plotted')
    print(f"Waiting for another time "+ str(datetime.datetime.now()+datetime.timedelta(seconds = times_between_measurements)).split(".")[0])
    time.sleep(times_between_measurements)
    
