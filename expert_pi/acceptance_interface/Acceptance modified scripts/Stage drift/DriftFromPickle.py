# plot needs to be created in main thread:
import matplotlib.pyplot as plt
import pickle
import json
import time
import numpy as np
import pandas as pd
import datetime

loadingTime = datetime.datetime(2024,8,13,12,13)
timeOfMeasurement = datetime.datetime.now()
elapsedFromLoading = str(timeOfMeasurement - loadingTime).split(".")[0]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 8),gridspec_kw={'height_ratios': [1, 2]})

rate_limit = 5  # limit drift, nm/min
settle_time = 240  # time to 'settled'



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
    ax[1].set_ylim([0, rate_limit*1.5])
    ax[0].legend()

    if returnData:
        return tsr, smooth_rate
    


with open("drift_measurements_Nonclamped2.pdat", 'rb') as f:
    A = pickle.load(f)

ts = A[0]
offsets = A[2]
ts, driftSpeed = plot_data(ts, offsets, True)

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

fig.suptitle(f'Plot updated at: {time.ctime()} \nTime since loading: {elapsedFromLoading} \nprecentiles after settle time: 90%: {Stats[0]["90%"]:{5}.{4}}, 95%: {Stats[0]["95%"]:{5}.{4}}, 99%: {Stats[0]["99%"]:{5}.{4}}')



plt.show()