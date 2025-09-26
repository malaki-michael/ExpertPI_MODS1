import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for Matplotlib

import tkinter as tk
from tkinter import ttk
import pathlib
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import threading
import cv2
import os
import time
from time import sleep

from expert_pi.acceptance_interface import specification_criteria

from expert_pi.app import app
from expert_pi.gui import main_window


window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client

from expert_pi.app import scan_helper
from expert_pi import grpc_client
from expert_pi.measurements import shift_measurements


def get_image(pixel_time,N):
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1)
    header, data = cache_client.get_item(scan_id, N**2)
    return data["stemData"]["BF"].reshape(N, N)

def get_rate(ts, cumulative, di=1):
    di = min(len(cumulative) - 1, di)
    dx = (cumulative[di:, 0] - cumulative[:-di, 0])
    dy = (cumulative[di:, 1] - cumulative[:-di, 1])
    rate = np.sqrt(dx**2 + dy**2)/((ts[di:] - ts[:-di])/60)
    return ts[di:], rate

def calculate_drift_rates(ts, offsets):
    ts = np.array(ts)
    cumulative = np.cumsum(offsets, axis=0)
    total = np.sqrt(cumulative[:, 1] ** 2 + cumulative[:, 0] ** 2)
    tr, rate = get_rate(ts, cumulative, di=1)
    tsr, smooth_rate = get_rate(ts, cumulative, di=40)

    return rate,smooth_rate




def measure_drift_rate():
    """
    Perform drift measurement with a popup progress bar.
    Generate a plot at each iteration and use OpenCV to display the updated plot.
    """
    # Popup window for progress bar
    popup = tk.Toplevel()
    popup.title("Measurement Progress")
    popup.geometry("400x150")
    popup.resizable(False, False)

    # Progress bar and label
    progress_bar = ttk.Progressbar(popup, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=20)
    progress_label = tk.Label(popup, text="Moving stage...")
    progress_label.pack()

    # Prepare data storage
    filepath = specification_criteria.shared_directory+"//drift_measurement_data"
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    file = filepath + "//drift_data.pdat"

    # Initialize file paths for plots
    intermediate_plot_path = filepath + "//current_plot.png"  # File path for the current plot image
    final_plot_path = specification_criteria.shared_directory + "//final_plot.png"  # File path for the final plot

    # Worker function
    def worker():
        """
        Perform drift measurement using the original Beta acceptance logic,
        updating progress, generating intermediate plots, and saving the final plot.
        """
        ts, offsets, rates, smoothed_rates = [], [], [], []

        # Perform pre-drift stage movements
        xy0 = grpc_client.stage.get_x_y()
        for move in [[7, 7], [0.2, 0.2], [0, 0]]:
            grpc_client.stage.set_x_y(
                xy0["x"] + move[0] * 1e-6, xy0["y"] + move[1] * 1e-6,
                fixed_beta=True, fixed_alpha=True
            )
            grpc_client.stage.stop()
            time.sleep(1)

        grpc_client.scanning.set_field_width(0.5 * 1e-6)

        # Start measurement
        start = time.time()
        max_duration = specification_criteria.max_drift_measuring_time  # 20 minutes in seconds
        settling_time = specification_criteria.drift_settling_time
        t = 0
        img = get_image(N=1024, pixel_time=1e-6)  # Anchor image
        ts.append(0)
        offsets.append(np.zeros(2))
        imgs = [img]

        # Measurement loop
        while t < max_duration:
            t = time.time() - start
            img = get_image(N=1024, pixel_time=1e-6)

            # Measure offsets
            offset = shift_measurements.get_offset_of_pictures(
                imgs[-1], img, fov=0.5, method=shift_measurements.Method.PatchesPass2
            )
            offsets.append(offset * 1000)  # Convert to nm
            ts.append(t)
            imgs.append(img)



            # Generate the plot image for the current iteration
            rate,smoothed_rate = generate_plot(ts, offsets, intermediate_plot_path)

            # Show the plot image using OpenCV
            show_plot_with_opencv(intermediate_plot_path)

            # Update progress bar based on elapsed time
            progress_percentage = (t / max_duration) * 100
            popup.after(0, update_progress, progress_bar, progress_label, progress_percentage)
            sleep(1)

        # Generate and save the final plot
        rate,smoothed_rate = generate_plot(ts, offsets, final_plot_path)

        # Save intermediate results
        with open(file, 'wb') as f:
            p.dump((ts,rate,smoothed_rate), f)

        # Delete the intermediate plot after processing
        if os.path.exists(intermediate_plot_path):
            os.remove(intermediate_plot_path)

        # Close the popup when done
        popup.after(0, popup.destroy)

    # Start the measurement in a separate thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()

    with open(file, 'rb') as f:
        output = p.load(f)

    time_points = output[0]
    raw_rates = output[1]
    smoothed_rates = output[2]

    spec_pass = validate_rates(time_points,smoothed_rates, threshold=specification_criteria.drift_spec,start_time=specification_criteria.drift_settling_time)

    results = calculate_rate_values(time_points,raw_rates,start_time=specification_criteria.drift_settling_time)

    average_drift_rate = results[0]
    max_rate_after_settling = results[1]
    min_rate_after_settling = results[2]

    return spec_pass, (average_drift_rate,max_rate_after_settling,min_rate_after_settling)

def calculate_rate_values(time_points, rates, start_time): #TODO something is not working properly here
    """
    Calculate average rate, maximum rate, and minimum rate after the start time.

    Parameters:
    - time_points (list of float): List of time points corresponding to the rates.
    - rates (list of float): List of rates at each time point.
    - start_time (float): The time after which calculations should begin.

    Returns:
    - tuple: (average_rate, max_rate, min_rate) after start_time
    """

    #TODO handle different lengths of time points and smoothed rates or just use the raw rates

    # Filter rates corresponding to time points after start_time
    filtered_rates = [rate for t, rate in zip(time_points, rates) if t >= start_time]
    #extract the rates for the timepoints where the time is above the settling time

    if not filtered_rates:
        raise ValueError("No rates found after the specified start_time.")

    # Compute average, max, and min rates
    average_rate = sum(filtered_rates) / len(filtered_rates)
    max_rate = max(filtered_rates)
    min_rate = min(filtered_rates)

    return (average_rate, max_rate, min_rate)


def validate_rates(time_points, rates, threshold, start_time):
    """
    Validates rates against two conditions:
    1. The minimum rate must be below the user-defined threshold.
    2. After the specified start_time, the rate must not exceed the threshold.

    Parameters:
    - time_points (list of float): List of time points corresponding to the rates.
    - rates (list of float): List of rates at each time point.
    - threshold (float): User-defined threshold value for the rates.
    - start_time (float): Time after which the rates must not exceed the threshold.

    Returns:
    - bool: True if both conditions are met, False otherwise.
    """

    # Condition 1: Minimum rate must be below the threshold
    if min(rates) >= threshold:
        return False

    # Condition 2: After the start_time, no rate should exceed the threshold
    for t, rate in zip(time_points, rates):
        if t >= start_time and rate > threshold:
            return False

    # Both conditions are satisfied
    return True


def update_progress(progress_bar, progress_label, progress_percentage):#, drift_rate):
    """
    Update the progress bar and label in the main thread.
    """

    #print(drift_rate)
    progress_bar["value"] = progress_percentage
    progress_label.config(
        text=f"Progress: {progress_percentage:.1f}% |" #Last Drift Rate: {drift_rate} nm/min"
    )


def generate_plot(ts, offsets, image_path):
    """
    Generate and save a plot for drift rates.
    """
    # Delete the previous plot image if it exists
    if os.path.exists(image_path):
        os.remove(image_path)

    fig, ax = plt.subplots(2, 1)

    ts = np.array(ts)
    cumulative = np.cumsum(offsets, axis=0)
    total = np.sqrt(cumulative[:, 1] ** 2 + cumulative[:, 0] ** 2)
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


    ax[1].plot(tsr, smooth_rate, "-",label="smoothed drift rate")

    ax[1].plot([specification_criteria.drift_settling_time, specification_criteria.drift_settling_time], [0, specification_criteria.drift_spec * 5], 'k--', linewidth=0.5)  # plot settle time objective
    ax[1].plot([0, specification_criteria.max_drift_measuring_time], [specification_criteria.drift_spec, specification_criteria.drift_spec], 'k--', linewidth=0.5)
    ax[1].set_ylim([0, specification_criteria.drift_spec * 5])
    ax[0].legend()

    ax[1].plot(tr, rate, ".",label="Raw drift rate",color="red")
    ax[1].legend()

    plt.savefig(image_path)
    plt.close()

    return rate,smooth_rate

def show_plot_with_opencv(image_path):
    """
    Display the plot image using OpenCV in a thread-safe manner.
    """
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imshow("Drift Measurement Plot", img)
        cv2.waitKey(1)  # Small delay to allow OpenCV to render the image


# Example Usage (standalone script)
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window since we only want the popup

    measure_drift_rate()
    root.mainloop()
    thread.join()
    # Close the OpenCV window after the main loop ends
    cv2.destroyAllWindows()


#TODO after 300 seconds, ensure drift does not go above the specification point
#TODO add in average drift rate after 300s

