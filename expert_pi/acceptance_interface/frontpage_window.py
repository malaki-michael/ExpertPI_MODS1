import tkinter as tk
from tkinter import filedialog,Listbox
import time
import random
import threading
import easygui as g
import cv2
import numpy as np
import json
import os
from expert_pi.acceptance_interface import EDX
from expert_pi.acceptance_interface.STEM import HR_resolution_check
from expert_pi.acceptance_interface.camera_speed import check_camera_speed
from expert_pi.acceptance_interface.hw_integration_test import grpc_test
from expert_pi.acceptance_interface.scanning import scan_mag_test
from expert_pi.acceptance_interface.new_drift_measurement import measure_drift_rate

from expert_pi.acceptance_interface import specification_criteria
from expert_pi.acceptance_interface.specification_criteria import shared_directory
from expert_pi.gui.console_threads import threaded




# Simulate different task functions
def task_1(button):
    if specification_criteria.shared_directory:
        button.config(bg="yellow")
        time.sleep(2)  # Simulate a task using the directory
        result = random.choice([True, False])
        button.config(bg="green" if result else "red")
    else:
        button.config(bg="red")
        print("No directory selected for Task 1")


def scan_mag_calibration(button):
    if specification_criteria.shared_directory:
        button.config(bg="yellow")
        image = filedialog.askopenfilename(initialdir=r"%userprofile%\documents")

        spec_pass,result = scan_mag_test(image)

        append_to_json_file(specification_criteria.shared_directory,{"Scan magnification calibration x":result[0],"Scan magnification calibration y":result[1]})

        button.config(bg="green" if spec_pass else "red")
    else:
        button.config(bg="red")
        print("No output directory selected")


def measure_edx_resolution(button):
    if specification_criteria.shared_directory:
        button.config(bg="yellow")
        path_det_0 = filedialog.askopenfilename(initialdir=r"%userprofile%\documents",filetypes=[("Text Files", "*.txt")])
        path_det_1 = filedialog.askopenfilename(initialdir=r"%userprofile%\documents",filetypes=[("Text Files", "*.txt")])

        spec_pass,result = EDX.measure_detector_width(path_det_0,path_det_1)
        button.config(bg="green" if spec_pass else "red")

        append_to_json_file(specification_criteria.shared_directory, {"EDX detector 0 resolution": result[0],
                                                                      "EDX detector 1 resolution": result[1]})

    else:
        button.config(bg="red")
        print("No output directory selected")


def grpc_metadata_test(button):
    button.config(bg="yellow")
    results = grpc_test()
    if results:
        spec_met = True
    else:
        spec_met = False
        print("Cannot connect to hardware")
    button.config(bg="green" if spec_met else "red")


def camera_speed(button):
    button.config(bg="yellow")
    results = check_camera_speed()
    if results[0] and results[1] is True:
        spec_met = True
    else:
        spec_met = False
    button.config(bg="green" if spec_met else "red")

def measure_drift(button):
    button.config(bg="yellow")
    spec_pass,result = measure_drift_rate()
    print(result)

    drift_results = {"Average rate after settling time nm/min":result[0],
                                                                 "Minimum drift rate after settling time nm/min":result[2],
                                                                 "Maximum drift rate after settling time nm/min":result[1]}

    append_to_json_file(specification_criteria.shared_directory,{"Drift Results":drift_results})

    button.config(bg="green" if spec_pass else "red")


def validate_quantification(button):
    button.config(bg="yellow")
    standard_options = list(specification_criteria.quant_dictionary.keys())
    selected_standard = g.choicebox("Select the standard used for quantification validation","Quantification validation",choices=standard_options)

    elements =  list(specification_criteria.quant_dictionary[selected_standard].keys())

    quant_values_list = g.multenterbox("Enter the quantification values for the respective elements in Weight Percent",fields=elements)

    quant_values = dict(zip(elements,quant_values_list))

    result = EDX.check_quant_accuracy(selected_standard,quant_values)

    if result  is True:
        spec_met = True
    else:
        spec_met = False
    button.config(bg="green" if spec_met else "red")


def read_json_file(directory_path):
    """
    Reads and returns the contents of a JSON file named 'acceptance results.json' in the specified directory.
    If the file does not exist, returns False. Raises exceptions for other errors.

    Parameters:
        directory_path (str): Path to the directory containing the JSON file.

    Returns:
        list: The contents of the JSON file as a list.
        False: If the file does not exist.
    """
    file_path = os.path.join(directory_path, "acceptance results.json")

    if not os.path.isfile(file_path):
        return False  # Return False if the file does not exist

    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            # Ensure the JSON data is a list; if not, convert it to a list
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError("Unsupported JSON structure: Must be a list or a dictionary.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e


def append_to_json_file(directory_path, new_data):
    """
    Appends new data to a JSON file named 'acceptance results.json' in the specified directory.
    If the file does not exist, it creates a new file.

    Parameters:
        directory_path (str): Path to the directory containing the JSON file.
        new_data (dict): Dictionary of data to append.
    """
    file_path = os.path.join(directory_path, "acceptance results.json")

    # Read the existing data from the file
    existing_data = read_json_file(directory_path)

    if existing_data is False:  # If no file exists, initialize an empty list
        existing_data = []

    # Append the new data to the existing list
    existing_data.append(new_data)

    # Write the updated list back to the file
    with open(file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)
    print(f"Successfully appended data to {file_path}.")


def put_image_in_directory(button):
    if specification_criteria.shared_directory is not None:
        button.config(bg="yellow")
        shared_directory = specification_criteria.shared_directory
        append_to_json_file(shared_directory,{"Is the directory there":True})
        button.config(bg="green")
    else:
        button.config(bg="red")
        print("No directory selected for Task 1")


# Function to open a directory dialog and store the selected directory path in shared_data
def select_directory(button):
    # Open directory dialog and get the selected directory path
    directory = filedialog.askdirectory(title="Select a Directory")
    if directory:
        specification_criteria.shared_directory = directory  # Update the shared variable
        button.config(bg="green")
        print(f"Directory selected: {specification_criteria.shared_directory}")
        existing_report = read_json_file(directory)
        if existing_report is False:
            file_path = os.path.join(directory, "acceptance results.json")
            with open(file_path, 'w') as json_file:
                json.dump({"acceptance_results":True}, json_file, indent=4)
            print(f"Starting acceptance tracking document {file_path}.")

    else:
        button.config(bg="red")
        print("No directory selected")

def check_HR_resolution(button): #TODO checked ok
    if specification_criteria.shared_directory:

        button.config(bg="yellow")
        image_BF = filedialog.askopenfilename(initialdir=r"%userprofile%\documents")
        spec_pass_BF = HR_resolution_check(image_BF)
        image_ADF = filedialog.askopenfilename(initialdir=r"%userprofile%\documents")
        spec_pass_ADF = HR_resolution_check(image_ADF)

        append_to_json_file(specification_criteria.shared_directory,{"BF resolution nm":spec_pass_BF[1],"ADF resolution nm":spec_pass_ADF[1]})


        spec_pass = False
        if spec_pass_BF[0] and spec_pass_ADF[0] is True:
            spec_pass = True

        button.config(bg="green" if spec_pass else "red")
    else:
        button.config(bg="red")
        print("No output directory selected")

# Handler function for button clicks (runs in a separate thread)
def on_button_click(button, task_function):
    threading.Thread(target=task_function, args=(button,)).start()


# Function to create buttons and bind event handlers to the appropriate function
def create_button_event_handler(button, task_function):
    def event_handler():
        on_button_click(button, task_function)

    return event_handler


# Main GUI setup
def create_button_window(function_dict):
    # Create the main window
    window = tk.Tk()
    window.title("Button Task GUI")

    # Create a button to select a directory
    directory_button = tk.Button(window, text="Select Directory", bg="grey", width=20, height=2)
    directory_button.config(command=lambda: select_directory(directory_button))
    directory_button.grid(row=0, column=0, padx=10, pady=10)

    # Set up the grid layout (2 columns, n/2 rows)
    row, col = 1, 0
    for title, task_function in function_dict.items():
        # Create a button with the title from the function dictionary, initially grey
        button = tk.Button(window, text=title, bg="grey", width=20, height=2)

        # Bind the event handler to the task function
        button.config(command=create_button_event_handler(button, task_function))

        # Place the button in the grid (2 columns, calculate row and column)
        button.grid(row=row, column=col, padx=10, pady=10)
        col = (col + 1) % 2
        if col == 0:
            row += 1

    # Start the tkinter main loop
    window.mainloop()


# Create the button window using the function dictionary

def task_n(button):
    button.config(bg="yellow")
    time.sleep(2)
    print("Task N completed")
    result = random.choice([True,False])
    button.config(bg="green" if result else "red")

function_dict = {
    "Test directory": put_image_in_directory,
    "Test hardware connection": grpc_metadata_test,
    "Measure scan calibration": scan_mag_calibration,
    "Measure EDX resolution":measure_edx_resolution,
    "Measure Stage drift": measure_drift,
    "Check EDX quant":validate_quantification,
    "HR-STEM":check_HR_resolution,

}


threaded(create_button_window(function_dict))