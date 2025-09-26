import csv
import math
from bisect import bisect_left
import matplotlib.pyplot as plt
import xraydb
from tkinter import *
from expert_pi.acceptance_interface import specification_criteria
from expert_pi.acceptance_interface.specification_criteria import edx_detector_resolution, shared_directory
import numpy as np
from scipy.optimize import curve_fit


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

# Define the Gaussian function for fitting
def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def fit_gaussian_peak(x_data, y_data, num_points=500):
    """
    Fit a Gaussian curve to the given data, return key parameters and the Gaussian peak for plotting.

    Parameters:
    - x_data: array-like, the x positions of the data.
    - y_data: array-like, the y intensities of the data.
    - num_points: int, the number of points for generating the fitted Gaussian curve.

    Returns:
    - result: dict with 'FWHM', 'Peak Center', 'Half Maximum Intensity'.
    - x_fit: array, x values for the fitted Gaussian peak plot.
    - y_fit: array, y values for the fitted Gaussian peak plot.
    """
    # Initial guess for the parameters [amplitude, mean, std deviation]
    initial_guess = [max(y_data), x_data[np.argmax(y_data)], np.std(x_data) / 2]

    # Fit the Gaussian curve to the data
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    a, mu, sigma = popt  # Extract fitting parameters

    # Calculate FWHM from sigma
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    # Calculate the half maximum intensity
    half_max_intensity = a / 2

    # Generate x values for the Gaussian curve (for a smooth plot)
    x_fit = np.linspace(min(x_data), max(x_data), num_points)
    y_fit = gaussian(x_fit, a, mu, sigma)

    # Return the result dictionary and the x, y values for the fitted Gaussian peak
    result = {
        "FWHM": fwhm,
        "Peak Center": mu,
        "Half Maximum Intensity": half_max_intensity
    }
    return result, x_fit, y_fit


# This function will now return the fit results and (x, y) values for plotting the Gaussian peak.


def measure_FWHM_Mn(read_spectrum,det_num):
    shared_directory = specification_criteria.shared_directory

    energy_list = []
    data_list = []
    for i in read_spectrum:
        energy = int(float(i[0]))
        data = int(float(i[1]))
        energy_list.append(energy)
        data_list.append(data)

    measurement_peak_energy = 8040

    lower_bound=measurement_peak_energy-500
    upper_bound = measurement_peak_energy+500

    scale = energy_list[1]-energy_list[0]
    offset = energy_list[0]-(scale/2)

    energy_index_lower = energy_list.index(take_closest(energy_list,lower_bound))
    energy_index_upper = energy_list.index(take_closest(energy_list,upper_bound))

    clipped_energy = energy_list[energy_index_lower:energy_index_upper]
    clipped_data = data_list[energy_index_lower:energy_index_upper]

    results, xfit,yfit = fit_gaussian_peak(clipped_energy,clipped_data)

    FWHM = results["FWHM"]
    x0 = results["Peak Center"]
    HM = results["Half Maximum Intensity"]


    fig, ax = plt.subplots()

    # Plot the FWHM bounding lines
    ax.axhline(HM)  # Half maximum indicator
    ax.axvline(x0 - (FWHM / 2), 0, 0.5, linestyle="dotted", color="black", label="FWHM")  # lower FWHM bound
    ax.axvline(x0 + (FWHM / 2), 0, 0.5, linestyle="dotted", color="black")  # upper FWHM bound
    ax.axvline(x0, label="Peak center")

    # Scatter plot for real data
    ax.scatter(clipped_energy, clipped_data, label="Experimental")
    ax.plot(xfit,yfit,color="black",linestyle="-", linewidth=2, label="Gaussian Fit")

    # Gaussian peak plot
    """    ax.plot(clipped_energy, gauss(clipped_energy, *gauss_fit(clipped_energy, clipped_data)), '--r',
            label='Gaussian fit')"""
    #TODO put in catch to enable or disable rescaling
    rescaled_FWHM = rescale_det_res(peak_width_measured=FWHM,peak_energy_measured=x0)

    if rescaled_FWHM <= specification_criteria.edx_detector_resolution:
        result = "meets"
    else:
        result = "does not meet"

    ax.set_title(f"Energy resolution of detector {det_num} is {np.round(rescaled_FWHM,1)} eV" "\n" f"which {result} the specification of {specification_criteria.edx_detector_resolution} eV")
    # Add legend
    ax.legend()

    # Set axis labels
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (Counts)')
    # Save figure
    fig.savefig(shared_directory + "/EDX detector" + str(det_num) + ".jpg", format="jpg", transparent=False)

    return (FWHM,rescaled_FWHM)


def convert_peak_width(peak_width, peak_energy, energy_mn):
    # Apply the formula to convert peak width from Copper Kα to Manganese Kα
    peak_width_mn = peak_width * math.sqrt(energy_mn / peak_energy)
    return peak_width_mn


def rescale_det_res(peak_width_measured,peak_energy_measured):
    # Energies of the Kα peaks (in keV)
    #peak_energy_measured*1e-3  # keV for Copper Kα
    energy_mn_k_alpha = 5.90  # keV for Manganese Kα

    peak_width_mn = convert_peak_width(peak_width_measured, peak_energy_measured*1e-3, energy_mn_k_alpha)

    #print(f"Peak width for Copper Kα: {peak_width_cu:.1f} eV")
    #print(f"Converted peak width for Manganese Kα: {peak_width_mn:.1f} eV")

    return peak_width_mn

def measure_detector_width(path_det_0,path_det_1):

    with open(path_det_0, newline='') as f:
        reader = csv.reader(f)
        spectrum_0 = [tuple(row) for row in reader]
        del spectrum_0[0]

    with open(path_det_1, newline='') as f:
        reader = csv.reader(f)
        spectrum_1 = [tuple(row) for row in reader]
        del spectrum_1[0]



    FWHM_cu_0 = measure_FWHM_Mn(spectrum_0,0)
    FWHM_cu_1 = measure_FWHM_Mn(spectrum_1,1)


    if FWHM_cu_0[1] and FWHM_cu_1[1] <= specification_criteria.edx_detector_resolution:
        spec_met = True
    else:
        spec_met = False

    return spec_met,(FWHM_cu_0[1],FWHM_cu_1[1])





def check_quant_accuracy(selected_standard,quant_values_dictionary):

    calibrated_values = specification_criteria.quant_dictionary[selected_standard]

    #TODO check quant values add up to 100 before starting

    all_element_passes = []

    for item in calibrated_values.items():
        element = item[0]
        quant = item[1]
        quant_value = quant_values_dictionary[element]
        difference = abs(int(quant)-int(quant_value))

        if element != "O":
            if  difference < specification_criteria.permitted_quantification_deviance:
                print(f"For {element}, the difference is {difference} %, which is within bounds")
                all_element_passes.append(True)
            else:
                print (f"For {element}, the difference is {difference} %, which is too high")
                all_element_passes.append(False)
        else:
            print("Oxygen quantification check")
            if difference < specification_criteria.permitted_quantification_deviance_oxygen:
                print(f"For {element}, the difference is {difference} %, which is within bounds")
                all_element_passes.append(True)
            else:
                print(f"For {element}, the difference is {difference} %, which is too high")
                all_element_passes.append(False)

    specification_pass = None

    for i in all_element_passes:
        if i is False:
            specification_pass = False
        else:
            pass

    if specification_pass is None: #checks for change to original state
        specification_pass = True #if no failures, test is sucessful

    return specification_pass


#result = check_quant_accuracy("Baryte",{"element1":70,"O":14,"element3":17})
#print(result)