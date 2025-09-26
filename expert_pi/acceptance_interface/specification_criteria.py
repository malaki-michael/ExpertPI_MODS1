
shared_directory = None
#shared_directory = r"C:\Users\robert.hooley\Documents"
#TODO look into MQTT client to read vacuum values?

"""Input the specification criteria to be compared against"""
"""DO NOT CHANGE ANY VALUES WITHOUT CONSULTING TENSOR PM"""

edx_detector_resolution = 140 #in ev

drift_spec = 5 #nm/min
drift_settling_time = 300
max_drift_measuring_time = 600


stem_resolution = 0.28 #Angstroms
scan_calibration_error = 0.03

permitted_quantification_deviance = 5 #in percent
permitted_quantification_deviance_oxygen = 7 #in percent

quant_dictionary = {
"K453":{"Ge":28.4,"Pb":54.2,"O":16.7},
"Olivine":{"Fe":7.75,"Si":17.97,"Mg":31.1,"O":43.2},
"Orthoclase":{"Na":2.4,"Al":11.4,"K":11.7,"Si":32.1,"O":42.3},
"Zinc Sulphide":{"Z":62.5,"S":37.5}}