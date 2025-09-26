import numpy as np
import scipy
from datetime import datetime
from expert_pi import grpc_client
from expert_pi.gui import main_window
from expert_pi.app import app
from expert_pi.__main__ import window
from expert_pi.grpc_client.modules._common import DetectorType as DT
import easygui
import json
from skimage.measure import block_reduce



#controller = app.MainApp(window)
#cache_client = controller.cache_client

max_detector_angles = grpc_client.projection.get_max_detector_angles()

def collect_metadata(acquisition_type = None, fps = None, dwell_time = None, scan_width = None):
    """Makes various GRPC calls and user-defined functions to collect data of acquisition parameters"""

 
    
    metadata ={
        "Acquisition Date and Time": datetime.now().strftime("%d_%m_%Y %H_%M"),
        "Optical Mode": grpc_client.microscope.get_optical_mode().name,
        "High Tension (kV)": grpc_client.gun.get_high_voltage()*1e-3,
        "Wavelength (pm)": calculate_wavelength(),
        "Probe Current (nA)": np.round(grpc_client.illumination.get_current()*1e9,2),
        "Convergence Semiangle (mrad)": np.round(grpc_client.illumination.get_convergence_half_angle()*1e3,2),
        "Beam Diameter (nm)": np.round(grpc_client.illumination.get_beam_diameter()*1e9,2),
        "FOV (micron)": grpc_client.scanning.get_field_width()*1e6,
        "Scan Width (px)": calculate_scanwidth(scan_width),
        "pixel size (nm)": grpc_client.scanning.get_field_width()*1e9/calculate_scanwidth(),
        "Dwell-time (microsec)" : get_dwelltime(acquisition_type, fps, dwell_time),
        "Scan Rotation (deg)": np.rad2deg(grpc_client.scanning.get_rotation()),  
        "Acquisition Type": acquisition_type,
        "Full Camera Angle (mrad)": 2*1e3*grpc_client.projection.get_max_camera_angle(),
        "BF collection Semi-angle (mrad)": get_BFdetectorangles(acquisition_type),
        "DF Collection Semi-angle innner, outer (mrad)": get_DFdetectorangles(acquisition_type),
        "Alpha Tilt (deg)": np.rad2deg(grpc_client.stage.get_alpha()),
        "Beta Tilt (deg)": np.rad2deg(grpc_client.stage.get_beta()),
        "X (micron)": np.round(grpc_client.stage.get_x_y()["x"]*1e6,2),
        "Y (micron)": np.round(grpc_client.stage.get_x_y()["y"]*1e6,2),
        "Z (micron)": np.round(grpc_client.stage.get_z()*1e6, 2),

                }
    return metadata
    
    
    
def get_BFdetectorangles(acquisition_type = None):
    """
    Checks type of acquisition and returns BF detector angles
    """
    if (acquisition_type.lower() == "stem") & (grpc_client.stem_detector.get_is_inserted(DT.BF)):
        if grpc_client.stem_detector.get_is_inserted(DT.HAADF):
            return max_detector_angles["haadf"]["start"]*1e3
        else:
            return max_detector_angles["bf"]["end"]*1e3
    elif (acquisition_type.lower() == "stem") & (grpc_client.projection.get_is_off_axis_stem_enabled() == True):
        if grpc_client.stem_detector.get_is_inserted(DT.HAADF):
            return max_detector_angles["haadf"]["start"]*1e3
        else:
            return max_detector_angles["bf"]["end"]*1e3
    else:
        return "Detector Not Used in Measurement"

def get_DFdetectorangles(acquisition_type = None):
    """
    Checks type of acquisition and returns DF detector angles
    """
    if (acquisition_type.lower() == "stem") & (grpc_client.stem_detector.get_is_inserted(DT.HAADF)):
        return f"{max_detector_angles["haadf"]["start"]*1e3}, {max_detector_angles["haadf"]["end"]*1e3}"
    else:
        return "Detector Not Used in Measurement"



def get_dwelltime(acquisition_type = None, fps = None, dwell_time = None):
    """
    Checks type of acquisition and camera FPS returns dwell-time in µs
    """
    if acquisition_type.lower() == "stem":
        if dwell_time == None:
            dwell_time = window.scanning.pixel_time_spin.value()
        else:
            dwell_time = dwell_time 
    else:
        dwell_time = (1/fps)*1e6      # reciprocal of the camera fps
    
    return dwell_time
   
    

def calculate_wavelength(energy = None):
    """
    Calculates relativistic wavelength using high-tension voltage and returns it in pm.
     The volatge can be from GRPC call or user specified.
     Eg: calculate_wavelength 
        calculate_wavelength(200000)
    """
    if energy == None:
        energy = grpc_client.gun.get_high_voltage()   # in volts
    energy_rel = energy*(1+ ((scipy.constants.e*energy)/(2*scipy.constants.m_e*scipy.constants.c**2)))
    denominator = np.sqrt(2*scipy.constants.m_e*scipy.constants.e*energy_rel)
    wavelength = scipy.constants.h/denominator
    return wavelength*1e12

def calculate_scanwidth(scan_width = None):
    """
    Calculates and return scan width in pixels by reading it from the GUI
    """
    if scan_width == None:
        scan_width_px = window.scanning.size_combo.currentText()  # gets the curent text in number of pixels box of SCANNING Toolbar 
        scan_width = int(scan_width_px.replace("px", ""))
    else:
        scan_width = scan_width
    return scan_width


def set_scope_conditions():
    """Loads saved metadata and sets the microscope conditions"""
    
    metadata_file = easygui.fileopenbox("Select Metadata File", "SELECT FILE")
    
    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    probe_current_A = 1e-9*metadata["Probe Current (nA)"]
    convergence_angle_rad = 1e-3*metadata["Convergence Semiangle (mrad)"]
    fov_m = 1e-6*metadata["FOV (micron)"]
    scan_width = metadata["Scan Width (px)"]
    dwell_time = metadata["Dwell-time (microsec)"]
    scan_rotation_rad = np.deg2rad(metadata["Scan Rotation (deg)"])
    #acquisition_type = metadata["Acquisition Type"]
    half_camera_angle_rad = 1e-3*0.5*metadata["Full Camera Angle (mrad)"] 
    #bf_collection_angle = metadata["BF collection Semi-angle (mrad)"]
    #df_collection_angles_in = 1e-3*metadata["DF Collection Semi-angle innner, outer (mrad)"][0]
    #df_collection_angles_out = 1e-3*metadata["DF Collection Semi-angle innner, outer (mrad)"][1]

    grpc_client.illumination.set_illumination_values(current= probe_current_A, 
                                                     angle = convergence_angle_rad, keep_adjustments=True)
    grpc_client.scanning.set_field_width(fov_m)
    
    print(f"\nScan-width (px):{scan_width}, Please input directly in basic_acquire function or GUI")
    print(f"\nDwelltime(microsec):{dwell_time}, Please input directly in basic_acquire function or GUI")
    
    
    grpc_client.scanning.set_rotation(scan_rotation_rad)

    grpc_client.projection.set_max_camera_angle(DT.Camera, half_camera_angle_rad, keep_adjustments= True)
    print(f"\nChange camera angle to {2*1e3*half_camera_angle_rad} on GUI to set BF, HAADF detetector angles")
    #Check if the detetcors and inserted or not, if not remove

    if metadata["BF collection Semi-angle (mrad)"] == "Detector Not Used in Measurement":
        grpc_client.stem_detector.set_is_inserted(DT.BF, False)
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
    
    if metadata["DF Collection Semi-angle innner, outer (mrad)"] == "Detector Not Used in Measurement":
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False)
    
    print("\nAll Conditions set except for scan width, dwell-time, and detector angles")

    #TODO Ask Rob how to set the BF, and DF detetcor angles- cannot find grpc calls for this
    #TODO also ask if stage x,y,z should be moved

    

def calc_dose(curr, fov, scan_width, dwell_time):
    """Take current in nA, FOV in micron, scan_width in pixels, dwell-time in micro-seconds
    and return pizel size (Å/px) and dose caluclated in e/Å^2"""
    curr = curr*10*(10**(-10))
    fov = fov*(10**4)
    dwell_time = dwell_time*(10**(-6))
    e = 1.6*(10**(-19))
    pixel_size = fov/scan_width


    dose = (curr*dwell_time)/(e*(pixel_size**2))
    return f"Pixel Size = {pixel_size} Å/px, Dose = {dose} e/Å^2" 




