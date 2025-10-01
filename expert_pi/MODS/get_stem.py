import numpy as np
from expert_pi import grpc_client
from expert_pi.__main__ import window                    # importing user interface functionalities without opening window
from expert_pi.app import app                            # importing application functions
from expert_pi.grpc_client.modules._common import DetectorType as DT  # importing detetctor functions
from expert_pi.app import scan_helper                    # import a wrapper to help with aqusitions
from expert_pi.MODS import generals                      # import user-defined modules

import cv2 as cv2
import os
import glob
import easygui
import json



"""
creates another GUI 
window = main_window.MainWindow()  
instead use --> from expert_pi.__main__ import window
"""


            
# initialize the controller of the app
controller = app.MainApp(window)
"""
the controller acts as the central maanger that connects GUI window with backend logic and services, 
It often manages application-wide settings, event handling, and communication between different modules.
"""
cache_client = controller.cache_client
# Accesses the memory where STEM rawdata and metadata is stored

          



# Basic STEM Acquisition

def basic_acquire(fov=None, scan_width=None, pixel_time=None, scan_rotation=None):
    """
    Takes and sets fov in µm, scan_width in pixels, pixel_time in µs, scan_rotation in degrees
    The scan_helper wrapper doesn't set fov and scan_rotation, these must be done prior to calling scan_helper
    Acquires a single STEM image and returns bf, adf, and metadata
    """

    if fov == None:
        fov= grpc_client.scanning.get_field_width()
    else:
        fov = grpc_client.scanning.set_field_width(fov*1e-6)            

    if scan_width == None:
        scan_width_px = window.scanning.size_combo.currentText()
        scan_width = int(scan_width_px.replace("px", ""))
    
    if pixel_time == None:
        pixel_time = window.scanning.pixel_time_spin.value() 
    else:
        pixel_time = pixel_time

    if scan_rotation == None:
        pass
    else:
        scan_rotation = grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation))
    
    print("\n\t--Scan Conditions Set--")

    """Enable off-axis BF only when BF is retracted"""
    if grpc_client.stem_detector.get_is_inserted(DT.BF) == False and grpc_client.stem_detector.get_is_inserted(DT.HAADF) == False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True)
    elif grpc_client.stem_detector.get_is_inserted(DT.BF) == False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True)
    else:
        grpc_client.projection.set_is_off_axis_stem_enabled(False)
    
    print("\n\t--Detectors Inserted--")

    # Collect Metadata
    metadata = generals.collect_metadata(acquisition_type="Stem", dwell_time = pixel_time, scan_width = scan_width)
    
    print("\n\t--Metadata Collected--")
    #Acquire scans
    scan_id = scan_helper.start_rectangle_scan(pixel_time*1e-6, total_size = scan_width, frames = 1,
                                               detectors=[DT.BF, DT.HAADF])     # pixel time needs to be seconds
    
    """No matter which detector is inserted since both detectors are given in scan_helper,
      we will get both bf and df data array from cache_client"""
    
    header, data = cache_client.get_item(scan_id, pixel_count = scan_width**2)

    bf_image = data["stemData"]["BF"].reshape(scan_width, scan_width)
    df_image = data["stemData"]["HAADF"].reshape(scan_width, scan_width)

    print("\n\t***Acquisition Complete***")
    return bf_image, df_image, metadata


def save_acquire(bf, df, metadata= None):
    """Takes bf, df, metadata and writes tiff images for bf, df, and json for metadata
       bf = NxN numpy array
       df = NxN numpy array
       metdata = dictionary
       Used after basic_acquire()"""
    
    folder = easygui.diropenbox("Select Save Location")
    folder+= "\\"

    filelist = glob.glob(folder+"\\*tiff")      #gives list of all files with extension tiff
    num_of_files = len(filelist)
    bf_name = f"STEM_BF_000{str(num_of_files)}"
    df_name = f"STEM_DF_000{str(num_of_files)}"
    bf_file = folder+bf_name+".tiff"
    df_file = folder+df_name+".tiff"

    cv2.imwrite(bf_file, bf)
    cv2.imwrite(df_file, df)
    print("\n\t***Saving File***")

    if metadata == None:
        metadata = generals.collect_metadata(acquisition_type= "stem")
        

    with open(folder+bf_name+"_metadata.json", "w") as file:
        json.dump(metadata, file, indent = 4)
        print("\n\t***Saving Metadata***")
    
    print("\n\t***Saving Completed***")


    























