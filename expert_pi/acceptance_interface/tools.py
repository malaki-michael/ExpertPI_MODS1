import numpy as np
from time import sleep
from tqdm import tqdm
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT,RoiMode as RM, MicroscopeState as MS
from expert_pi.app import app
from expert_pi.gui import main_window
import datetime
from datetime import timedelta

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client


def scan_4D(scan_width_px=32,camera_frequency_hz=4500):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 4500"""
    if scan_width_px < 32:
        if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True: #if either STEM detector is inserted
            grpc_client.stem_detector.set_is_inserted(DT.BF,False) #retract BF detector
            grpc_client.stem_detector.set_is_inserted(DT.HAADF, False) #retract ADF detector
            for i in tqdm(range(5),desc="stabilising after detector retraction",unit=""):
                sleep(1) #wait for 5 seconds
        grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera if in off-axis mode
        sleep(0.2)  # stabilisation after deflector change

        grpc_client.scanning.set_camera_roi(roi_mode=RM.Disabled,use16bit=True)
         #sets to ROI mode

        time_now = datetime.datetime.now()
        print(time_now)

        scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera])
        for i in range(0,10000):
            state = grpc_client.microscope.get_state().name
            if state == "Acquiring":
                pass
            else:
                time_after = datetime.datetime.now()
                break
        print(time_after)

        delta_time = datetime.timedelta(time_now,time_after)
        print("Delta time",delta_time.microseconds)

        print("time increment in FPS",1/(delta_time.microseconds/(scan_width_px*scan_width_px))*1e6)
        header, data = cache_client.get_item(scan_id, scan_width_px*scan_width_px)  # cache retrieval in rows
        camera_size = data["cameraData"].shape[1], data["cameraData"].shape[2]  # gets shape of diffraction patterns
        image_data = data["cameraData"]  # take the data for that row
        image_array = np.reshape(image_data, (scan_width_px,scan_width_px, camera_size[0], camera_size[1]))  # reshapes data to an individual image #TODO necessary?

        return image_array
    else:
        print("Too many scan pixels, use fewer than 32")