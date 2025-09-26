import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from expert_pi.acceptance_interface.specification_criteria import shared_directory
from expert_pi.acceptance_interface.tools import scan_4D
from expert_pi import grpc_client


def check_camera_speed(scan_pixels=16):

    grpc_client.illumination.set_illumination_values(100e-12,10e-3,keep_adjustments=False)
    #current_probe_current = grpc_client.illumination.get_current()*1e9 #nA

    dataset_2250,acquisition_time = scan_4D(scan_pixels,2250)
    dataset_4500,acquisition_time = scan_4D(scan_pixels,4500)

    shape_2250 = dataset_2250[0][0].shape
    shape_4500 = dataset_4500[0][0].shape

    summed_pattern_2250 = np.sum(dataset_2250)
    average_2250 = summed_pattern_2250/(scan_pixels*scan_pixels*shape_2250[0]*shape_2250[1])
    summed_pattern_4500 = np.sum(dataset_4500)
    average_4500 = summed_pattern_4500/(scan_pixels*scan_pixels*shape_4500[0]*shape_4500[1])

    ratio = average_2250/average_4500

    spec_pass = False

    if 2.1 < ratio > 1.9:
        spec_pass = True


    if spec_pass is True:
        dp_4500 = dataset_4500[(scan_pixels/2)-1][(scan_pixels/2)-1]
        cv2.putText(dp_4500, "4500 FPS", (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(255, 225, 255))
        dp_2250 = dataset_2250[(scan_pixels/2)-1][(scan_pixels/2)-1].astype(np.uint8)
        cv2.putText(dp_2250, "2250 FPS", (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5,
                    color=(255, 225, 255))

        combo_image = np.concatenate((dp_2250,dp_4500),axis=1)
        combo_image.astype(np.uint8)

        filepath = shared_directory+"//camera speed test image.tiff"

        cv2.imwrite(filename=filepath,img=combo_image)
        #plt.imshow(combo_image)
        #plt.show()

    return (pass_2250,pass_4500)
