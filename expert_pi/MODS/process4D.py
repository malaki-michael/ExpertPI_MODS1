import tkinter as tk
import easygui
import numpy as np
import os
import sys
import json
from skimage import feature
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

def bin_data(dataset, blocksize, filename):
    """
    Function downsamples the diffraction space of a 4D array and returns the binned dataset
    arguments:
    dataset = 4D numpy array - (x,y, kx, ky)  x,y ==> real space shape, kx. ky ==> diffraction space shape
    m = integer value by how many times you want to downsample
    """
    binned_data = block_reduce(dataset, block_size = (1,1, blocksize, blocksize), func = np.mean)
    print(f"\n\tThe shape of original data is {dataset.shape}")
    print(f"\n\tThe shape of downasampled data is {binned_data.shape}")
    np.save(filename+"_binned.npy", binned_data)
    print("\n\t***Downsampled Datasaved saved***")
    return binned_data


def sliders(area, avgDPB, metadata =None):
    """
    Displays a slider gui to select ADF angular ranges in mrad
    Arguments:
    area = area of the direct beam in pixels
    cal_angle = pixels per mrad
    metadata = dictionary
    """ 

    if metadata == None:
        filename = easygui.fileopenbox("Select Metadata File (.json)", "File Explorer")
        with open(filename, "r") as file:
            metadata = json.load(file)
    
    max_angle = metadata["Diffraction angle (mrad)"]
    cal_angle = avgDPB.shape[0]/max_angle                # calibration angle

    main_window = tk.Tk()
    var1 = tk.IntVar()
    var2 = tk.IntVar()

    def get_angles():
        global inner_angle
        global outer_angle
        if var1.get()>var2.get():
            print("\n\tIncorrect Angles selected: Inner angle should be greater than outer angle")
            main_window.destroy()
            sys.exit(0)
        else:
            inner_angle = var1.get()
            outer_angle = var2.get()
        main_window.destroy()

    def quit_gui():
        global inner_angle
        global outer_angle
        inner_angle ="Not Defined"
        outer_angle="Not Defined"
        main_window.destroy()

    frame = tk.LabelFrame(main_window, text = "Select Angular Range for ADF", padx=10, pady=10)
    frame.pack(padx=5, pady=5)

    in_label = tk.Label(frame, text ="Inner Angle",
                        font = ("Sans", 8, "bold"),
                        fg = "black")
    in_label.grid(row=0, column=0)

    in_slider = tk.Scale(frame, orient="horizontal", from_= area*cal_angle, to= 5*area*cal_angle, variable= var1)
    in_slider.grid(row=0, column=1)

    out_label = tk.Label(frame, text="Outer Angle",
                        font = ("Sans", 8, "bold"),
                        fg = "black")
    out_label.grid(row=1,column=0)

    out_slider = tk.Scale(frame, orient="horizontal", from_=5*area*cal_angle, to=metadata["Diffraction angle (mrad)"],variable = var2)
    out_slider.grid(row=1, column=1)

    get_btn = tk.Button(frame, text="Get Angles",
                        font =("Sans", 10, "bold"),
                        command= get_angles)
    get_btn.grid(row =2, column = 0)

    quit_btn = tk.Button(frame, text="QUIT",
                        font =("Sans", 10, "bold"),
                        fg ="red", command = quit_gui)
    quit_btn.grid(row=2, column=2)

    main_window.mainloop()

    return inner_angle, outer_angle, cal_angle
        
    
def virtual_images(data = None):
    """
    Takes a 4d numpy dataset and plots BF and ADF images from user defined angular range
    Arguments:
    dataset = (x, y, kx, ky) numpy array
    """

    if data ==None:
        filename = easygui.fileopenbox("Select Dataset (.npy)", "File Explorer")
        data = np.load(filename)

    print("\n\t---Binning---")
    diff_shape = data[0][0].shape         # assuming max diffshape is 512X512
    if diff_shape[0] == 512:
        blocksize = 4                      # to bin to 128
    elif diff_shape[0]  == 256:
        blocksize = 2                      # bin to 128
    else:
        blocksize = 1                      # keep as is

    if blocksize !=1:
        binned_data = bin_data(data, blocksize, filename)
    else:
        binned_data = data
        print("\n\tDiffraction Space shape already <= (128, 128)")
        print(f"\n\tData Shape: {binned_data.shape}")

    avgDP = np.average(data, axis = (0,1))
    avgImage = np.average(data, axis = (2,3))

    avgDPB = np.average(binned_data, axis = (0,1))
    avgImageB = np.average(binned_data, axis = (2,3))

    avg_int = np.average(avgDP)
    avg_intB = np.average(avgDPB)

    # code snippet to check the data initially both full size and binned to make sure binning has been properly done
    print("\n\t***Plotting initial Images to validate***")
    fig, ax = plt.subplots(2,2, figsize = (12,12))
    ax[0][0].imshow(avgDP, cmap ="gray", vmax= 10*avg_int)
    ax[0][0].set_title("Original DP")
    ax[0][1].imshow(avgImage, cmap ="gray")
    ax[0][1].set_title("Average Image")
    ax[1][0].imshow(avgDPB, cmap="gray", vmax= 10*avg_intB)
    ax[1][0].set_title("Binned DP")
    ax[1][1].imshow(avgImageB, cmap ="gray")
    ax[1][1].set_title("Average Image from Binned DP")
    plt.tight_layout()
    plt.show()


    print("\n\t***Finding Center of DP***")
   # code snippet to find the central beam and radius
    blob = feature.blob_log(avgDPB, min_sigma =3, max_sigma = 6, threshold = 70)
    row, column, area = blob[0,:]
    print(f"\n\tCenter: Row_no: {row}, Column_no: {row})")

    
    x, y = np.indices((avgDPB.shape[0], avgDPB.shape[1]))
    
    # code snippet to generate BF image data 
    bf_mask = (x-row)**2 + (y-column)**2 < (np.sqrt(3)*area)**2
    BF = np.zeros((avgImageB.shape[0], avgImageB.shape[1]))

    for i in range(0, avgDPB.shape[0]):
        for j in range(0, avgDPB.shape[1]):
            if bf_mask[i,j]:
                BF += binned_data[:,:,i,j]
    
    print("\n\t***VBF Image Calculated***")

    # code snippet to generate ADF images
    inner_angle = "Not defined"
    while inner_angle == "Not defined":
        inner_angle, outer_angle, cal_angle = sliders(area, avgDPB)
    
    print("\n\t***Angular Range Selected***")
    print(f"\n\tInner Angle: {inner_angle} mrad, Outer Angle:{outer_angle} mrad")
    inner_angle_px = cal_angle*inner_angle
    outer_angle_px = cal_angle*outer_angle

    mask_1 = (x-row)**2 + (y-column)**2 > inner_angle_px**2
    mask_2 = (x-row)**2 + (y-column)**2 < (0.5*outer_angle_px)**2
    adf_mask = np.logical_and(mask_1, mask_2)

    ADF = np.zeros((avgImageB.shape[0], avgImage.shape[1]))
    for i in range(0, avgDPB.shape[0]):
        for j in range(0, avgDPB.shape[1]):
            if adf_mask[i,j]:
                ADF += binned_data[:,:,i,j]
    
    print("\n\t***ADF Image Calculated***")

    print("\n\t***Plotting VBF and ADF Images**")

    fig, ax = plt.subplots(2,2, figsize = (12,12))
    
    ax[0][0].imshow(avgDPB, cmap ="gray", vmax= 10*avg_intB)
    ax[0][0].imshow(bf_mask, cmap ="OrRd", alpha = 0.5)
    ax[0][0].set_title("BF Dectector Overlay")
    ax[0][1].imshow(BF, cmap ="gray")
    ax[0][1].set_title("VBF Image")
    ax[1][0].imshow(avgDPB, cmap = "gray", vmax=10*avg_intB)
    ax[1][0].imshow(adf_mask, cmap = 'OrRd', alpha = 0.5)
    ax[1][0].set_title("Annular Detector Overlay")
    ax[1][1].imshow(ADF, cmap ="gray")
    ax[1][1].set_title("Virtual ADF Image")
    plt.tight_layout()
    plt.show()


    # Go through py4dstem tutorial

    









    
    
    
    
    
    