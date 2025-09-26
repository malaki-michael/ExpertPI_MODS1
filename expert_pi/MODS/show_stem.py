import matplotlib.pyplot as plt
import numpy as np
from expert_pi.MODS import generals
import easygui
import cv2 as cv2
import tkinter as tk





def bfdf_images(bf_image = None, df_image = None, metadata = None):
    """
    Plot images from numpy data based on acqusition type.
    bf_image = NxN array
    df_image = NxN array
    metadata = dictionary
    """

    message = ""

    if metadata == None:
        metadata = generals.collect_metadata()
    
        
    plt.subplot(1,2,1)
    #plt.axis("off")
    plt.xlabel("BF")
    plt.imshow(bf_image, cmap = "gray")

    plt.subplot(1,2,2)
    #plt.axis("off")
    plt.xlabel(f"DF")
    plt.imshow(df_image, cmap = "gray")
    plt.tight_layout()



def virtual_adf(dataset =None):
    """
    Creates a virtual df image and displays it
    dataset should by a 4D numpy array
    """
    main_window = tk.Tk()
    #main_window.geometry("300x300")

    frame = tk.LabelFrame(main_window, text =" Select Angular range",
                          padx = 10, pady = 10, font= ("Sans", 15,"bold"))
    #frame.pack(padx=5, pady=5)
    frame.pack(fill = tk.BOTH, expand = True, padx=5, pady=5)

    text1 = tk.Label(frame, text ="Inner", font = ("Sans", 15, "bold"))
    text1.grid(row =0, column = 0, padx = 50, pady =30)
    slider1 = tk.Scale(frame, from_ =0, to = 50, orient = "horizontal")
    slider1.grid(row = 0, column = 1)
    
    text2 = tk.Label(frame, text ="Outer", font = ("Sans", 15, "bold"))
    text2.grid(row =1, column = 0, padx = 50, pady =30)
    slider2 =tk.Scale(frame, from_=30, to=300, orient='horizontal')
    slider2.grid(row =1, column =1)

    main_window.mainloop()


    
         
        







    

    


