import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.fft import fft2, fftshift
import tkinter as tk
from tkinter import simpledialog
import piexif
import multiprocessing as mp
from expert_pi.acceptance_interface.specification_criteria import shared_directory,stem_resolution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  Circle
from matplotlib.widgets import Slider
from scipy.fft import fft2, fftshift
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def askfloat_helper(*args,**kwargs):
    root = tk.Tk()
    root.withdraw()
    result_queue = kwargs.pop('result_queue')
    kwargs['parent'] = root
    result_queue.put(simpledialog.askfloat(prompt="FOV is missing from metadata, please add the scan x-axis FOV in microns",title="Scan distortion test",*args,**kwargs))
    root.destroy()


def ask_float(*args, **kwargs):
    result_queue = mp.Queue()
    kwargs['result_queue'] = result_queue
    askfloat_thread = mp.Process(target=askfloat_helper,args=args,kwargs=kwargs)
    askfloat_thread.start()
    result = result_queue.get()
    if askfloat_thread.is_alive():
        askfloat_thread.join()
    return result


def interactive_fft_tool_with_scaling(image_array, pixel_size_nm):
    """
    Displays an FFT of an image in a thread-safe interactive window.
    Allows brightness adjustment and calculates distances, spatial frequency, and real-space wavelength.

    Parameters:
        image_array (numpy.ndarray): The input 2D image array.
        pixel_size_nm (float): The size of a single pixel in nanometers.

    Returns:
        results (list): List of tuples containing:
            - (x, y): Pixel coordinates of the clicked point.
            - distance_pixels: Distance in pixels from the center.
            - spatial_frequency: Frequency in 1/nm.
            - real_space_distance: Real-space wavelength in nm.
    """
    # Compute FFT and log magnitude
    fft_result = fft2(image_array)
    fft_shifted = fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # Get image dimensions and center
    ny, nx = image_array.shape
    center_y, center_x = ny // 2, nx // 2

    # FFT size
    fft_size = nx

    #spec = stem_resolution

    #spec_distance_pixels = 1/(spec / (fft_size * pixel_size_nm))
    #real_space_distance = (fft_size * pixel_size_nm) / distance_pixels
    #spec_circle = plt.Circle(xy=(fft_size/2,fft_size/2),radius=spec_distance_pixels,alpha=1,color="r")

    crop_size = 512
    crop_start_y, crop_start_x = center_y - crop_size // 2, center_x - crop_size // 2
    crop_end_y, crop_end_x = crop_start_y + crop_size, crop_start_x + crop_size
    cropped_magnitude = log_magnitude[crop_start_y:crop_end_y, crop_start_x:crop_end_x]



    # Create the figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(12,12)
    plt.subplots_adjust(bottom=0.25)  # Space for the slider
    fft_image = ax.imshow(cropped_magnitude, cmap='grey')
    #circle = Circle((fft_size/2, fft_size/2), spec_distance_pixels, color='red', fill=False, lw=1,alpha=0.2)
    #ax.add_patch(circle)
    ax.set_title("HR spacing measurement \n Click on the FFT spots to indicate spot positions, then click continue to measure them")

    # Add Close Button
    ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])  # Position: [left, bottom, width, height]
    close_button = Button(ax_button, "Continue")

    # Slider to adjust brightness
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    brightness_slider = Slider(ax_slider, "Brightness", 0.1, 3.0, valinit=1.0)

    # Function to update brightness
    def update_brightness(val):
        brightness = brightness_slider.val
        fft_image.set_data(cropped_magnitude * brightness)
        fig.canvas.draw_idle()

    brightness_slider.on_changed(update_brightness)

    # Click event to capture points and calculate distances
    results = []

    def close_window(event):
        plt.close(fig)


    def on_click(event):
        if event.inaxes == ax:  # Ensure clicks are within the FFT plot
            x_cropped, y_cropped = int(event.xdata), int(event.ydata)
            # Map back to full FFT coordinates
            x_full = x_cropped + crop_start_x
            y_full = y_cropped + crop_start_y
            # Calculate distance in pixels from the center of the full FFT
            distance_pixels = np.sqrt((x_full - center_x)**2 + (y_full - center_y)**2)
            # Calculate spatial frequency in 1/nm
            spatial_frequency = distance_pixels / (fft_size * pixel_size_nm)
            # Calculate real-space wavelength in nm
            lattice_spacing = (fft_size * pixel_size_nm) / distance_pixels if distance_pixels > 0 else np.inf
            results.append(lattice_spacing)


    fig.canvas.mpl_connect("button_press_event", on_click)
    close_button.on_clicked(close_window)
    # Display the plot
    plt.show(block=False)

    # Return results for further processing
    return results


def interactive_fft_tk(image_array, pixel_size_nm):
    """
    Displays the central cropped FFT of an image in a Tkinter window.
    Allows brightness adjustment, calculates distances, spatial frequency, real-space wavelength,
    and adds a circle annotation for selected points.

    Parameters:
        image_array (numpy.ndarray): The input 2D image array.
        pixel_size_nm (float): The size of a single pixel in nanometers.

    Returns:
        results (list): List of tuples containing:
            - (x, y): Pixel coordinates of the clicked point in the cropped image.
            - distance_pixels: Distance in pixels from the center.
            - spatial_frequency: Frequency in 1/nm.
            - real_space_wavelength: Real-space wavelength in nm.
    """
    # FFT and Log Magnitude Calculation
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # Image Dimensions
    ny, nx = image_array.shape
    center_y, center_x = ny // 2, nx // 2

    # Crop Central 512x512 Region for Display
    crop_size = 512
    crop_start_y, crop_start_x = center_y - crop_size // 2, center_x - crop_size // 2
    crop_end_y, crop_end_x = crop_start_y + crop_size, crop_start_x + crop_size
    cropped_magnitude = log_magnitude[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    # Initialize Tkinter GUI
    root = tk.Tk()
    root.title("Interactive FFT Tool")
    results = []

    # Create Matplotlib Figure
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    fft_image = ax.imshow(cropped_magnitude, cmap='viridis', extent=(-crop_size // 2, crop_size // 2, -crop_size // 2, crop_size // 2))
    ax.set_title("Cropped FFT")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    circle = Circle((0, 0), 10, color='red', fill=False, lw=2)
    ax.add_patch(circle)

    # Embed Matplotlib Figure in Tkinter Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Brightness Adjustment Slider
    brightness_label = ttk.Label(root, text="Brightness:")
    brightness_label.pack(side=tk.LEFT, padx=5)
    brightness_slider = ttk.Scale(root, from_=0.1, to=3.0, value=1.0, orient=tk.HORIZONTAL, length=200)

    def adjust_brightness(val):
        brightness = float(val)
        fft_image.set_data(cropped_magnitude * brightness)
        canvas.draw()

    brightness_slider.config(command=adjust_brightness)
    brightness_slider.pack(side=tk.LEFT, padx=5)

    # Click Event to Capture Points
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            # Coordinates within the cropped image (relative to its center)
            x_cropped, y_cropped = int(event.xdata), int(event.ydata)
            x_relative = x_cropped - crop_size // 2
            y_relative = y_cropped - crop_size // 2

            # Map to full FFT coordinates (relative to the full image center)
            x_full = center_x + x_relative
            y_full = center_y + y_relative

            # Calculate distance from the center of the full FFT
            distance_pixels = np.sqrt((x_full - center_x) ** 2 + (y_full - center_y) ** 2)

            # Calculate spatial frequency in 1/nm
            spatial_frequency = distance_pixels / (nx * pixel_size_nm)

            # Calculate real-space wavelength in nm
            real_space_wavelength = (nx * pixel_size_nm) / distance_pixels if distance_pixels > 0 else np.inf
            results.append(real_space_wavelength)
            canvas.draw()
            print(f"Clicked: ({x_cropped}, {y_cropped}), Distance: {distance_pixels:.2f} pixels, "
                  f"Spatial Frequency: {spatial_frequency:.5f} 1/nm, Wavelength: {real_space_wavelength:.2f} nm")

    canvas.mpl_connect("button_press_event", on_click)

    # Close Button
    close_button = ttk.Button(root, text="Close", command=root.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)

    # Run Tkinter Main Loop
    root.mainloop()

    return results



def interactive_fft_full(image_array, pixel_size_nm):
    """
    Displays the FFT of an image in a Tkinter window.
    Allows brightness adjustment, calculates distances, spatial frequency, and real-space wavelength.

    Parameters:
        image_array (numpy.ndarray): The input 2D image array.
        pixel_size_nm (float): The size of a single pixel in nanometers.

    Returns:
        results (list): List of tuples containing:
            - (x, y): Pixel coordinates of the clicked point.
            - distance_pixels: Distance in pixels from the center.
            - spatial_frequency: Frequency in 1/nm.
            - real_space_wavelength: Real-space wavelength in nm.
    """
    # FFT and Log Magnitude Calculation
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # Image Dimensions
    ny, nx = image_array.shape
    center_y, center_x = ny // 2, nx // 2

    # Initialize Tkinter GUI
    root = tk.Tk()
    root.title("Interactive FFT Tool (Full)")

    results = []

    # Create Matplotlib Figure
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    fft_image = ax.imshow(log_magnitude, cmap='viridis')
    ax.set_title("Full FFT")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    circle = Circle((0, 0), 10, color='red', fill=False, lw=2)
    ax.add_patch(circle)

    # Embed Matplotlib Figure in Tkinter Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Brightness Adjustment Slider
    brightness_label = ttk.Label(root, text="Brightness:")
    brightness_label.pack(side=tk.LEFT, padx=5)
    brightness_slider = ttk.Scale(root, from_=0.1, to=3.0, value=1.0, orient=tk.HORIZONTAL, length=200)

    def adjust_brightness(val):
        brightness = float(val)
        fft_image.set_data(log_magnitude * brightness)
        canvas.draw()

    brightness_slider.config(command=adjust_brightness)
    brightness_slider.pack(side=tk.LEFT, padx=5)

    # Click Event to Capture Points
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            x_full, y_full = int(event.xdata), int(event.ydata)
            # Calculate distance from the center of the FFT
            distance_pixels = np.sqrt((x_full - center_x) ** 2 + (y_full - center_y) ** 2)
            # Calculate spatial frequency in 1/nm
            spatial_frequency = distance_pixels / (nx * pixel_size_nm)
            # Calculate real-space wavelength in nm
            real_space_wavelength = (nx * pixel_size_nm) / distance_pixels if distance_pixels > 0 else np.inf
            results.append((x_full, y_full, distance_pixels, spatial_frequency, real_space_wavelength))
            circle.set_center((x_full, y_full))
            circle.set_radius(2)
            canvas.draw()
            print(f"Clicked: ({x_full}, {y_full}), Distance: {distance_pixels:.2f} pixels, "
                  f"Spatial Frequency: {spatial_frequency:.5f} 1/nm, Wavelength: {real_space_wavelength:.2f} nm")

    canvas.mpl_connect("button_press_event", on_click)

    # Close Button
    close_button = ttk.Button(root, text="Close", command=root.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)

    # Run Tkinter Main Loop
    root.mainloop()

    return results

# Updated function with reintroduced cropping
def interactive_fft_with_cropping(image_array, pixel_size_nm):
    """
    Displays the central cropped FFT of an image in a Tkinter window.
    Allows brightness adjustment, calculates distances, spatial frequency, real-space wavelength,
    and adds a circle annotation for selected points.

    Parameters:
        image_array (numpy.ndarray): The input 2D image array.
        pixel_size_nm (float): The size of a single pixel in nanometers.

    Returns:
        results (list): List of tuples containing:
            - (x, y): Pixel coordinates of the clicked point in the cropped image.
            - distance_pixels: Distance in pixels from the center of the full FFT.
            - spatial_frequency: Frequency in 1/nm.
            - real_space_wavelength: Real-space wavelength in nm.
    """
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Circle

    # Perform FFT and compute log magnitude
    fft_result = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # Original FFT dimensions and center
    ny, nx = image_array.shape
    center_y, center_x = ny // 2, nx // 2

    # Crop parameters
    crop_size = 512
    crop_start_y, crop_start_x = center_y - crop_size // 2, center_x - crop_size // 2
    crop_end_y, crop_end_x = crop_start_y + crop_size, crop_start_x + crop_size
    cropped_magnitude = log_magnitude[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    # Initialize Tkinter GUI
    root = tk.Tk()
    root.title("Interactive FFT Tool (Cropped)")

    results = []

    # Create Matplotlib Figure
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    fft_image = ax.imshow(cropped_magnitude, cmap='viridis', extent=(-crop_size // 2, crop_size // 2, -crop_size // 2, crop_size // 2))
    ax.set_title("Cropped FFT")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    circle = Circle((0, 0), 10, color='red', fill=False, lw=2)
    ax.add_patch(circle)

    # Embed Matplotlib Figure in Tkinter Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Brightness Adjustment Slider
    brightness_label = ttk.Label(root, text="Brightness:")
    brightness_label.pack(side=tk.LEFT, padx=5)
    brightness_slider = ttk.Scale(root, from_=0.1, to=3.0, value=1.0, orient=tk.HORIZONTAL, length=200)

    def adjust_brightness(val):
        brightness = float(val)
        fft_image.set_data(cropped_magnitude * brightness)
        canvas.draw()

    brightness_slider.config(command=adjust_brightness)
    brightness_slider.pack(side=tk.LEFT, padx=5)

    # Click Event to Capture Points
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            # Coordinates as displayed in the cropped FFT plot (centered at 0,0)
            x_display, y_display = event.xdata, event.ydata

            # Calculate distance from the center (0, 0) in display coordinates
            distance_pixels = np.sqrt(x_display ** 2 + y_display ** 2)

            # Map back to full FFT coordinates for scaling
            x_full = int(x_display + center_x)
            y_full = int(y_display + center_y)

            # Calculate spatial frequency in 1/nm
            spatial_frequency = distance_pixels / (nx * pixel_size_nm)

            # Calculate real-space wavelength in nm
            lattice_spacing = (nx * pixel_size_nm) / distance_pixels if distance_pixels > 0 else np.inf

            # Store results
            results.append(lattice_spacing)
            circle = Circle((x_display, y_display), 5, color='red', fill=False, lw=1)
            ax.add_patch(circle)
            # Update circle annotation
            canvas.draw()

    canvas.mpl_connect("button_press_event", on_click)

    # Close Button
    close_button = ttk.Button(root, text="Close", command=root.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)

    # Run Tkinter Main Loop
    root.mainloop()

    return results




def HR_resolution_check(image_path):

    image = cv2.imread(image_path,-1)
    # Try to get FOV from metadata if FOV is not provided
    fov = None
    if fov is None:
        try:
            metadata_object = piexif.load(image_path)
            pixel_size = metadata_object["0th"][piexif.ImageIFD.XResolution][1] / metadata_object["0th"][piexif.ImageIFD.XResolution][0]
            fov_meters = pixel_size * image.shape[0]
            fov = fov_meters * 1e9  # Convert to nanometers
        except (KeyError, piexif.InvalidImageDataError):
            print("Metadata retrieval failed, proceeding to user input.")

    # Fallback to `simpledialog.askfloat()` only if FOV is still None
    if fov is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        fov = simpledialog.askfloat(
            title="Scan distortion test",
            prompt="FOV is missing from metadata, please add the scan x-axis FOV in nanometers",
            parent=root
        )
        root.destroy()

    pixel_size_nm = fov / image.shape[0]

    results = interactive_fft_with_cropping(image,pixel_size_nm)

    spec_pass=False

    best_spacing = np.round(min(results),3)
    print(best_spacing)
    if best_spacing <=stem_resolution*1.1:
        spec_pass=True

    #If any spacing is smaller than or equal to the spec resolution, return pass




    return spec_pass,best_spacing


#image = cv2.imread(r"C:\Users\robert.hooley\Desktop\SKH HAADF 1.tiff",-1)

#spec_pass,result = HR_resolution_check(image,14)
#print(spec_pass,result)

