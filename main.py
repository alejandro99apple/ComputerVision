import tkinter as tk
from tkinter import filedialog
import os
import SimpleITK as sitk  # Import SimpleITK
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import vtk
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor, vtkImageSlice, vtkImageSliceMapper
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget, vtkSliderRepresentation2D
from PyQt5.QtCore import Qt  # Import Qt for slider orientation

def load_dicom_images(folder_path):
    # Load DICOM images using SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    images = sitk.GetArrayFromImage(image)  # Convert to NumPy array (z, y, x)
    return images

def numpy_to_vtk_image(slice_data):
    # Convert NumPy array to vtkImageData
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    # Flatten the NumPy array and set it as the scalars for vtkImageData
    vtk_array = numpy_to_vtk(slice_data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image

def render_vtk_image_with_slider(images, title, axis):
    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background

    # Create a VTK render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)

    # Create a VTK render window interactor
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create a mapper and slice for the image
    slice_mapper = vtkImageSliceMapper()
    if axis == 0:  # XY Plane
        slice_mapper.SetInputData(numpy_to_vtk_image(images[0, :, :]))
    elif axis == 1:  # XZ Plane
        slice_mapper.SetInputData(numpy_to_vtk_image(images[:, 0, :]))
    elif axis == 2:  # YZ Plane
        slice_mapper.SetInputData(numpy_to_vtk_image(images[:, :, 0]))

    image_slice = vtkImageSlice()
    image_slice.SetMapper(slice_mapper)

    # Add the slice to the renderer
    renderer.AddViewProp(image_slice)

    # Create a slider representation
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(images.shape[axis] - 1)
    slider_rep.SetValue(0)
    slider_rep.SetTitleText("Slice")
    slider_rep.GetSliderProperty().SetColor(0, 0, 1)  # Blue slider
    slider_rep.GetTitleProperty().SetColor(0, 0, 0)  # Black title
    slider_rep.GetLabelProperty().SetColor(0, 0, 0)  # Black label
    slider_rep.GetTubeProperty().SetColor(0.8, 0.8, 0.8)  # Gray tube
    slider_rep.GetCapProperty().SetColor(0, 0, 1)  # Blue caps
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.03)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(0.9, 0.1)

    # Create a slider widget
    slider_widget = vtkSliderWidget()
    slider_widget.SetInteractor(render_window_interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()

    # Slider callback to update the slice
    def slider_callback(obj, event):
        slice_index = int(obj.GetRepresentation().GetValue())
        if axis == 0:  # XY Plane
            slice_mapper.SetInputData(numpy_to_vtk_image(images[slice_index, :, :]))
        elif axis == 1:  # XZ Plane
            slice_mapper.SetInputData(numpy_to_vtk_image(images[:, slice_index, :]))
        elif axis == 2:  # YZ Plane
            slice_mapper.SetInputData(numpy_to_vtk_image(images[:, :, slice_index]))
        render_window.Render()

    slider_widget.AddObserver("InteractionEvent", slider_callback)

    # Render and start the interactor
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

def render_3d_reconstruction(images):
    # Convert the 3D NumPy array to vtkImageData
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(images.shape[2], images.shape[1], images.shape[0])
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    # Flatten the NumPy array and set it as the scalars for vtkImageData
    vtk_array = numpy_to_vtk(images.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    vtk_image.GetPointData().SetScalars(vtk_array)

    # Create a volume mapper and volume
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)

    # Set volume properties
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # Set opacity and color transfer functions
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(0, 0.0)
    opacity_transfer_function.AddPoint(255, 1.0)

    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(255, 1.0, 1.0, 1.0)

    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.SetColor(color_transfer_function)
    volume.SetProperty(volume_property)

    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(1, 1, 1)  # White background

    # Create a VTK render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("3D Reconstruction")

    # Create a render window interactor
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render and start the interactor
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

def display_views_in_separate_windows_with_3d(images):
    # Display the XY Plane
    render_vtk_image_with_slider(images, "XY Plane (Axial)", axis=0)

    # Display the XZ Plane
    render_vtk_image_with_slider(images, "XZ Plane (Sagittal)", axis=1)

    # Display the YZ Plane
    render_vtk_image_with_slider(images, "YZ Plane (Coronal)", axis=2)

    # Display the 3D reconstruction
    render_3d_reconstruction(images)

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        print(f"Selected folder: {folder_selected}")
        try:
            images = load_dicom_images(folder_selected)
            display_views_in_separate_windows_with_3d(images)
        except Exception as e:
            print(f"Error loading or displaying DICOM images: {e}")

def update_options2_frame(selected_option):
    # Clear the options2_frame
    for widget in options2_frame.winfo_children():
        widget.destroy()

    if selected_option == "Modificar resolución":
        # Add radio buttons for "Modificar resolución"
        resolution_options = ["Resolución espacial", "Resolución radiométrica", "Resolución temporal"]
        resolution_var.set(resolution_options[0])  # Default to the first option

        for option in resolution_options:
            radio_button = tk.Radiobutton(options2_frame, text=option, variable=resolution_var, value=option, bg="lightgray")
            radio_button.pack(anchor="w", padx=10, pady=2)

    elif selected_option == "Transformar coordenadas":
        # Add radio buttons for "Transformar coordenadas"
        transform_options = ["Rotación", "Traslación", "Inclinación", "Escalamiento"]
        transform_var.set(transform_options[0])  # Default to the first option

        for option in transform_options:
            radio_button = tk.Radiobutton(options2_frame, text=option, variable=transform_var, value=option, bg="lightgray")
            radio_button.pack(anchor="w", padx=10, pady=2)

# Create the main application window
root = tk.Tk()
root.title("DICOM Viewer")
root.iconbitmap("icono.ico")  # Set the icon if available
root.geometry("1280x720")  # Set default dimensions
root.resizable(False, False)  # Allow resizing
root.configure(bg="white")  # Set background color
root.configure(cursor="hand2")  # Change cursor to hand2

# Initialize global variables for options2_frame radio button states
resolution_var = tk.StringVar()
transform_var = tk.StringVar()

# Divide the window into three sections
options_frame = tk.Frame(root)
options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

main_frame = tk.Frame(root)
main_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)  

visualization_frame = tk.Frame(main_frame, bg="gray")
visualization_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

options2_frame = tk.Frame(main_frame, bg="lightgray", width=int(1280 * 0.4))
options2_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10, expand=True)

# Add the "Select Folder" button to the options frame
select_button = tk.Button(options_frame, text="Select Folder", command=select_folder)
select_button.pack(side=tk.LEFT, padx=10)

# Add radio buttons for main options
radio_frame = tk.Frame(options_frame)
radio_frame.pack(side=tk.LEFT, padx=10)

radio_options = ["Modificar resolución", "Transformar coordenadas", "Filtrar", "Mejoramiento espacial"]
selected_option = tk.StringVar(value=radio_options[0])  # Default to the first option

for option in radio_options:
    radio_button = tk.Radiobutton(radio_frame, text=option, variable=selected_option, value=option, command=lambda: update_options2_frame(selected_option.get()))
    radio_button.pack(side=tk.LEFT, padx=5)

# Initialize the options2_frame with the default selection
resolution_var.set("Resolución espacial")  # Ensure the default is set before the first update
transform_var.set("Rotación")  # Ensure the default is set before the first update
update_options2_frame(selected_option.get())

try:
    # Run the application
    root.mainloop()
except KeyboardInterrupt:
    print("Application interrupted by user.")
