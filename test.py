import os
import SimpleITK as sitk  # Import SimpleITK
import numpy as np
import vtk
from tkinter import filedialog, Tk
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor, vtkImageSlice, vtkImageSliceMapper
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget, vtkSliderRepresentation2D

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

def render_vtk_image_with_slider(images):
    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background

    # Create a VTK render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a VTK render window interactor
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create a mapper and slice for the image
    slice_mapper = vtkImageSliceMapper()
    slice_mapper.SetInputData(numpy_to_vtk_image(images[0, :, :]))  # Start with the first slice

    image_slice = vtkImageSlice()
    image_slice.SetMapper(slice_mapper)

    # Add the slice to the renderer
    renderer.AddViewProp(image_slice)

    # Create a slider representation
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(images.shape[0] - 1)
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
        slice_mapper.SetInputData(numpy_to_vtk_image(images[slice_index, :, :]))
        render_window.Render()

    slider_widget.AddObserver("InteractionEvent", slider_callback)

    # Render and start the interactor
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

# Ask the user to select a folder
root = Tk()
root.withdraw()  # Hide the root window
folder_selected = filedialog.askdirectory(title="Select Folder with DICOM Images")

if folder_selected:
    try:
        images = load_dicom_images(folder_selected)
        render_vtk_image_with_slider(images)  # Render with slider
    except Exception as e:
        print(f"Error loading or displaying DICOM images: {e}")
else:
    print("No folder selected.")
