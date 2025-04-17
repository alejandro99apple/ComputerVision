import os
import vtk
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkImageSliceMapper, vtkImageSlice,
    vtkImageProperty, vtkVolume, vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkImagingCore import vtkImageFlip
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QSlider, QSplitter
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer(QMainWindow):
    def __init__(self, dicom_dir):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.init_ui()
        self.load_dicom_series()
        self.setup_volumetric_visualization()
        self.create_sliders()
        self.setup_cameras()
        
    def init_ui(self):
        self.setWindowTitle("Visualizador Volumétrico DICOM")
        self.setGeometry(100, 100, 1200, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Vista axial tradicional
        self.axial_widget = QWidget()
        self.axial_layout = QVBoxLayout()
        self.axial_widget.setLayout(self.axial_layout)
        
        # Vista volumétrica sagital
        self.volume_widget = QWidget()
        self.volume_layout = QVBoxLayout()
        self.volume_widget.setLayout(self.volume_layout)
        
        self.axial_vtk_widget = QVTKRenderWindowInteractor(self.axial_widget)
        self.volume_vtk_widget = QVTKRenderWindowInteractor(self.volume_widget)
        
        self.axial_layout.addWidget(self.axial_vtk_widget)
        self.volume_layout.addWidget(self.volume_vtk_widget)
        
        self.splitter.addWidget(self.axial_widget)
        self.splitter.addWidget(self.volume_widget)
        
    def load_dicom_series(self):
        self.reader = vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.dicom_dir)
        self.reader.Update()
        extent = self.reader.GetOutput().GetExtent()
        self.axial_slices = extent[5] + 1
        print(f"Dimensiones DICOM: {extent}")

    def setup_volumetric_visualization(self):
        # Configuración del mapper volumétrico
        self.volume_mapper = vtkGPUVolumeRayCastMapper()
        self.volume_mapper.SetInputConnection(self.reader.GetOutputPort())
        self.volume_mapper.SetBlendModeToComposite()
        
        # Función de transferencia para CT
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Aire
        color_func.AddRGBPoint(-500, 0.0, 0.0, 0.1)   # Pulmón
        color_func.AddRGBPoint(40, 0.9, 0.6, 0.3)     # Tejidos blandos
        color_func.AddRGBPoint(400, 1.0, 1.0, 0.9)    # Hueso
        
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-1000, 0.0)
        opacity_func.AddPoint(-500, 0.1)
        opacity_func.AddPoint(40, 0.2)
        opacity_func.AddPoint(400, 0.8)
        
        # Propiedades del volumen
        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        
        # Crear el volumen
        self.volume = vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(volume_property)
        
        # Configurar renderer volumétrico
        self.volume_renderer = vtkRenderer()
        self.volume_renderer.SetBackground(0, 0, 0)
        self.volume_renderer.AddVolume(self.volume)
        self.volume_vtk_widget.GetRenderWindow().AddRenderer(self.volume_renderer)
        
        # Configurar plano de corte sagital
        self.volume_mapper.SetBlendModeToSlice()
        self.volume_mapper.SetSlice(256)  # Posición inicial central
        
        # Vista axial tradicional
        self.setup_axial_view()

    def setup_axial_view(self):
        self.axial_renderer = vtkRenderer()
        self.axial_renderer.SetBackground(0, 0, 0)
        self.axial_vtk_widget.GetRenderWindow().AddRenderer(self.axial_renderer)
        
        flip = vtkImageFlip()
        flip.SetInputConnection(self.reader.GetOutputPort())
        flip.SetFilteredAxis(1)
        
        self.axial_mapper = vtkImageSliceMapper()
        self.axial_mapper.SetInputConnection(flip.GetOutputPort())
        self.axial_mapper.SetSliceNumber(self.axial_slices // 2)
        
        prop = vtkImageProperty()
        prop.SetColorWindow(2000)
        prop.SetColorLevel(500)
        
        actor = vtkImageSlice()
        actor.SetMapper(self.axial_mapper)
        actor.SetProperty(prop)
        self.axial_renderer.AddActor(actor)

    def create_sliders(self):
        # Slider para vista axial
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(self.axial_slices - 1)
        self.axial_slider.setValue(self.axial_slices // 2)
        self.axial_slider.valueChanged.connect(self.on_axial_slice_changed)
        self.axial_layout.addWidget(self.axial_slider)
        
        # Slider para plano sagital volumétrico
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(511)  # Asumiendo 512 slices sagitales
        self.volume_slider.setValue(256)
        self.volume_slider.valueChanged.connect(self.on_volume_slice_changed)
        self.volume_layout.addWidget(self.volume_slider)

    def on_axial_slice_changed(self, value):
        self.axial_mapper.SetSliceNumber(value)
        self.axial_vtk_widget.GetRenderWindow().Render()

    def on_volume_slice_changed(self, value):
        self.volume_mapper.SetSlice(value)
        self.volume_vtk_widget.GetRenderWindow().Render()

    def setup_cameras(self):
        # Configuración cámara axial
        self.axial_renderer.ResetCamera()
        axial_cam = self.axial_renderer.GetActiveCamera()
        axial_cam.ParallelProjectionOn()
        axial_cam.SetParallelScale(300)
        
        # Configuración cámara volumétrica
        self.volume_renderer.ResetCamera()
        vol_cam = self.volume_renderer.GetActiveCamera()
        vol_cam.Azimuth(30)  # Rotación inicial para mejor visualización 3D
        vol_cam.Elevation(30)

def main():
    import sys
    
    dicom_directory = "Img"
    if not os.path.isdir(dicom_directory):
        print(f"Error: No se encontró la carpeta {dicom_directory}")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    viewer = DICOMViewer(dicom_directory)
    viewer.show()
    viewer.axial_vtk_widget.Initialize()
    viewer.volume_vtk_widget.Initialize()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()