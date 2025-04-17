import sys
import os
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QSlider, QFileDialog, QMessageBox, QSizePolicy,
                            QFrame)
from PyQt5.QtCore import Qt
import vtk
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkRenderWindow, 
                                        vtkImageSlice, vtkImageSliceMapper,
                                        vtkVolume, vtkVolumeProperty)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM 4-View Visualizer")
        self.setGeometry(100, 100, 1600, 900)
        self.images = None
        self.current_slices = [0, 0, 0]
        self.window_level = 400
        self.window_width = 1500
        self.create_ui()

    def create_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Barra 1 - Solo botón de carga
        barra1 = QWidget()
        barra1_layout = QHBoxLayout()
        barra1.setLayout(barra1_layout)

        load_btn = QPushButton("Load DICOM")
        load_btn.clicked.connect(self.load_dicom)
        barra1_layout.addWidget(load_btn)
        barra1_layout.addStretch()
        main_layout.addWidget(barra1)

        # Barra 2 - Controles de Window/Level
        barra2 = QWidget()
        barra2_layout = QHBoxLayout()
        barra2.setLayout(barra2_layout)

        barra2_layout.addWidget(QLabel("Window Level:"))
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(-1000, 3000)
        self.wl_slider.setValue(self.window_level)
        self.wl_slider.valueChanged.connect(self.update_all_views)
        barra2_layout.addWidget(self.wl_slider)

        barra2_layout.addWidget(QLabel("Window Width:"))
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(1, 3000)
        self.ww_slider.setValue(self.window_width)
        self.ww_slider.valueChanged.connect(self.update_all_views)
        barra2_layout.addWidget(self.ww_slider)

        main_layout.addWidget(barra2)

        # Separador visual
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Contenedor principal para visualización y opciones
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Barra de visualización (75% del ancho)
        self.viz_widget = QWidget()
        self.viz_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout = QGridLayout()
        self.viz_widget.setLayout(self.viz_layout)
        content_layout.addWidget(self.viz_widget, stretch=75)

        # Barra de opciones 2 (25% del ancho)
        self.options2_widget = QWidget()
        self.options2_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        options2_layout = QVBoxLayout()
        self.options2_widget.setLayout(options2_layout)
        
        # Botón de ejemplo
        hello_btn = QPushButton("Hola")
        hello_btn.clicked.connect(lambda: QMessageBox.information(self, "Saludo", "¡Hola! Esta es la barra de opciones"))
        options2_layout.addWidget(hello_btn)
        options2_layout.addStretch()
        
        self.options2_widget.setStyleSheet("background-color: #f0f0f0;")
        content_layout.addWidget(self.options2_widget, stretch=25)

        # Crear widgets VTK
        self.create_vtk_widgets()

    def load_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
            
        try:
            # Buscar archivos DICOM recursivamente
            dicom_files = []
            for root, _, files in os.walk(folder):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(file_path)
                        reader.ReadImageInformation()
                        dicom_files.append(file_path)
                    except:
                        continue
            
            if not dicom_files:
                raise ValueError("No se encontraron archivos DICOM válidos en la carpeta seleccionada")
            
            # Leer la serie
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            self.images = sitk.GetArrayFromImage(image)
            self.spacing = image.GetSpacing()
            self.origin = image.GetOrigin()
            
            # Inicializar posiciones de slice
            self.current_slices = [
                self.images.shape[0] // 2,
                self.images.shape[1] // 2,
                self.images.shape[2] // 2
            ]
            
            # Configurar sliders
            self.axial_slider.setRange(0, self.images.shape[0] - 1)
            self.axial_slider.setValue(self.current_slices[0])
            
            self.sagittal_slider.setRange(0, self.images.shape[1] - 1)
            self.sagittal_slider.setValue(self.current_slices[1])
            
            self.coronal_slider.setRange(0, self.images.shape[2] - 1)
            self.coronal_slider.setValue(self.current_slices[2])
            
            self.create_vtk_widgets()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar serie DICOM:\n{str(e)}")

    def create_vtk_widgets(self):
        # Limpiar widgets anteriores
        for i in reversed(range(self.viz_layout.count())): 
            widget = self.viz_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Top-left: Axial
        axial_header = QWidget()
        axial_header_layout = QHBoxLayout()
        axial_header.setLayout(axial_header_layout)
        axial_header_layout.addWidget(QLabel("Axial View"))
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.valueChanged.connect(self.update_slices)
        axial_header_layout.addWidget(self.axial_slider)
        self.viz_layout.addWidget(axial_header, 0, 0)
        
        self.axial_vtk = QVTKRenderWindowInteractor()
        self.axial_vtk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout.addWidget(self.axial_vtk, 1, 0)

        # Top-right: Sagittal
        sagittal_header = QWidget()
        sagittal_header_layout = QHBoxLayout()
        sagittal_header.setLayout(sagittal_header_layout)
        sagittal_header_layout.addWidget(QLabel("Sagittal View"))
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(self.update_slices)
        sagittal_header_layout.addWidget(self.sagittal_slider)
        self.viz_layout.addWidget(sagittal_header, 0, 1)
        
        self.sagittal_vtk = QVTKRenderWindowInteractor()
        self.sagittal_vtk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout.addWidget(self.sagittal_vtk, 1, 1)

        # Bottom-left: Coronal
        coronal_header = QWidget()
        coronal_header_layout = QHBoxLayout()
        coronal_header.setLayout(coronal_header_layout)
        coronal_header_layout.addWidget(QLabel("Coronal View"))
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(self.update_slices)
        coronal_header_layout.addWidget(self.coronal_slider)
        self.viz_layout.addWidget(coronal_header, 2, 0)
        
        self.coronal_vtk = QVTKRenderWindowInteractor()
        self.coronal_vtk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout.addWidget(self.coronal_vtk, 3, 0)

        # Bottom-right: 3D Volume
        volume_header = QWidget()
        volume_header_layout = QHBoxLayout()
        volume_header.setLayout(volume_header_layout)
        volume_header_layout.addWidget(QLabel("3D Volume"))
        self.viz_layout.addWidget(volume_header, 2, 1)
        
        self.volume_vtk = QVTKRenderWindowInteractor()
        self.volume_vtk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout.addWidget(self.volume_vtk, 3, 1)

        # Crear renderers
        self.axial_renderer = vtk.vtkRenderer()
        self.sagittal_renderer = vtk.vtkRenderer()
        self.coronal_renderer = vtk.vtkRenderer()
        self.volume_renderer = vtk.vtkRenderer()

        # Asignar renderers a ventanas
        self.axial_vtk.GetRenderWindow().AddRenderer(self.axial_renderer)
        self.sagittal_vtk.GetRenderWindow().AddRenderer(self.sagittal_renderer)
        self.coronal_vtk.GetRenderWindow().AddRenderer(self.coronal_renderer)
        self.volume_vtk.GetRenderWindow().AddRenderer(self.volume_renderer)

        # Configurar estilos de interactor
        style = vtkInteractorStyleImage()
        self.axial_vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.sagittal_vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.coronal_vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        if self.images is not None:
            self.update_all_views()

    def update_slices(self):
        self.current_slices = [
            self.axial_slider.value(),
            self.sagittal_slider.value(),
            self.coronal_slider.value()
        ]
        self.update_all_views()

    def update_all_views(self):
        if self.images is None:
            return
            
        self.window_level = self.wl_slider.value()
        self.window_width = self.ww_slider.value()
        
        # Actualizar vistas 2D
        self.update_2d_views()
        
        # Actualizar vista 3D
        self.update_3d_volume()

    def update_2d_views(self):
        # Limpiar actores anteriores
        self.axial_renderer.RemoveAllViewProps()
        self.sagittal_renderer.RemoveAllViewProps()
        self.coronal_renderer.RemoveAllViewProps()
        
        # Crear imágenes VTK desde arrays numpy
        axial_image = self.numpy_to_vtk_image(self.images[self.current_slices[0], :, :])
        sagittal_image = self.numpy_to_vtk_image(self.images[:, self.current_slices[1], :])
        coronal_image = self.numpy_to_vtk_image(self.images[:, :, self.current_slices[2]])
        
        # Configurar vistas
        self.setup_slice_view(self.axial_renderer, axial_image)
        self.setup_slice_view(self.sagittal_renderer, sagittal_image)
        self.setup_slice_view(self.coronal_renderer, coronal_image)
        
        # Renderizar todas las vistas
        self.axial_vtk.GetRenderWindow().Render()
        self.sagittal_vtk.GetRenderWindow().Render()
        self.coronal_vtk.GetRenderWindow().Render()

    def update_3d_volume(self):
        self.volume_renderer.RemoveAllViewProps()
        
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(self.images.shape[2], self.images.shape[1], self.images.shape[0])
        vtk_image.SetSpacing(self.spacing)
        vtk_image.SetOrigin(self.origin)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(
            self.images.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_image.GetPointData().SetScalars(vtk_array)
        
        volume_mapper = vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)
        
        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        
        # Ajustar funciones de transferencia basadas en window level/width
        min_val = self.window_level - self.window_width/2
        max_val = self.window_level + self.window_width/2
        
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
        
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(min_val, 0.0)
        opacity_func.AddPoint(max_val, 0.7)
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        self.volume_renderer.AddVolume(volume)
        self.volume_renderer.ResetCamera()
        self.volume_vtk.GetRenderWindow().Render()

    def numpy_to_vtk_image(self, slice_data):
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(
            slice_data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image

    def setup_slice_view(self, renderer, image):
        mapper = vtkImageSliceMapper()
        mapper.SetInputData(image)
        
        actor = vtkImageSlice()
        actor.SetMapper(mapper)
        
        prop = actor.GetProperty()
        prop.SetColorWindow(self.window_width)
        prop.SetColorLevel(self.window_level)
        
        renderer.AddViewProp(actor)
        renderer.ResetCamera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())