import os
import vtk
import numpy as np
from vtk.util import numpy_support
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, 
                            QWidget, QSlider, QSplitter, QLabel)
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer3D(QMainWindow):
    def __init__(self, dicom_dir):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.current_axial = 0
        self.current_sagittal = 0
        self.current_coronal = 0
        
        self.init_ui()
        self.load_dicom_series()
        self.setup_visualizations()
        self.create_sliders()
        self.setup_cameras()
        
    def init_ui(self):
        """Inicializa la interfaz gráfica principal con 3 vistas"""
        self.setWindowTitle("Visualizador DICOM 3D")
        self.setGeometry(100, 100, 1200, 600)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: black;")
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Crear widgets para cada vista
        self.axial_widget = self.create_view_widget("Axial")
        self.sagittal_widget = self.create_view_widget("Sagital")
        self.coronal_widget = self.create_view_widget("Coronal")
        
        self.splitter.addWidget(self.axial_widget)
        self.splitter.addWidget(self.sagittal_widget)
        self.splitter.addWidget(self.coronal_widget)
        
    def create_view_widget(self, view_name):
        """Crea un widget para una vista específica"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        # Etiqueta para el nombre de la vista
        label = QLabel(view_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(label)
        
        # Widget VTK
        vtk_widget = QVTKRenderWindowInteractor(widget)
        vtk_widget.setStyleSheet("background-color: black;")
        layout.addWidget(vtk_widget)
        
        return widget
        
    def load_dicom_series(self):
        """Carga la serie DICOM y la convierte a matriz 3D"""
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.dicom_dir)
        self.reader.Update()
        
        # Obtener datos como matriz numpy 3D
        vtk_data = self.reader.GetOutput().GetPointData().GetScalars()
        dims = self.reader.GetOutput().GetDimensions()
        self.np_volume = numpy_support.vtk_to_numpy(vtk_data).reshape(dims[2], dims[1], dims[0])
        
        # Guardar dimensiones
        self.dims = dims
        self.spacing = self.reader.GetOutput().GetSpacing()

        # Punto inicial de los sliders
        self.current_axial = dims[2] // 2
        self.current_sagittal = dims[0] // 2
        self.current_coronal = dims[1] // 2
        
        print(f"Dimensiones del volumen: {dims}")
        print(f"Espaciado: {self.spacing}")
        
    def setup_visualizations(self):
        """Configura las visualizaciones para las 3 vistas"""
        # Configuración común para todas las vistas
        self.window = 2000
        self.level = 500
        
        # Vista Axial
        self.axial_vtk_widget = self.axial_widget.findChild(QVTKRenderWindowInteractor)
        self.axial_renderer = vtk.vtkRenderer()
        self.axial_renderer.SetBackground(0, 0, 0)
        self.axial_vtk_widget.GetRenderWindow().AddRenderer(self.axial_renderer)
        
        # Vista Sagital
        self.sagittal_vtk_widget = self.sagittal_widget.findChild(QVTKRenderWindowInteractor)
        self.sagittal_renderer = vtk.vtkRenderer()
        self.sagittal_renderer.SetBackground(0, 0, 0)
        self.sagittal_vtk_widget.GetRenderWindow().AddRenderer(self.sagittal_renderer)
        
        # Vista Coronal
        self.coronal_vtk_widget = self.coronal_widget.findChild(QVTKRenderWindowInteractor)
        self.coronal_renderer = vtk.vtkRenderer()
        self.coronal_renderer.SetBackground(0, 0, 0)
        self.coronal_vtk_widget.GetRenderWindow().AddRenderer(self.coronal_renderer)
        
        # Actualizar todas las vistas
        self.update_all_views()
        
    def update_all_views(self):
        """Actualiza las tres vistas con los slices actuales"""
        self.update_axial_view()
        self.update_sagittal_view()
        self.update_coronal_view()
        
    def update_axial_view(self):
        """Actualiza la vista axial con el slice actual"""
        # Obtener slice axial
        axial_slice = self.np_volume[self.current_axial, :, :]
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(axial_slice, self.axial_renderer, self.axial_vtk_widget)
        
    def update_sagittal_view(self):
        """Actualiza la vista sagital con el slice actual"""
        # Obtener slice sagital (necesitamos transponer para orientación correcta)
        sagittal_slice = self.np_volume[:, :, self.current_sagittal].T
        sagittal_slice = np.rot90(sagittal_slice, k=1, axes=(0, 1))
        sagittal_slice = cv2.resize(sagittal_slice, (512,512), interpolation=cv2.INTER_LANCZOS4)
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(sagittal_slice, self.sagittal_renderer, self.sagittal_vtk_widget)
        
    def update_coronal_view(self):
        """Actualiza la vista coronal con el slice actual"""
        # Obtener slice coronal (necesitamos transponer para orientación correcta)
        coronal_slice = self.np_volume[:, self.current_coronal, :].T
        coronal_slice = np.rot90(coronal_slice, k=1, axes=(0, 1))
        coronal_slice = cv2.resize(coronal_slice, (self.dims[0]-1,self.dims[1]-1), interpolation=cv2.INTER_LANCZOS4)
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(coronal_slice, self.coronal_renderer, self.coronal_vtk_widget)
        
    def display_slice(self, slice_data, renderer, vtk_widget):
        """Muestra un slice 2D en el renderer especificado"""
        # Limpiar el renderer primero
        renderer.RemoveAllViewProps()
        
        # Convertir numpy array a imagen VTK
        vtk_image = numpy_support.numpy_to_vtk(slice_data.ravel(), deep=True)
        vtk_image.SetNumberOfComponents(1)
        
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
        image_data.GetPointData().SetScalars(vtk_image)
        
        # Crear mapper y actor
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputData(image_data)
        mapper.SetSliceNumber(0)
        
        property = vtk.vtkImageProperty()
        property.SetColorWindow(self.window)
        property.SetColorLevel(self.level)
        
        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.SetProperty(property)
        
        renderer.AddActor(actor)
        renderer.ResetCamera()
        vtk_widget.GetRenderWindow().Render()
        
    def create_sliders(self):
        """Crea sliders para navegar en las tres vistas"""
        # Slider Axial
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(self.dims[2] - 1)
        self.axial_slider.setValue(self.dims[2] // 2)
        self.set_slider_style(self.axial_slider)
        self.axial_slider.valueChanged.connect(self.on_axial_slider_changed)
        self.axial_widget.layout().addWidget(self.axial_slider)
        
        # Slider Sagital
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(self.dims[0] - 1)
        self.sagittal_slider.setValue(self.dims[0] // 2)
        self.set_slider_style(self.sagittal_slider)
        self.sagittal_slider.valueChanged.connect(self.on_sagittal_slider_changed)
        self.sagittal_widget.layout().addWidget(self.sagittal_slider)
        
        # Slider Coronal
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(self.dims[1] - 1)
        self.coronal_slider.setValue(self.dims[1] // 2)
        self.set_slider_style(self.coronal_slider)
        self.coronal_slider.valueChanged.connect(self.on_coronal_slider_changed)
        self.coronal_widget.layout().addWidget(self.coronal_slider)
        
    def set_slider_style(self, slider):
        """Configura el estilo visual del slider"""
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff0000;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #666;
            }
        """)
        
    def setup_cameras(self):
        """Configura las cámaras para todas las vistas"""
        for renderer in [self.axial_renderer, self.sagittal_renderer, self.coronal_renderer]:
            renderer.ResetCamera()
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            
    def on_axial_slider_changed(self, value):
        """Manejador para cambios en el slider axial"""
        self.current_axial = value
        self.update_axial_view()
        
    def on_sagittal_slider_changed(self, value):
        """Manejador para cambios en el slider sagital"""
        self.current_sagittal = value
        self.update_sagittal_view()
        
    def on_coronal_slider_changed(self, value):
        """Manejador para cambios en el slider coronal"""
        self.current_coronal = value
        self.update_coronal_view()
        
    def closeEvent(self, event):
        """Maneja el cierre de la ventana"""
        # Limpiar los renderers
        self.axial_vtk_widget.GetRenderWindow().Finalize()
        self.sagittal_vtk_widget.GetRenderWindow().Finalize()
        self.coronal_vtk_widget.GetRenderWindow().Finalize()
        
        super().closeEvent(event)

def main():
    import sys
    
    dicom_directory = "Img"  # Cambia esto por tu directorio DICOM
    if not os.path.isdir(dicom_directory):
        print(f"Error: No se encontró la carpeta {dicom_directory}")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    viewer = DICOMViewer3D(dicom_directory)
    viewer.show()
    
    # Inicializar los widgets VTK
    viewer.axial_vtk_widget.Initialize()
    viewer.sagittal_vtk_widget.Initialize()
    viewer.coronal_vtk_widget.Initialize()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()