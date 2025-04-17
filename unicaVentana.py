import os
import vtk
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkImageSliceMapper, vtkImageSlice,
    vtkImageProperty
)
from vtkmodules.vtkImagingCore import vtkImageFlip
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer(QMainWindow):
    def __init__(self, dicom_dir):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.init_ui()
        self.load_dicom_series()
        self.setup_visualization()
        self.create_slider()
        self.setup_camera()
        
    def init_ui(self):
        """Inicializa la interfaz gráfica principal"""
        self.setWindowTitle("Trabajo Final Alejandro Díaz Montes de Oca")
        self.setGeometry(100, 100, 800, 600)
        
        # Widget central con fondo negro
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: black;")
        self.setCentralWidget(central_widget)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(self.layout)
        
        # Widget VTK con fondo negro
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        self.vtk_widget.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.vtk_widget)
        
    def load_dicom_series(self):
        """Carga la serie de imágenes DICOM desde el directorio"""
        self.reader = vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.dicom_dir)
        self.reader.Update()
        
        # Información de diagnóstico
        self.num_slices = self.reader.GetOutput().GetExtent()[5] + 1
        print(f"Número de imágenes DICOM cargadas: {self.num_slices}")
        print(f"Rango de valores: {self.reader.GetOutput().GetScalarRange()}")
        
    def setup_visualization(self):
        """Configura la visualización VTK con volteo vertical"""
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetViewport(0, 0, 1, 1)
        
        # Aplicar volteo vertical a la imagen
        flip_filter = vtkImageFlip()
        flip_filter.SetInputConnection(self.reader.GetOutputPort())
        flip_filter.SetFilteredAxis(1)  # 1 = eje Y para volteo vertical
        flip_filter.Update()
        
        self.mapper = vtkImageSliceMapper()
        self.mapper.SetInputConnection(flip_filter.GetOutputPort())
        self.mapper.SetSliceNumber(0)
        
        image_property = vtkImageProperty()
        image_property.SetColorWindow(2000)
        image_property.SetColorLevel(500)
        
        self.actor = vtkImageSlice()
        self.actor.SetMapper(self.mapper)
        self.actor.SetProperty(image_property)
        self.renderer.AddActor(self.actor)
        
    def create_slider(self):
        """Crea y configura el slider para navegar entre slices"""
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_slices - 1)
        self.slider.setValue(0)
        self.set_slider_style()
        self.slider.valueChanged.connect(self.on_slice_changed)
        self.layout.addWidget(self.slider)
        
    def set_slider_style(self):
        """Configura el estilo visual del slider"""
        self.slider.setStyleSheet("""
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
        
    def setup_camera(self):
        """Configura la cámara para visualización óptima"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        
        # Ajustar el tamaño de la proyección
        bounds = self.reader.GetOutput().GetBounds()
        camera.SetParallelScale(max(bounds[1]-bounds[0], bounds[3]-bounds[2])/2)
        
    def on_slice_changed(self, value):
        """Manejador para cambios en el slider de slices"""
        self.mapper.SetSliceNumber(value)
        self.vtk_widget.GetRenderWindow().Render()

def main():
    import sys
    
    # Configuración inicial
    dicom_directory = "Img2"
    if not os.path.isdir(dicom_directory):
        print(f"Error: No se encontró la carpeta {dicom_directory}")
        sys.exit(1)
    
    # Crear y mostrar la aplicación
    app = QApplication(sys.argv)
    viewer = DICOMViewer(dicom_directory)
    viewer.show()
    viewer.vtk_widget.Initialize()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()