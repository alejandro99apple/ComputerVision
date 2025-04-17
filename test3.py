import sys
import os
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QSlider, QFileDialog, QMessageBox, QSizePolicy,
                            QFrame, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
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
        self.setWindowIcon(QIcon('icono.ico'))
        self.setGeometry(100, 100, 1600, 900)
        self.images = None
        self.current_slices = [0, 0, 0]
        self.window_level = 400
        self.window_width = 1500
        
        # Inicializar grupos de botones
        self.transform_options = QButtonGroup()
        self.filter_options = QButtonGroup()
        self.resolution_options = QButtonGroup()
        
        self.create_ui()

    def create_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Barra 1 - Botón de carga y opciones principales
        barra1 = QWidget()
        barra1_layout = QHBoxLayout()
        barra1.setLayout(barra1_layout)

        # Botón de carga
        load_btn = QPushButton("Load DICOM")
        load_btn.clicked.connect(self.load_dicom)
        barra1_layout.addWidget(load_btn)

        # Grupo de radio buttons para opciones principales
        self.transform_rb = QRadioButton("Transformar coordenadas")
        self.filter_rb = QRadioButton("Filtrar")
        self.resolution_rb = QRadioButton("Modificar Resolución")
        
        self.transform_rb.setChecked(True)
        
        barra1_layout.addWidget(self.transform_rb)
        barra1_layout.addWidget(self.filter_rb)
        barra1_layout.addWidget(self.resolution_rb)
        
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
        self.options2_layout = QVBoxLayout()
        self.options2_layout.setContentsMargins(0, 0, 0, 0)
        self.options2_layout.setSpacing(0)
        self.options2_widget.setLayout(self.options2_layout)
        self.options2_widget.setStyleSheet("background-color: #f0f0f0;")
        content_layout.addWidget(self.options2_widget, stretch=25)

        # Crear los contenedores de opciones
        self.create_options_containers()
        
        # Conectar señales
        self.transform_rb.toggled.connect(lambda: self.show_options_container("transform"))
        self.filter_rb.toggled.connect(lambda: self.show_options_container("filter"))
        self.resolution_rb.toggled.connect(lambda: self.show_options_container("resolution"))
        
        # Mostrar opciones iniciales
        self.show_options_container("transform")

        # Crear widgets VTK
        self.create_vtk_widgets()

    def create_options_containers(self):
        """Crea los contenedores para cada tipo de opciones alineados arriba"""
        
        # Contenedor principal para mantener todo arriba
        self.main_options_container = QWidget()
        main_options_layout = QVBoxLayout()
        main_options_layout.setContentsMargins(5, 5, 5, 5)
        main_options_layout.setSpacing(10)
        self.main_options_container.setLayout(main_options_layout)
        
        # Contenedor para transformaciones
        self.transform_container = QWidget()
        transform_layout = QVBoxLayout()
        transform_layout.setContentsMargins(0, 0, 0, 0)
        transform_layout.setSpacing(5)
        self.transform_container.setLayout(transform_layout)
        
        transform_title = QLabel("Opciones de Transformación:")
        transform_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        transform_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        transform_layout.addWidget(transform_title)
        
        options = ["Rotación", "Traslación", "Escalamiento", "Inclinación"]
        for opt in options:
            rb = QRadioButton(opt)
            rb.setStyleSheet("text-align: left; padding: 3px;")
            transform_layout.addWidget(rb, alignment=Qt.AlignTop)
            self.transform_options.addButton(rb)
        
        if self.transform_options.buttons():
            self.transform_options.buttons()[0].setChecked(True)
        
        main_options_layout.addWidget(self.transform_container, alignment=Qt.AlignTop)
        self.transform_container.hide()

        # Contenedor para filtros
        self.filter_container = QWidget()
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(5)
        self.filter_container.setLayout(filter_layout)
        
        filter_title = QLabel("Opciones de Filtrado:")
        filter_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        filter_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        filter_layout.addWidget(filter_title)
        
        options = ["Dominio Frecuencial", "Dominio Espacial"]
        for opt in options:
            rb = QRadioButton(opt)
            rb.setStyleSheet("text-align: left; padding: 3px;")
            filter_layout.addWidget(rb, alignment=Qt.AlignTop)
            self.filter_options.addButton(rb)
        
        if self.filter_options.buttons():
            self.filter_options.buttons()[0].setChecked(True)
        
        main_options_layout.addWidget(self.filter_container, alignment=Qt.AlignTop)
        self.filter_container.hide()

        # Contenedor para resolución
        self.resolution_container = QWidget()
        resolution_layout = QVBoxLayout()
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(5)
        self.resolution_container.setLayout(resolution_layout)
        
        resolution_title = QLabel("Opciones de Resolución:")
        resolution_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        resolution_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        resolution_layout.addWidget(resolution_title)
        
        options = ["Resolución Espacial", "Resolución Radiométrica", "Resolución Temporal"]
        for opt in options:
            rb = QRadioButton(opt)
            rb.setStyleSheet("text-align: left; padding: 3px;")
            resolution_layout.addWidget(rb, alignment=Qt.AlignTop)
            self.resolution_options.addButton(rb)
        
        if self.resolution_options.buttons():
            self.resolution_options.buttons()[0].setChecked(True)
        
        main_options_layout.addWidget(self.resolution_container, alignment=Qt.AlignTop)
        self.resolution_container.hide()

        # Añadir stretch para empujar todo hacia arriba
        main_options_layout.addStretch()
        
        # Añadir el contenedor principal al layout de opciones 2
        self.options2_layout.addWidget(self.main_options_container)
        self.options2_layout.addStretch()

    def show_options_container(self, option_type):
        """Muestra el contenedor de opciones correspondiente"""
        self.transform_container.setVisible(option_type == "transform")
        self.filter_container.setVisible(option_type == "filter")
        self.resolution_container.setVisible(option_type == "resolution")

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
                raise ValueError("No se encontraron archivos DICOM válidos")
            
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
            
            # Configurar sliders si existen
            if hasattr(self, 'axial_slider'):
                self.axial_slider.setRange(0, self.images.shape[0] - 1)
                self.axial_slider.setValue(self.current_slices[0])
                
                self.sagittal_slider.setRange(0, self.images.shape[1] - 1)
                self.sagittal_slider.setValue(self.current_slices[1])
                
                self.coronal_slider.setRange(0, self.images.shape[2] - 1)
                self.coronal_slider.setValue(self.current_slices[2])
            
            self.create_vtk_widgets()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar DICOM:\n{str(e)}")

    def create_vtk_widgets(self):
        # Limpiar widgets anteriores
        for i in reversed(range(self.viz_layout.count())): 
            widget = self.viz_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Configurar vistas
        views = [
            ("Axial View", 0, 0),
            ("Sagittal View", 0, 1), 
            ("Coronal View", 2, 0),
            ("3D Volume", 2, 1)
        ]
        
        for name, row, col in views:
            if name != "3D Volume":
                header = QWidget()
                header_layout = QHBoxLayout()
                header.setLayout(header_layout)
                header_layout.addWidget(QLabel(name))
                
                slider = QSlider(Qt.Horizontal)
                slider.valueChanged.connect(self.update_slices)
                header_layout.addWidget(slider)
                
                self.viz_layout.addWidget(header, row, col)
                
                # Guardar referencias a los sliders
                if name == "Axial View":
                    self.axial_slider = slider
                elif name == "Sagittal View":
                    self.sagittal_slider = slider
                elif name == "Coronal View":
                    self.coronal_slider = slider
            else:
                header = QLabel(name)
                self.viz_layout.addWidget(header, row, col)
            
            # Crear frame contenedor con bordes redondeados y fondo negro
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                    padding: 3px;
                }
            """)
            frame_layout = QVBoxLayout()
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame.setLayout(frame_layout)
            
            # Crear visor VTK
            vtk_widget = QVTKRenderWindowInteractor()
            vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            frame_layout.addWidget(vtk_widget)
            
            self.viz_layout.addWidget(frame, row+1, col)
            
            # Guardar referencias
            if name == "Axial View":
                self.axial_vtk = vtk_widget
                self.axial_frame = frame
            elif name == "Sagittal View":
                self.sagittal_vtk = vtk_widget
                self.sagittal_frame = frame
            elif name == "Coronal View":
                self.coronal_vtk = vtk_widget
                self.coronal_frame = frame
            elif name == "3D Volume":
                self.volume_vtk = vtk_widget
                self.volume_frame = frame

        # Configurar renderers con fondo negro
        self.axial_renderer = vtk.vtkRenderer()
        self.sagittal_renderer = vtk.vtkRenderer()
        self.coronal_renderer = vtk.vtkRenderer()
        self.volume_renderer = vtk.vtkRenderer()

        # Establecer fondo negro para todos los renderers
        for renderer in [self.axial_renderer, self.sagittal_renderer, 
                        self.coronal_renderer, self.volume_renderer]:
            renderer.SetBackground(0, 0, 0)  # Negro en RGB

        # Asignar renderers
        self.axial_vtk.GetRenderWindow().AddRenderer(self.axial_renderer)
        self.sagittal_vtk.GetRenderWindow().AddRenderer(self.sagittal_renderer)
        self.coronal_vtk.GetRenderWindow().AddRenderer(self.coronal_renderer)
        self.volume_vtk.GetRenderWindow().AddRenderer(self.volume_renderer)

        # Configurar estilos de interacción
        style = vtkInteractorStyleImage()
        for widget in [self.axial_vtk, self.sagittal_vtk, self.coronal_vtk]:
            widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        if self.images is not None:
            self.update_all_views()

    def update_slices(self):
        if not hasattr(self, 'axial_slider') or self.images is None:
            return
            
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
        # Limpiar vistas anteriores
        self.axial_renderer.RemoveAllViewProps()
        self.sagittal_renderer.RemoveAllViewProps()
        self.coronal_renderer.RemoveAllViewProps()
        
        # Crear imágenes VTK
        axial_img = self.numpy_to_vtk_image(self.images[self.current_slices[0], :, :])
        sagittal_img = self.numpy_to_vtk_image(self.images[:, self.current_slices[1], :])
        coronal_img = self.numpy_to_vtk_image(self.images[:, :, self.current_slices[2]])
        
        # Configurar vistas
        self.setup_slice_view(self.axial_renderer, axial_img)
        self.setup_slice_view(self.sagittal_renderer, sagittal_img)
        self.setup_slice_view(self.coronal_renderer, coronal_img)
        
        # Renderizar
        self.axial_vtk.GetRenderWindow().Render()
        self.sagittal_vtk.GetRenderWindow().Render()
        self.coronal_vtk.GetRenderWindow().Render()

    def update_3d_volume(self):
        self.volume_renderer.RemoveAllViewProps()
        
        # Convertir a VTK ImageData
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(self.images.shape[2], self.images.shape[1], self.images.shape[0])
        vtk_img.SetSpacing(self.spacing)
        vtk_img.SetOrigin(self.origin)
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(
            self.images.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_img.GetPointData().SetScalars(vtk_array)
        
        # Configurar mapeador de volumen
        volume_mapper = vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_img)
        
        # Configurar propiedades del volumen
        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        
        # Funciones de transferencia (invertidas)
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(min_val, 1.0, 1.0, 1.0)  # Blanco para huesos
        color_func.AddRGBPoint(max_val, 0.0, 0.0, 0.0)  # Negro para tejido blando
        
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(min_val, 0.7)  # Opacidad alta para huesos
        opacity_func.AddPoint(max_val, 0.0)  # Opacidad baja para tejido blando
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Crear volumen
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        self.volume_renderer.AddVolume(volume)
        self.volume_renderer.ResetCamera()
        self.volume_vtk.GetRenderWindow().Render()

    def numpy_to_vtk_image(self, slice_data):
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(
            slice_data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_img.GetPointData().SetScalars(vtk_array)
        return vtk_img

    def setup_slice_view(self, renderer, image):
        mapper = vtkImageSliceMapper()
        mapper.SetInputData(image)
        
        actor = vtkImageSlice()
        actor.SetMapper(mapper)
        
        prop = actor.GetProperty()
        prop.SetColorWindow(self.window_width)
        prop.SetColorLevel(self.window_level)
        
        # Remove the inversion of the color scale
        prop.SetUseLookupTableScalarRange(True)
        
        renderer.AddViewProp(actor)
        renderer.ResetCamera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())