import sys
import os
import numpy as np
import cv2
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
        self.dims = (0,0,0)          # Dimensiones de la imagen
        self.borders = (0,0)         # Valores mínimo y máximo de la imagen
        self.dicom_dir = None        # Directorio de DICOM
        self.current_axial = 0       # Índice de la imagen axial
        self.current_sagittal = 0    # Índice de la imagen sagital
        self.current_coronal = 0     # Índice de la imagen coronal
        self.current_isosurface = 0  # Índice de la imagen isosuperficie
        self.spacing = None          # Espaciado de la imagen
        self.window_level = 400      # Valor inicial de nivel de ventana
        self.window_width = 1500     # Valor inicial de ancho de ventana
        
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
            # Verificar si el directorio contiene archivos DICOM válidos
            reader = vtk.vtkDICOMImageReader()
            reader.SetDirectoryName(folder)
            reader.Update()
            
            if reader.GetOutput().GetDimensions() == (0, 0, 0):
                raise ValueError("El directorio no contiene imágenes DICOM válidas.")
            
            # Obtener datos como matriz numpy 3D
            vtk_data = reader.GetOutput().GetPointData().GetScalars()
            self.dims = reader.GetOutput().GetDimensions()
            self.images = numpy_support.vtk_to_numpy(vtk_data).reshape(self.dims[2], self.dims[1], self.dims[0])
            
            # Guardar dimensiones y espaciado
            self.spacing = reader.GetOutput().GetSpacing()

            # Calcular y almacenar los valores máximo y mínimo de las imágenes
            self.borders = (self.images.min(), self.images.max())
            self.current_isosurface = sum(self.borders) // 2

            # Punto inicial de los sliders
            self.current_axial = self.dims[2] // 2
            self.current_sagittal = self.dims[0] // 2
            self.current_coronal = self.dims[1] // 2
            
            print(f"Dimensiones del volumen: {self.dims}")
            print(f"Espaciado: {self.spacing}")
            print(f"Valores mínimo y máximo: {self.borders}")
            
            self.create_vtk_widgets()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar DICOM:\n{str(e)}")
            print(f"Error al cargar DICOM: {str(e)}")

    def create_vtk_widgets(self):
        """Configura las visualizaciones para las 3 vistas y la vista 3D"""

        # Limpiar widgets anteriores
        for i in reversed(range(self.viz_layout.count())): 
            widget = self.viz_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Vista Axial
        axial_header = QWidget()
        axial_header_layout = QHBoxLayout()
        axial_header.setLayout(axial_header_layout)
        axial_header_layout.addWidget(QLabel("Axial View"))
        
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setRange(0, self.dims[2] - 1)
        self.axial_slider.setValue(self.current_axial)
        self.axial_slider.valueChanged.connect(self.update_axial_view)
        axial_header_layout.addWidget(self.axial_slider)
        
        self.viz_layout.addWidget(axial_header, 0, 0)
        
        self.axial_vtk_widget = QVTKRenderWindowInteractor()
        self.axial_renderer = vtk.vtkRenderer()
        self.axial_renderer.SetBackground(0, 0, 0)
        self.axial_vtk_widget.GetRenderWindow().AddRenderer(self.axial_renderer)
        axial_frame = QFrame()
        axial_frame.setLayout(QVBoxLayout())
        axial_frame.layout().addWidget(self.axial_vtk_widget)
        self.viz_layout.addWidget(axial_frame, 1, 0)

        # Vista Sagital
        sagittal_header = QWidget()
        sagittal_header_layout = QHBoxLayout()
        sagittal_header.setLayout(sagittal_header_layout)
        sagittal_header_layout.addWidget(QLabel("Sagittal View"))
        
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setRange(0, self.dims[0] - 1)
        self.sagittal_slider.setValue(self.current_sagittal)
        self.sagittal_slider.valueChanged.connect(self.update_sagittal_view)
        sagittal_header_layout.addWidget(self.sagittal_slider)
        
        self.viz_layout.addWidget(sagittal_header, 0, 1)
        
        self.sagittal_vtk_widget = QVTKRenderWindowInteractor()
        self.sagittal_renderer = vtk.vtkRenderer()
        self.sagittal_renderer.SetBackground(0, 0, 0)
        self.sagittal_vtk_widget.GetRenderWindow().AddRenderer(self.sagittal_renderer)
        sagittal_frame = QFrame()
        sagittal_frame.setLayout(QVBoxLayout())
        sagittal_frame.layout().addWidget(self.sagittal_vtk_widget)
        self.viz_layout.addWidget(sagittal_frame, 1, 1)

        # Vista Coronal
        coronal_header = QWidget()
        coronal_header_layout = QHBoxLayout()
        coronal_header.setLayout(coronal_header_layout)
        coronal_header_layout.addWidget(QLabel("Coronal View"))
        
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setRange(0, self.dims[1] - 1)
        self.coronal_slider.setValue(self.current_coronal)
        self.coronal_slider.valueChanged.connect(self.update_coronal_view)
        coronal_header_layout.addWidget(self.coronal_slider)
        
        self.viz_layout.addWidget(coronal_header, 2, 0)
        
        self.coronal_vtk_widget = QVTKRenderWindowInteractor()
        self.coronal_renderer = vtk.vtkRenderer()
        self.coronal_renderer.SetBackground(0, 0, 0)
        self.coronal_vtk_widget.GetRenderWindow().AddRenderer(self.coronal_renderer)
        coronal_frame = QFrame()
        coronal_frame.setLayout(QVBoxLayout())
        coronal_frame.layout().addWidget(self.coronal_vtk_widget)
        self.viz_layout.addWidget(coronal_frame, 3, 0)

        # Vista 3D
        volume_header = QWidget()
        volume_header_layout = QHBoxLayout()
        volume_header.setLayout(volume_header_layout)
        volume_header_layout.addWidget(QLabel("3D Volume"))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(self.borders[0], self.borders[1])
        self.volume_slider.setValue(self.borders[0])
        self.volume_slider.valueChanged.connect(self.update_3d_volume)
        volume_header_layout.addWidget(self.volume_slider)
        
        self.viz_layout.addWidget(volume_header, 2, 1)
        
        self.volume_vtk = QVTKRenderWindowInteractor()
        self.volume_renderer = vtk.vtkRenderer()
        self.volume_renderer.SetBackground(1, 1, 1)  # Color de Fondo
        self.volume_vtk.GetRenderWindow().AddRenderer(self.volume_renderer)
        volume_frame = QFrame()
        volume_frame.setLayout(QVBoxLayout())
        volume_frame.layout().addWidget(self.volume_vtk)
        self.viz_layout.addWidget(volume_frame, 3, 1)

        # Actualizar todas las vistas
        if self.images is not None:
            self.update_all_views()

        # Al final de create_vtk_widgets()
        self.axial_vtk_widget.Initialize()
        self.sagittal_vtk_widget.Initialize()
        self.coronal_vtk_widget.Initialize()
        self.volume_vtk.Initialize()

        # Configurar estilos de interacción
        style = vtkInteractorStyleImage()
        self.axial_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.sagittal_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.coronal_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.volume_vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def update_all_views(self):
        if self.images is None:
            return
            
        self.window_level = self.wl_slider.value()
        self.window_width = self.ww_slider.value()
        
        # Actualizar vistas 2D
        self.update_axial_view()
        self.update_sagittal_view()
        self.update_coronal_view()
        
        # Actualizar vista 3D
        self.update_3d_volume()

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
        property.SetColorWindow(2000)
        property.SetColorLevel(500)
        
        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.SetProperty(property)
        
        renderer.AddActor(actor)
        renderer.ResetCamera()
        vtk_widget.GetRenderWindow().Render()

    def update_axial_view(self):
        """Actualiza la vista axial con el slice actual"""
        self.current_axial = self.axial_slider.value()
        # Obtener slice axial
        axial_slice = self.images[self.current_axial, :, :]
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(axial_slice, self.axial_renderer, self.axial_vtk_widget)

    def update_sagittal_view(self):
        """Actualiza la vista sagital con el slice actual"""
        self.current_sagittal = self.sagittal_slider.value()
        # Obtener slice sagital (necesitamos transponer para orientación correcta)
        sagittal_slice = self.images[:, :, self.current_sagittal].T
        sagittal_slice = np.rot90(sagittal_slice, k=1, axes=(0, 1))
        sagittal_slice = cv2.resize(sagittal_slice, (512,512), interpolation=cv2.INTER_LANCZOS4)
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(sagittal_slice, self.sagittal_renderer, self.sagittal_vtk_widget)
        
    def update_coronal_view(self):
        """Actualiza la vista coronal con el slice actual"""
        self.current_coronal = self.coronal_slider.value()
        # Obtener slice coronal (necesitamos transponer para orientación correcta)
        coronal_slice = self.images[:, self.current_coronal, :].T
        coronal_slice = np.rot90(coronal_slice, k=1, axes=(0, 1))
        coronal_slice = cv2.resize(coronal_slice, (self.dims[0]-1,self.dims[1]-1), interpolation=cv2.INTER_LANCZOS4)
        
        # Convertir a imagen VTK y mostrar
        self.display_slice(coronal_slice, self.coronal_renderer, self.coronal_vtk_widget)

    def update_isosurface(self):
        """Devuelve una copia de self.images con valores mayores o iguales a self.current_isosurface"""
        if self.images is None:
            return None
        
        # Crear una copia de self.images
        isosurface_images = self.images.copy()
        
        # Filtrar valores menores a self.current_isosurface
        isosurface_images[isosurface_images < self.current_isosurface] = self.borders[0]
        
        return isosurface_images

    def update_3d_volume(self):
        """Actualiza el volumen 3D con la configuración actual"""
        # Desactivar el interactor antes de realizar cambios
        interactor = self.volume_vtk.GetRenderWindow().GetInteractor()
        if interactor:
            interactor.Disable()

        # Limpiar el renderer
        self.volume_renderer.RemoveAllViewProps()
        self.current_isosurface = self.volume_slider.value()
        isosurface_images = self.update_isosurface()
        
        # Convertir a VTK ImageData
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(isosurface_images.shape[2], isosurface_images.shape[1], isosurface_images.shape[0])
        vtk_img.SetSpacing(self.spacing)
        vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(
            isosurface_images.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
        vtk_img.GetPointData().SetScalars(vtk_array)
        
        # Configurar mapeador de volumen
        volume_mapper = vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_img)
        
        # Configurar propiedades del volumen
        volume_property = vtkVolumeProperty()
        volume_property.ShadeOn()
        
        # Funciones de transferencia (color ajustado a rojo)
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(min_val, 1.0, 0.0, 0.0)  # Rojo para valores bajos
        color_func.AddRGBPoint(max_val, 1.0, 0.0, 0.0)  # Rojo para valores altos
        
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(min_val, 0.7)  # Opacidad alta para valores bajos
        opacity_func.AddPoint(max_val, 0.0)  # Opacidad baja para valores altos
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Crear volumen
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        self.volume_renderer.AddVolume(volume)
        self.volume_renderer.ResetCamera()
        
        # Reactivar el interactor después de realizar cambios
        if interactor:
            interactor.Enable()
        
        # Renderizar la ventana
        self.volume_vtk.GetRenderWindow().Render()

    def closeEvent(self, event):
        """Maneja el cierre de la ventana de manera segura"""
        try:
            # Desactivar todos los interactores VTK primero
            widgets = [
                ('axial_vtk_widget', self.axial_vtk_widget),
                ('sagittal_vtk_widget', self.sagittal_vtk_widget),
                ('coronal_vtk_widget', self.coronal_vtk_widget),
                ('volume_vtk', self.volume_vtk)
            ]
            
            for name, widget in widgets:
                if widget is not None:
                    try:
                        # Verificar si el widget tiene un render window
                        if hasattr(widget, 'GetRenderWindow'):
                            rw = widget.GetRenderWindow()
                            
                            # Desactivar el interactor si existe
                            if rw and hasattr(rw, 'GetInteractor'):
                                iren = rw.GetInteractor()
                                if iren:
                                    if hasattr(iren, 'ExitCallback'):
                                        iren.ExitCallback()
                                    if hasattr(iren, 'TerminateApp'):
                                        iren.TerminateApp()
                            
                            # Finalizar la ventana de renderizado
                            if rw and hasattr(rw, 'Finalize'):
                                rw.Finalize()
                        
                        # Eliminar el widget de su layout
                        if widget.parent():
                            widget.parent().layout().removeWidget(widget)
                        
                        # Eliminar el widget
                        widget.deleteLater()
                        
                    except Exception as e:
                        print(f"Error al liberar {name}: {str(e)}")

            # Limpiar renderers
            renderers = [
                self.axial_renderer,
                self.sagittal_renderer,
                self.coronal_renderer,
                self.volume_renderer
            ]
            
            for renderer in renderers:
                if renderer:
                    try:
                        renderer.RemoveAllViewProps()
                    except:
                        pass
            
        except Exception as e:
            print(f"Error durante el cierre: {str(e)}")
        
        # Llamar al método base para manejar el cierre de la ventana
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())





