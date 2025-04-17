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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)
        
        # Configuración de pyqtgraph
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        
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

        load_btn = QPushButton("Load DICOM")
        load_btn.clicked.connect(self.load_dicom)
        barra1_layout.addWidget(load_btn)

        self.transform_rb = QRadioButton("Transform Coordinates")
        self.filter_rb = QRadioButton("Filter")
        self.resolution_rb = QRadioButton("Modify Resolution")
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

        # Contenedor principal
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Área de visualización (75%)
        self.viz_widget = QWidget()
        self.viz_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout = QGridLayout()
        self.viz_widget.setLayout(self.viz_layout)
        content_layout.addWidget(self.viz_widget, stretch=75)

        # Área de opciones (25%)
        self.options2_widget = QWidget()
        self.options2_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.options2_layout = QVBoxLayout()
        self.options2_widget.setLayout(self.options2_layout)
        self.options2_widget.setStyleSheet("background-color: #f0f0f0;")
        content_layout.addWidget(self.options2_widget, stretch=25)

        self.create_options_containers()
        self.create_visualization_widgets()
        
        # Conectar señales
        self.transform_rb.toggled.connect(lambda: self.show_options_container("transform"))
        self.filter_rb.toggled.connect(lambda: self.show_options_container("filter"))
        self.resolution_rb.toggled.connect(lambda: self.show_options_container("resolution"))
        
        # Mostrar opciones iniciales
        self.show_options_container("transform")

    def create_options_containers(self):
        """Crea los contenedores para cada tipo de opciones"""
        # Contenedor principal
        self.main_options_container = QWidget()
        main_options_layout = QVBoxLayout()
        self.main_options_container.setLayout(main_options_layout)
        
        # Contenedor de transformación
        self.transform_container = QWidget()
        transform_layout = QVBoxLayout()
        self.transform_container.setLayout(transform_layout)
        transform_layout.addWidget(QLabel("Transform Options:"))
        for opt in ["Rotation", "Translation", "Scale", "Shear"]:
            rb = QRadioButton(opt)
            self.transform_options.addButton(rb)
            transform_layout.addWidget(rb)
        main_options_layout.addWidget(self.transform_container)
        
        # Contenedor de filtrado
        self.filter_container = QWidget()
        filter_layout = QVBoxLayout()
        self.filter_container.setLayout(filter_layout)
        filter_layout.addWidget(QLabel("Filter Options:"))
        for opt in ["Frequency Domain", "Spatial Domain"]:
            rb = QRadioButton(opt)
            self.filter_options.addButton(rb)
            filter_layout.addWidget(rb)
        main_options_layout.addWidget(self.filter_container)
        
        # Contenedor de resolución
        self.resolution_container = QWidget()
        resolution_layout = QVBoxLayout()
        self.resolution_container.setLayout(resolution_layout)
        resolution_layout.addWidget(QLabel("Resolution Options:"))
        for opt in ["Spatial", "Radiometric", "Temporal"]:
            rb = QRadioButton(opt)
            self.resolution_options.addButton(rb)
            resolution_layout.addWidget(rb)
        main_options_layout.addWidget(self.resolution_container)
        
        # Añadir contenedor principal al layout de opciones
        self.options2_layout.addWidget(self.main_options_container)

    def show_options_container(self, option_type):
        """Muestra el contenedor de opciones correspondiente"""
        self.transform_container.setVisible(option_type == "transform")
        self.filter_container.setVisible(option_type == "filter")
        self.resolution_container.setVisible(option_type == "resolution")

    def create_visualization_widgets(self):
        """Crea los widgets de visualización (Matplotlib + PyQtGraph)"""
        # Limpiar layout si ya tiene widgets
        for i in reversed(range(self.viz_layout.count())): 
            self.viz_layout.itemAt(i).widget().setParent(None)

        # Configurar vistas
        views = [
            ("Axial View", 0, 0),
            ("Sagittal View", 0, 1), 
            ("Coronal View", 2, 0),
            ("3D Volume", 2, 1)
        ]
        
        for name, row, col in views:
            # Añadir título
            self.viz_layout.addWidget(QLabel(name), row, col)
            
            # Crear widget apropiado
            if name != "3D Volume":
                # Widget contenedor para Matplotlib + Slider
                container = QWidget()
                layout = QVBoxLayout()
                container.setLayout(layout)
                
                # Canvas de Matplotlib
                canvas = FigureCanvas(plt.Figure())
                layout.addWidget(canvas)
                
                # Slider
                slider = QSlider(Qt.Horizontal)
                slider.valueChanged.connect(self.update_slices)
                layout.addWidget(slider)
                
                # Guardar referencias
                if name == "Axial View":
                    self.axial_canvas = canvas
                    self.axial_slider = slider
                elif name == "Sagittal View":
                    self.sagittal_canvas = canvas
                    self.sagittal_slider = slider
                elif name == "Coronal View":
                    self.coronal_canvas = canvas
                    self.coronal_slider = slider
                
                self.viz_layout.addWidget(container, row+1, col)
            else:
                # Widget 3D con PyQtGraph
                self.volume_widget = gl.GLViewWidget()
                self.volume_widget.setCameraPosition(distance=200)
                self.viz_layout.addWidget(self.volume_widget, row+1, col)

    def load_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
            
        try:
            # Leer archivos DICOM
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(folder)
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            self.images = sitk.GetArrayFromImage(image)
            self.spacing = image.GetSpacing()
            self.origin = image.GetOrigin()
            
            # Inicializar slices
            self.current_slices = [
                self.images.shape[0] // 2,
                self.images.shape[1] // 2,
                self.images.shape[2] // 2
            ]
            
            # Configurar sliders
            self.axial_slider.setRange(0, self.images.shape[0] - 1)
            self.sagittal_slider.setRange(0, self.images.shape[1] - 1)
            self.coronal_slider.setRange(0, self.images.shape[2] - 1)
            
            self.update_all_views()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading DICOM:\n{str(e)}")

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
        """Actualiza las vistas 2D con Matplotlib"""
        slices = {
            'axial': self.images[self.current_slices[0], :, :],
            'sagittal': self.images[:, self.current_slices[1], :],
            'coronal': self.images[:, :, self.current_slices[2]]
        }
        
        vmin = self.window_level - self.window_width/2
        vmax = self.window_level + self.window_width/2
        
        # Axial
        self.axial_canvas.figure.clear()
        ax = self.axial_canvas.figure.add_subplot(111)
        ax.imshow(slices['axial'].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        self.axial_canvas.draw()
        
        # Sagittal
        self.sagittal_canvas.figure.clear()
        ax = self.sagittal_canvas.figure.add_subplot(111)
        ax.imshow(slices['sagittal'].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        self.sagittal_canvas.draw()
        
        # Coronal
        self.coronal_canvas.figure.clear()
        ax = self.coronal_canvas.figure.add_subplot(111)
        ax.imshow(slices['coronal'].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        self.coronal_canvas.draw()

    def update_3d_volume(self):
        """Versión corregida para visualización 3D"""
        self.volume_widget.clear()
        
        if self.images is None:
            return
            
        try:
            # 1. Preparar datos
            data = np.swapaxes(self.images, 0, 2).astype(np.float32)
            
            # 2. Aplicar window/level
            w_min = self.window_level - self.window_width/2
            w_max = self.window_level + self.window_width/2
            data = np.clip(data, w_min, w_max)
            data = (data - w_min) / (w_max - w_min)
            
            # 3. Crear volumen con parámetros optimizados
            vol = gl.GLVolumeItem(
                data,
                sliceDensity=3,
                smooth=True,
                glOptions='translucent'
            )
            vol.scale(*self.spacing)
            
            # 4. Configurar colormap directamente en el shader
            # Escala de grises para imágenes médicas
            cmap = np.array([
                [0.0, 0.0, 0.0, 0.0],  # Negro transparente
                [0.5, 0.5, 0.5, 0.5],  # Gris semitransparente
                [1.0, 1.0, 1.0, 1.0]   # Blanco sólido
            ])
            vol.setShader('volume')
            vol.shader()['colorMap'] = cmap
            
            self.volume_widget.addItem(vol)
            
            # 5. Ajuste automático de cámara
            max_dim = max(data.shape)
            self.volume_widget.setCameraPosition(
                distance=max_dim * 3,
                elevation=30,
                azimuth=45
            )
            
        except Exception as e:
            QMessageBox.warning(self, "3D Error", f"Error en visualización 3D:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())