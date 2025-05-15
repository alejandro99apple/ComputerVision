from scipy.linalg import toeplitz
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QSlider, QFileDialog, QMessageBox, QSizePolicy,
                            QFrame, QRadioButton, QButtonGroup, QDoubleSpinBox, QComboBox, QSpinBox, QTableWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QHeaderView
from scipy.signal import convolve2d
import vtk
from PyQt5.QtWidgets import QTableWidgetItem
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import (vtkVolume, vtkVolumeProperty)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DICOM 4-View Visualizer")
        self.setWindowIcon(QIcon('icono.ico'))
        self.setGeometry(150, 100, 1600, 900)

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

        # Imagenes transformadas
        self.transformed_axial = None
        self.transformed_sagittal = None
        self.transformed_coronal = None

        # Barra de estado
        self.status_bar = QLabel("Listo")
        self.status_bar.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #a7a09f;
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-weight: bold;
            }
        """)
        self.status_bar.hide()
        self.status_bar_timer = QTimer()
        self.status_bar_timer.setSingleShot(True)
        self.status_bar_timer.timeout.connect(self.hide_status_bar)
        
        self.create_ui()

        # Conectar señales después de que todos los widgets han sido creados
        self.resolution_options.buttonClicked.connect(self.update_resolution_controls)

    def set_save_and_reset_enabled(self, enabled):
        self.reset_btn.setEnabled(enabled)

    def create_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 0)  # Margen interior de 20 píxeles
        main_layout.setSpacing(10)  # Espaciado entre elementos
        main_widget.setLayout(main_layout)

        # Barra 1 - Botón de carga y opciones principales
        barra1 = QWidget()
        barra1_layout = QHBoxLayout()
        barra1_layout.setContentsMargins(0, 0, 0, 0)  # Eliminar márgenes
        barra1_layout.setSpacing(10)  # Espaciado fijo
        barra1.setLayout(barra1_layout)

        # Botón de carga
        load_btn = QPushButton("Cargar DICOM")
        load_btn.clicked.connect(self.load_dicom)
        barra1_layout.addWidget(load_btn)

        # Grupo de radio buttons para opciones principales
        self.transform_rb = QRadioButton("Transformación de Coordenadas")
        self.filter_rb = QRadioButton("Filtrado")
        self.resolution_rb = QRadioButton("Modificación de Resolución")
        self.spatial_improvement_rb = QRadioButton("Mejoramiento de Resolución Espacial")

        self.transform_rb.setChecked(True)

        barra1_layout.addWidget(self.transform_rb)
        barra1_layout.addWidget(self.filter_rb)
        barra1_layout.addWidget(self.resolution_rb)
        barra1_layout.addWidget(self.spatial_improvement_rb)

        barra1_layout.addStretch()  # Empuja los siguientes widgets a la derecha

        # Botón Reiniciar
        self.reset_btn = QPushButton("Reiniciar")
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_views)
        self.reset_btn.setCursor(QCursor(Qt.PointingHandCursor))  # Set cursor to hand pointer
        barra1_layout.addWidget(self.reset_btn)

        main_layout.addWidget(barra1)

        # Separador visual
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Contenedor principal para visualización y opciones
        content_widget = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)  # Eliminar márgenes
        content_layout.setSpacing(10)  # Espaciado fijo
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Barra de visualización (75% del ancho)
        self.viz_widget = QWidget()
        self.viz_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_layout = QGridLayout()
        self.viz_layout.setContentsMargins(0, 0, 10, 0)  # Eliminar márgenes
        self.viz_layout.setSpacing(10)  # Espaciado fijo
        self.viz_widget.setLayout(self.viz_layout)
        content_layout.addWidget(self.viz_widget, stretch=60)

        # Separador visual vertical
        vertical_separator = QFrame()
        vertical_separator.setFrameShape(QFrame.VLine)
        vertical_separator.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(vertical_separator)

        # Barra de opciones 2 (25% del ancho)
        self.options2_widget = QWidget()
        self.options2_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.options2_layout = QVBoxLayout()
        self.options2_layout.setContentsMargins(10, 0, 0, 0)  # Eliminar márgenes
        self.options2_layout.setSpacing(10)  # Espaciado fijo
        self.options2_widget.setLayout(self.options2_layout)
        self.options2_widget.setStyleSheet("background-color: #f0f0f0;")
        content_layout.addWidget(self.options2_widget, stretch=40)

        # Crear los contenedores de opciones
        self.create_options_containers()
        
        # Conectar señales
        self.transform_rb.toggled.connect(lambda: self.handle_option_change("transform"))
        self.filter_rb.toggled.connect(lambda: self.handle_option_change("filter"))
        self.resolution_rb.toggled.connect(lambda: self.handle_option_change("resolution"))
        self.spatial_improvement_rb.toggled.connect(lambda: self.handle_option_change("spatial_improvement"))
        
        # Mostrar opciones iniciales
        self.show_options_container("transform")
        

        # Crear widgets VTK
        self.create_vtk_widgets()

        # Eliminar tamaño fijo de la ventana principal
        self.setMinimumSize(800, 600)  # Tamaño mínimo para evitar deformaciones

        # Configurar barra de estado
        self.status_bar.setParent(self)
        self.status_bar.setGeometry(self.width() - 200, self.height() - 40, 180, 30)
        self.status_bar.raise_()
        self.resizeEvent(None)  # Adjust position on initialization

    def resizeEvent(self, event):
        """Adjust the position of the status bar on window resize."""
        self.status_bar.setGeometry(self.width() - 200, self.height() - 40, 180, 30)
        super().resizeEvent(event)

    def show_status_bar(self, message):
        """Show the status bar with a message."""
        self.status_bar.setText(message)
        self.status_bar.show()

    def hide_status_bar(self):
        """Hide the status bar."""
        self.status_bar.hide()

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
        
        # Crear widgets para cada tipo de transformación
        self.create_transform_inputs(transform_layout)
        
        main_options_layout.addWidget(self.transform_container, alignment=Qt.AlignTop)
        self.transform_container.hide()

        # Contenedor para filtros
        self.filter_container = QWidget()
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(10)  # Espaciado entre grupos
        self.filter_container.setLayout(filter_layout)

        # Contenedor horizontal para el título y el icono de información de filtrado
        filter_title_container = QWidget()
        filter_title_layout = QHBoxLayout()
        filter_title_layout.setContentsMargins(0, 0, 0, 0)
        filter_title_layout.setSpacing(0)  # Sin espacio entre el texto y el ícono
        filter_title_container.setLayout(filter_title_layout)

        filter_title = QLabel("Opciones de Filtrado:")
        filter_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        filter_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        filter_title_layout.addWidget(filter_title)

        # Crear el icono de información tipo tooltip con SVG para filtrado
        filter_info_icon = QLabel()
        filter_pixmap = QPixmap("icon.svg").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        filter_info_icon.setPixmap(filter_pixmap)
        filter_info_icon.setCursor(QCursor(Qt.PointingHandCursor))
        filter_info_icon.setStyleSheet("margin-bottom: 5px;")  # Espacio entre el título y el icono
        filter_info_icon.setToolTip("Permite aplicar filtros espaciales y frecuenciales a las imágenes " \
                                    "para mejorar la visualización o resaltar características específicas. " \
                                    "Puedes seleccionar entre diferentes tipos de filtros pasa bajas y pasa altas, " \
                                    "así como ajustar parámetros como el tipo de ventana, el tamaño del kernel o el " \
                                    "factor de ruido/borde. Los filtros ayudan a reducir el ruido, suavizar la imagen " \
                                    "o resaltar bordes y detalles importantes.")
        filter_title_layout.addWidget(filter_info_icon)
        filter_title_layout.addStretch()

        filter_layout.addWidget(filter_title_container)

        # Grupo de botones para dominios
        self.filter_options = QButtonGroup()
        self.create_frequency_domain_filter_group(filter_layout)
        self.create_spatial_domain_filter_group(filter_layout)


        # Botón APLICAR para ejecutar ambas funciones de filtrado
        self.apply_filter_btn = QPushButton("APLICAR")
        self.apply_filter_btn.setCursor(QCursor(Qt.PointingHandCursor))  # Cambiar cursor a mano
        self.apply_filter_btn.setFixedSize(150, 30)
        self.apply_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;

            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.apply_filter_btn.clicked.connect(self.apply_filters)  # Conectar señal
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_filter_btn)
        button_layout.addStretch()
        filter_layout.addLayout(button_layout)

        # Añadir al contenedor principal
        self.main_options_container.layout().addWidget(self.filter_container, alignment=Qt.AlignTop)
        self.filter_container.hide()

        # Conectar señales para habilitar/deshabilitar inputs
        self.filter_options.buttonClicked.connect(self.update_filter_controls)
        self.update_filter_controls()  # Actualizar inputs según el botón seleccionado por defecto

        # Contenedor para resolución (definimos el layout primero)
        self.resolution_container = QWidget()
        resolution_layout = QVBoxLayout()
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(10)  # Más espacio entre grupos
        self.resolution_container.setLayout(resolution_layout)
        
        # Contenedor horizontal para el título y el icono de información de resolución
        resolution_title_container = QWidget()
        resolution_title_layout = QHBoxLayout()
        resolution_title_layout.setContentsMargins(0, 0, 0, 0)
        resolution_title_layout.setSpacing(0)  # Sin espacio entre el texto y el ícono
        resolution_title_container.setLayout(resolution_title_layout)

        resolution_title = QLabel("Opciones de Modificación de Resolución:")
        resolution_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        resolution_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        resolution_title_layout.addWidget(resolution_title)

        # Crear el icono de información tipo tooltip con SVG para resolución
        resolution_info_icon = QLabel()
        resolution_pixmap = QPixmap("icon.svg").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        resolution_info_icon.setPixmap(resolution_pixmap)
        resolution_info_icon.setCursor(QCursor(Qt.PointingHandCursor))
        resolution_info_icon.setStyleSheet("margin-bottom: 5px;")  # Espacio entre el título y el icono
        resolution_info_icon.setToolTip("Permite modificar la resolución espacial, radiométrica o temporal de las imágenes. " \
                                        "Puedes realizar submuestreo o sobremuestreo para cambiar la cantidad de píxeles (resolución " \
                                        "espacial), ajustar la cantidad de niveles de gris (resolución radiométrica) o aplicar desenfoque " \
                                        "por movimiento (resolución temporal). Estas opciones ayudan a analizar el efecto de la resolución " \
                                        "en la calidad y el detalle de las imágenes.")
        resolution_title_layout.addWidget(resolution_info_icon)
        resolution_title_layout.addStretch()

        resolution_layout.addWidget(resolution_title_container)

        # Grupo de opciones de resolución con RadioButtons
        self.resolution_options = QButtonGroup()
        
        # Crear los grupos de opciones de resolución con estilo similar a transformación
        self.create_spatial_resolution_group(resolution_layout)
        self.create_radiometric_resolution_group(resolution_layout)
        self.create_temporal_resolution_group(resolution_layout)
        
        # Configurar el RadioButton de resolución espacial como seleccionado por defecto
        if self.resolution_options.buttons():
            self.resolution_options.buttons()[0].setChecked(True)
            self.update_resolution_controls()
        
        # Botón APLICAR centrado (igual al de transformación)
        self.apply_resolution_btn = QPushButton("APLICAR")
        self.apply_resolution_btn.setCursor(QCursor(Qt.PointingHandCursor))  # Cambiar cursor a mano
        self.apply_resolution_btn.setFixedSize(150, 30)
        self.apply_resolution_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Contenedor para centrar el botón
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_resolution_btn)
        button_layout.addStretch()
        button_container.setLayout(button_layout)
        
        resolution_layout.addWidget(button_container)
        resolution_layout.addStretch()
        
        main_options_layout.addWidget(self.resolution_container, alignment=Qt.AlignTop)
        self.resolution_container.hide()

        # Conectar señal del botón
        self.apply_resolution_btn.clicked.connect(self.apply_resolution_changes)


        # Contenedor para mejorar espacialmente (sin título)
        self.spatial_improvement_container = QWidget()
        spatial_layout = QVBoxLayout()
        spatial_layout.setContentsMargins(0, 0, 0, 0)
        spatial_layout.setSpacing(8)  # Un poco más de espacio entre elementos
        self.spatial_improvement_container.setLayout(spatial_layout)

        # Contenedor horizontal para el título y el icono de información de mejoramiento espacial
        spatial_title_container = QWidget()
        spatial_title_layout = QHBoxLayout()
        spatial_title_layout.setContentsMargins(0, 0, 0, 0)
        spatial_title_layout.setSpacing(0)  # Sin espacio entre el texto y el ícono
        spatial_title_container.setLayout(spatial_title_layout)

        spatial_title = QLabel("Opciones de Mejoramiento de Resolución Espacial:")
        spatial_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        spatial_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        spatial_title_layout.addWidget(spatial_title)

        # Crear el icono de información tipo tooltip con SVG para mejoramiento espacial
        spatial_info_icon = QLabel()
        spatial_pixmap = QPixmap("icon.svg").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        spatial_info_icon.setPixmap(spatial_pixmap)
        spatial_info_icon.setCursor(QCursor(Qt.PointingHandCursor))
        spatial_info_icon.setStyleSheet("margin-bottom: 5px;")  # Espacio entre el título y el icono
        spatial_info_icon.setToolTip("Permite restaurar o mejorar la calidad de las imágenes degradadas mediante " \
                                    "algoritmos avanzados de reconstrucción, como CLS (Constrained Least Squares), " \
                                    "WCLS (Weighted CLS) y BMR (Bayesian Mean Restoration). Estos métodos ayudan a reducir " \
                                    "el desenfoque y el ruido, recuperando detalles y mejorando la visualización. Puedes " \
                                    "comparar los resultados de cada método y visualizar métricas objetivas de calidad (PSNR, " \
                                    "IOSNR, MAE, SSIM) para cada vista.")
        spatial_title_layout.addWidget(spatial_info_icon)
        spatial_title_layout.addStretch()

        spatial_layout.addWidget(spatial_title_container)

        # Input: Ancho de la función de dispersión
        dispersion_label = QLabel("Ancho de la función de dispersión (pixeles):")
        self.dispersion_width_input = QDoubleSpinBox()
        self.dispersion_width_input.setRange(10.0, 100.0)
        self.dispersion_width_input.setSingleStep(10.0)
        self.dispersion_width_input.setValue(5.0)
        self.dispersion_width_input.setStyleSheet("""
            QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
        """)
        spatial_layout.addWidget(dispersion_label)
        spatial_layout.addWidget(self.dispersion_width_input)

        # Botón APLICAR (centrado)
        self.apply_button = QPushButton("APLICAR")
        self.apply_button.setCursor(QCursor(Qt.PointingHandCursor))  # Cambiar cursor a mano
        self.apply_button.setFixedSize(120, 30)
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

            # Conectar el botón APLICAR de mejoramiento espacial
        self.apply_button.clicked.connect(self.apply_spatial_improvement)
        
        # Contenedor para centrar el botón
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_container.setLayout(button_layout)
        
        spatial_layout.addWidget(button_container)
        
        # Selector: Visualización
        visualization_label = QLabel("Visualización:")
        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems(["CLS", "WCLS", "BMR", "Imagen Degradada", "Imagen Original"])
        self.visualization_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                min-width: 120px;
            }
        """)
        self.visualization_combo.setEnabled(False)  # Desactivado por defecto
        spatial_layout.addWidget(visualization_label)
        spatial_layout.addWidget(self.visualization_combo)

        # Tabla de métricas
        self.metrics_table = QTableWidget(12, 3)  # Cambiar a 3 columnas
        self.metrics_table.setHorizontalHeaderLabels(["CLS", "WCLS", "BMR"])
        self.metrics_table.setVerticalHeaderLabels([
            "PSNR Axial (dB)", "IOSNR Axial (dB)", "MAE Axial", "SSIM Axial",
            "PSNR Sagital (dB)", "IOSNR Sagital (dB)", "MAE Sagital", "SSIM Sagital",
            "PSNR Coronal (dB)", "IOSNR Coronal (dB)", "MAE Coronal", "SSIM Coronal", ""
        ])
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Deshabilitar edición
        self.metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # Expandir verticalmente
        self.metrics_table.setFixedHeight(self.metrics_table.verticalHeader().length() + self.metrics_table.horizontalHeader().height())
        self.metrics_table.horizontalHeader().setStretchLastSection(True)  # Ajustar última columna
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Expandir columnas
        self.metrics_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Expandir filas

        # Aplicar estilo y margen superior
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                margin-top: 20px;  /* Margen superior */
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                gridline-color: #ddd;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                font-weight: bold;
                border: 1px solid #ddd;
                padding: 4px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #d0e7ff;
                color: #000;
            }
        """)

        # Aplicar estilo a la celda superior izquierda de los encabezados
        self.metrics_table.setCornerButtonEnabled(False)  # Deshabilitar el botón de esquina
        corner_widget = self.metrics_table.findChild(QHeaderView, "")
        if corner_widget:
            corner_widget.setStyleSheet("""
                QHeaderView::section {
                    background-color: #f0f0f0;
                    font-weight: bold;
                    border: 1px solid #ddd;
                }
            """)

        spatial_layout.addWidget(self.metrics_table)

        # Añadir al contenedor principal
        main_options_layout.addWidget(self.spatial_improvement_container, alignment=Qt.AlignTop)
        self.spatial_improvement_container.hide()  # Ocultar inicialmente

        # Contenedor para transformaciones (el resto se mantiene igual)
        self.transform_container = QWidget()
        transform_layout = QVBoxLayout()
        transform_layout.setContentsMargins(0, 0, 0, 0)
        transform_layout.setSpacing(5)
        self.transform_container.setLayout(transform_layout)

        # Contenedor horizontal para el título y el icono de información
        transform_title_container = QWidget()
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        transform_title_container.setLayout(title_layout)

        transform_title = QLabel("Opciones de Transformación de Coordenadas:")
        transform_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        transform_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        title_layout.addWidget(transform_title)

        # Crear el icono de información tipo tooltip con SVG
        info_icon = QLabel()
        pixmap = QPixmap("icon.svg").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        info_icon.setPixmap(pixmap)
        info_icon.setStyleSheet("margin-bottom: 5px;")  # Espacio entre el título y el icono
        info_icon.setCursor(QCursor(Qt.PointingHandCursor))
        info_icon.setToolTip("Permite aplicar transformaciones geométricas a las imágenes, como " \
                            "rotación, traslación, escalamiento e inclinación. Estas operaciones modifican la posición, " \
                            "orientación o tamaño de las imágenes para facilitar su análisis, comparación o alineación. " \
                            "Puedes ajustar los parámetros de cada transformación y visualizar el resultado en tiempo " \
                            "real en las diferentes vistas.")
        title_layout.addWidget(info_icon)
        title_layout.addStretch()

        transform_layout.addWidget(transform_title_container)

        self.create_transform_inputs(transform_layout)

        main_options_layout.addWidget(self.transform_container, alignment=Qt.AlignTop)
        self.transform_container.hide()

        # Añadir stretch para empujar todo hacia arriba
        main_options_layout.addStretch()

        # Añadir el contenedor principal al layout de opciones 2
        self.options2_layout.addWidget(self.main_options_container)
        self.options2_layout.addStretch()

        # Conectar el ComboBox de visualización al método de instancia
        self.visualization_combo.currentTextChanged.connect(lambda metodo: self._actualizar_vistas_wrapper(metodo))

    def _actualizar_vistas_wrapper(self, metodo):
        # Llama a la función interna actualizar_vistas definida en apply_spatial_improvement
        if hasattr(self, '_actualizar_vistas_func'):
            self._actualizar_vistas_func(metodo)



        # Mostrar los resultados actualizados
        self.display_transformed_slices()

    def create_frequency_domain_filter_group(self, parent_layout):
        """Grupo para Filtrado en el Dominio Frecuencial"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)

        # RadioButton
        self.frequency_domain_rb = QRadioButton("Dominio Frecuencial")
        self.frequency_domain_rb.setStyleSheet("font-weight: bold;")
        self.frequency_domain_rb.setChecked(True)  # Seleccionar por defecto
        self.filter_options.addButton(self.frequency_domain_rb)
        layout.addWidget(self.frequency_domain_rb)

        # Primera fila: Tipo y Factor de borde
        row1_layout = QHBoxLayout()
        self.frequency_type_combo = QComboBox()
        self.frequency_type_combo.addItems(["Pasa Bajas", "Pasa Altas"])
        self.frequency_type_combo.currentTextChanged.connect(self.update_frequency_input_label)  # Conectar señal
        row1_layout.addWidget(QLabel("Tipo:"))
        row1_layout.addWidget(self.frequency_type_combo)

        self.frequency_input_label = QLabel("Factor de ruido:")  # Etiqueta dinámica
        self.frequency_input = QDoubleSpinBox()
        self.frequency_input.setRange(0.5, 5.0)
        self.frequency_input.setValue(1.0)
        self.frequency_input.setSingleStep(0.1)
        self.frequency_input.setEnabled(False)
        row1_layout.addWidget(self.frequency_input_label)
        row1_layout.addWidget(self.frequency_input)

        layout.addLayout(row1_layout)

        # Segunda fila: Ventana y Dimensión del radio
        row2_layout = QHBoxLayout()
        self.frequency_window_combo = QComboBox()
        self.frequency_window_combo.addItems(["Gaussiana","Gaussiana Mod", "Coseno", "Barlett", "Hanning"])
        self.frequency_window_combo.setEnabled(False)
        row2_layout.addWidget(QLabel("Ventana:"))
        row2_layout.addWidget(self.frequency_window_combo)

        self.frequency_radius_dimension_input = QDoubleSpinBox()
        self.frequency_radius_dimension_input.setRange(0.1, 0.9)
        self.frequency_radius_dimension_input.setValue(0.1)
        self.frequency_radius_dimension_input.setSingleStep(0.1)
        self.frequency_radius_dimension_input.setEnabled(False)
        row2_layout.addWidget(QLabel("Dimensión del radio:"))
        row2_layout.addWidget(self.frequency_radius_dimension_input)

        layout.addLayout(row2_layout)

        parent_layout.addWidget(container)

    def update_frequency_input_label(self):
        """Actualiza la etiqueta del input según el tipo de filtro seleccionado"""
        if self.frequency_type_combo.currentText() == "Pasa Bajas":
            self.frequency_input_label.setText("Factor de ruido:")
            self.frequency_input.setRange(0.5, 5.0)
            self.frequency_input.setValue(1.0)
        else:
            self.frequency_input_label.setText("Factor de borde:")
            self.frequency_input.setRange(0.1, 5.0)
            self.frequency_input.setValue(1.0)

    def create_spatial_domain_filter_group(self, parent_layout):
        """Grupo para Filtrado en el Dominio Espacial"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)

        # RadioButton
        self.spatial_domain_rb = QRadioButton("Dominio Espacial")
        self.spatial_domain_rb.setStyleSheet("font-weight: bold;")
        self.filter_options.addButton(self.spatial_domain_rb)
        layout.addWidget(self.spatial_domain_rb)

        # Fila: Tipo y Factor de ruido
        row_layout = QHBoxLayout()
        self.spatial_type_combo = QComboBox()
        self.spatial_type_combo.addItems(["Pasa Bajas", "Pasa Altas"])
        self.spatial_type_combo.currentTextChanged.connect(self.update_spatial_input_label)  # Conectar señal
        row_layout.addWidget(QLabel("Tipo:"))
        row_layout.addWidget(self.spatial_type_combo)

        self.spatial_input_label = QLabel("Factor de ruido:")  # Etiqueta dinámica
        self.spatial_input = QDoubleSpinBox()
        self.spatial_input.setRange(0.5, 2.0)
        self.spatial_input.setValue(1.0)
        self.spatial_input.setSingleStep(0.1)
        self.spatial_input.setEnabled(False)
        row_layout.addWidget(self.spatial_input_label)
        row_layout.addWidget(self.spatial_input)

        layout.addLayout(row_layout)

        # Fila 2: Ventana y Tamaño del Kernel
        row2_layout = QHBoxLayout()
        self.spatial_window_combo = QComboBox()
        self.spatial_window_combo.addItems(["Media"])  # Inicialmente solo "Media"
        row2_layout.addWidget(QLabel("Ventana:"))
        row2_layout.addWidget(self.spatial_window_combo)

        self.kernel_size_input = QSpinBox()
        self.kernel_size_input.setRange(3, 7)
        self.kernel_size_input.setValue(3)
        self.kernel_size_input.setSingleStep(2)  # Solo valores impares
        self.kernel_size_input.setEnabled(False)  # Solo habilitado para Pasa Bajas
        row2_layout.addWidget(QLabel("Tamaño del Kernel:"))
        row2_layout.addWidget(self.kernel_size_input)

        layout.addLayout(row2_layout)

        parent_layout.addWidget(container)

    def update_spatial_input_label(self):
        """Actualiza la etiqueta del input y las opciones según el tipo de filtro seleccionado"""
        if self.spatial_type_combo.currentText() == "Pasa Bajas":
            self.spatial_input_label.setText("Factor de ruido:")
            self.spatial_window_combo.clear()
            self.spatial_window_combo.addItems(["Media"])
            self.kernel_size_input.setEnabled(True)
        else:
            self.spatial_input_label.setText("Factor de borde:")
            self.spatial_window_combo.clear()
            self.spatial_window_combo.addItems([
                # Sobel (8 direcciones)
                "Sobel-N", "Sobel-NE", "Sobel-E", "Sobel-SE",
                "Sobel-S", "Sobel-SW", "Sobel-W", "Sobel-NW",
                
                # Prewitt (8 direcciones)
                "Prewitt-N", "Prewitt-NE", "Prewitt-E", "Prewitt-SE",
                "Prewitt-S", "Prewitt-SW", "Prewitt-W", "Prewitt-NW",
                
                # Laplace (3 variantes)
                "Laplace", "Laplace-8", "Laplace-D"
            ])
            self.kernel_size_input.setEnabled(False)

    def update_filter_controls(self):
        """Habilita solo los controles correspondientes al RadioButton seleccionado"""
        is_frequency_selected = self.frequency_domain_rb.isChecked()
        is_spatial_selected = self.spatial_domain_rb.isChecked()

        # Habilitar/deshabilitar inputs del Dominio Frecuencial
        self.frequency_type_combo.setEnabled(is_frequency_selected)
        self.frequency_input.setEnabled(is_frequency_selected)
        # Habilitar/deshabilitar inputs del Dominio Frecuencial
        self.frequency_type_combo.setEnabled(is_frequency_selected)
        self.frequency_input.setEnabled(is_frequency_selected)
        self.frequency_window_combo.setEnabled(is_frequency_selected)
        self.frequency_radius_dimension_input.setEnabled(is_frequency_selected)

        # Habilitar/deshabilitar inputs del Dominio Espacial
        self.spatial_type_combo.setEnabled(is_spatial_selected)
        self.spatial_input.setEnabled(is_spatial_selected)
        self.spatial_window_combo.setEnabled(is_spatial_selected)
        self.kernel_size_input.setEnabled(is_spatial_selected and self.spatial_type_combo.currentText() == "Pasa Bajas")

    def create_transform_inputs(self, transform_layout):
        """Crea los grupos de opciones con estilo de secciones diferenciadas"""
        # Contenedor principal
        self.transform_options_container = QWidget()
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 10)
        options_layout.setSpacing(10)
        self.transform_options_container.setLayout(options_layout)
        transform_layout.addWidget(self.transform_options_container)

        # Crear grupos con estilo
        self.create_rotation_option(options_layout)
        self.create_translation_option(options_layout)
        self.create_scaling_option(options_layout)
        self.create_shearing_option(options_layout)



        # Botón APLICAR
        self.apply_button = QPushButton("APLICAR")
        self.apply_button.setFixedSize(150, 30)
        self.apply_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
         # Conectar el botón APLICAR a la función apply_transform
        self.apply_button.clicked.connect(self.apply_transform)

        # Contenedor para centrar el botón
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_container.setLayout(button_layout)
        
        options_layout.addWidget(button_container)

        # Selección inicial - Asegurar que Rotación esté seleccionada
        self.rotation_rb.setChecked(True)  # Seleccionar explícitamente "Rotación"
        self.update_transform_inputs()  # Actualizar los inputs para reflejar la selección inicial

    def create_rotation_option(self, parent_layout):
        """Crea el grupo para Rotación con estilo de sección"""
        # Frame contenedor
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Radio button
        self.rotation_rb = QRadioButton("Rotación")
        self.rotation_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.transform_options.addButton(self.rotation_rb)
        layout.addWidget(self.rotation_rb)
        
        # Input
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(-360, 360)
        self.angle_input.setSingleStep(1)
        self.angle_input.setValue(0)
        self.angle_input.setDecimals(0)  # 0 decimales de precisión

        layout.addWidget(QLabel("Ángulo (grados):"))
        layout.addWidget(self.angle_input)
        
        parent_layout.addWidget(container)
        self.rotation_rb.toggled.connect(self.update_transform_inputs)
        
    def create_translation_option(self, parent_layout):
        """Crea el grupo para Traslación con estilo de sección"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.translation_rb = QRadioButton("Traslación")
        self.translation_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.transform_options.addButton(self.translation_rb)
        layout.addWidget(self.translation_rb)
        
        # Inputs
        self.h_translation_input = QDoubleSpinBox()
        self.v_translation_input = QDoubleSpinBox()
        for inp in [self.h_translation_input, self.v_translation_input]:
            inp.setRange(-1000, 1000)
            inp.setSingleStep(1)
            inp.setValue(0)
            inp.setDecimals(0)  # 0 decimales de precisión
            
        layout.addWidget(QLabel("Factor Horizontal (pixeles):"))
        layout.addWidget(self.h_translation_input)
        layout.addWidget(QLabel("Factor Vertical (pixeles):"))
        layout.addWidget(self.v_translation_input)
        
        parent_layout.addWidget(container)
        self.translation_rb.toggled.connect(self.update_transform_inputs)

    def create_scaling_option(self, parent_layout):
        """Crea el grupo para Escalamiento con valores iniciales"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.scaling_rb = QRadioButton("Escalamiento")
        self.scaling_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.transform_options.addButton(self.scaling_rb)
        layout.addWidget(self.scaling_rb)
        
        # Inputs con valores iniciales
        self.h_scaling_input = QDoubleSpinBox()
        self.v_scaling_input = QDoubleSpinBox()
        
        # Configuración común
        for inp in [self.h_scaling_input, self.v_scaling_input]:
            inp.setRange(0.1, 3.0)  # Rango de escalamiento (10% a 300%)
            inp.setSingleStep(0.1)
            inp.setValue(1.0)  # Valor inicial = sin escalamiento
            inp.setDecimals(1)  # 1 decimales de precisión
        
        layout.addWidget(QLabel("Factor Horizontal (relación de aspecto):"))
        layout.addWidget(self.h_scaling_input)
        layout.addWidget(QLabel("Factor Vertical (relación de aspecto):"))
        layout.addWidget(self.v_scaling_input)
        
        parent_layout.addWidget(container)
        self.scaling_rb.toggled.connect(self.update_transform_inputs)

    def create_shearing_option(self, parent_layout):
        """Crea el grupo para Inclinación con valores iniciales"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.shearing_rb = QRadioButton("Inclinación")
        self.shearing_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.transform_options.addButton(self.shearing_rb)
        layout.addWidget(self.shearing_rb)
        
        # Inputs con valores iniciales
        self.h_shearing_input = QDoubleSpinBox()
        self.v_shearing_input = QDoubleSpinBox()
        
        # Configuración común
        for inp in [self.h_shearing_input, self.v_shearing_input]:
            inp.setRange(-30, 30)  # Rango razonable para inclinación
            inp.setSingleStep(1)   # Pasos de 0.05
            inp.setValue(0.0)         # Valor inicial = sin inclinación
            inp.setDecimals(0)        #  decimales de precisión
        
        layout.addWidget(QLabel("Factor Horizontal (grados):"))
        layout.addWidget(self.h_shearing_input)
        layout.addWidget(QLabel("Factor Vertical (grados):"))
        layout.addWidget(self.v_shearing_input)
        
        parent_layout.addWidget(container)
        self.shearing_rb.toggled.connect(self.update_transform_inputs)

    def update_transform_inputs(self):
        """Habilita solo los inputs correspondientes a la transformación seleccionada"""
        # Verificar que todos los atributos necesarios existan
        if not all(hasattr(self, attr) for attr in ['angle_input', 'h_translation_input', 
                                                'v_translation_input', 'h_scaling_input',
                                                'v_scaling_input', 'h_shearing_input',
                                                'v_shearing_input']):
            return
            
        checked_button = self.transform_options.checkedButton()
        if checked_button:
            text = checked_button.text()
            
            # Habilitar/deshabilitar todos los inputs
            all_inputs = [
                self.angle_input,
                self.h_translation_input, self.v_translation_input,
                self.h_scaling_input, self.v_scaling_input,
                self.h_shearing_input, self.v_shearing_input
            ]
            
            # Primero deshabilitar todos
            for inp in all_inputs:
                inp.setEnabled(False)
            
            # Luego habilitar solo los correspondientes
            if text == "Rotación":
                self.angle_input.setEnabled(True)
            elif text == "Traslación":
                self.h_translation_input.setEnabled(True)
                self.v_translation_input.setEnabled(True)
            elif text == "Escalamiento":
                self.h_scaling_input.setEnabled(True)
                self.v_scaling_input.setEnabled(True)
            elif text == "Inclinación":
                self.h_shearing_input.setEnabled(True)
                self.v_shearing_input.setEnabled(True)

    def show_transform_inputs(self, transform_type):
        """Muestra solo los inputs del tipo de transformación especificado"""
        if hasattr(self, 'rotation_container') and self.rotation_container:
            self.rotation_container.setVisible(transform_type == "rotation")
        if hasattr(self, 'translation_container') and self.translation_container:
            self.translation_container.setVisible(transform_type == "translation")
        if hasattr(self, 'scaling_container') and self.scaling_container:
            self.scaling_container.setVisible(transform_type == "scaling")
        if hasattr(self, 'shearing_container') and self.shearing_container:
            self.shearing_container.setVisible(transform_type == "shearing")

    def show_options_container(self, option_type):
        """Muestra el contenedor de opciones correspondiente"""
        self.transform_container.setVisible(option_type == "transform")
        self.filter_container.setVisible(option_type == "filter")
        self.resolution_container.setVisible(option_type == "resolution")
        self.spatial_improvement_container.setVisible(option_type == "spatial_improvement")
        
        # Si es transformación, asegurar que se muestren los inputs correctos
        if option_type == "transform":
            self.update_transform_inputs()
        if option_type == "resolution":
            self.update_resolution_controls()

    def handle_option_change(self, option_type):
        """Maneja el cambio de opción en la barra de opciones 1."""
        self.reset_views()
        self.show_options_container(option_type)

    def load_dicom(self):
        self.show_status_bar("CARGANDO...")
        QApplication.processEvents()  # Ensure UI updates immediately
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            self.hide_status_bar()
            return

        try:
            # Suppress VTK error messages
            vtk.vtkObject.GlobalWarningDisplayOff()

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

            # Actualizar los valores de inputs de traslación
            self.h_translation_input.setRange(-self.dims[0], self.dims[0])
            self.v_translation_input.setRange(-self.dims[1], self.dims[1])

            self.create_vtk_widgets()

            # Al cargar imágenes, habilitar Guardar y Reiniciar
            self.set_save_and_reset_enabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar DICOM:\n{str(e)}")
            print(f"Error al cargar DICOM: {str(e)}")
        finally:
            self.hide_status_bar()

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
        axial_header_layout.addWidget(QLabel("Vista Axial"))
        
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
        sagittal_header_layout.addWidget(QLabel("Vista Sagital"))
        
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
        coronal_header_layout.addWidget(QLabel("Vista Coronal"))
        
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
        volume_header_layout.addWidget(QLabel("Vista 3D"))

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

        # Configurar estilos de interacción (cada vista 2D debe tener su propio objeto)
        style_axial = vtkInteractorStyleImage()
        style_sagittal = vtkInteractorStyleImage()
        style_coronal = vtkInteractorStyleImage()
        self.axial_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style_axial)
        self.sagittal_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style_sagittal)
        self.coronal_vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style_coronal)
        self.volume_vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.set_fixed_vtk_widget_sizes()

    def set_fixed_vtk_widget_sizes(self):
        """Establece tamaños fijos para las vistas de visualización"""
        fixed_width = 450  # Ancho fijo en píxeles (más ancho para hacerlo rectangular)
        fixed_height = 300  # Alto fijo en píxeles

        # Configurar tamaños fijos para cada widget VTK
        self.axial_vtk_widget.setFixedSize(fixed_width, fixed_height)
        self.sagittal_vtk_widget.setFixedSize(fixed_width, fixed_height)
        self.coronal_vtk_widget.setFixedSize(fixed_width, fixed_height)
        self.volume_vtk.setFixedSize(fixed_width, fixed_height)

        # Asegurar que los layouts no ajusten automáticamente los tamaños
        self.viz_layout.setSizeConstraint(QGridLayout.SetFixedSize)

    def reset_views(self):
        self.transformed_axial = self.current_axial
        self.transformed_sagittal = self.current_sagittal
        self.transformed_coronal = self.current_coronal

        # Vaciar directamente la tabla de métricas
        for row in range(self.metrics_table.rowCount()):
            for col in range(self.metrics_table.columnCount()):
                self.metrics_table.setItem(row, col, None)

        self.update_all_views()

        # Al reiniciar, deshabilitar Guardar y Reiniciar si no hay imágenes
        if self.images is None:
            self.set_save_and_reset_enabled(False)

    def update_all_views(self):
        if self.images is None:
            return
            
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
        self.current_sagittal = self.sagittal_slider.value()
        self.transformed_sagittal = None  # Reset al cambiar slice
        # Mostramos la imagen con transformaciones iniciales
        sagittal_slice = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        self.display_slice(sagittal_slice, self.sagittal_renderer, self.sagittal_vtk_widget)

    def update_coronal_view(self):
        self.current_coronal = self.coronal_slider.value()
        self.transformed_coronal = None  # Reset al cambiar slice
        # Mostramos la imagen con transformaciones iniciales
        coronal_slice = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])
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

    def prepare_sagittal_slice(self, slice_img):
        """Aplica las transformaciones iniciales a la vista sagital"""
        # Transponer y rotar
        slice_img = slice_img.T
        slice_img = np.rot90(slice_img, k=1, axes=(0, 1))
        # Redimensionar para mantener relación de aspecto
        slice_img = cv2.resize(slice_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        return slice_img

    def prepare_coronal_slice(self, slice_img):
        """Aplica las transformaciones iniciales a la vista coronal"""
        # Transponer y rotar
        slice_img = slice_img.T
        slice_img = np.rot90(slice_img, k=1, axes=(0, 1))
        # Redimensionar para mantener relación de aspecto
        slice_img = cv2.resize(slice_img, (self.dims[0]-1, self.dims[1]-1), interpolation=cv2.INTER_LANCZOS4)
        return slice_img

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

    #=====================================================================================================
    # Trasformación de Coordenadas

    def get_current_slices(self):
        """Obtiene los slices actualmente visibles en las vistas"""
        # Axial (no transformada previamente)
        current_axial = self.images[self.current_axial, :, :].copy()
        
        # Sagital y Coronal (ya transformadas)

        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])

        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])
        
        return current_axial, current_sagittal, current_coronal

    def apply_transform(self):
        self.show_status_bar("Cargando...")
        QApplication.processEvents()  # Ensure UI updates immediately
        """Aplica la transformación a las imágenes actualmente visibles"""
        if self.images is None:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas para transformar.")
            self.hide_status_bar()
            return
        
        # Obtenemos las imágenes actuales (ya transformadas si aplica)
        current_axial, current_sagittal, current_coronal = self.get_current_slices()
        
        checked_button = self.transform_options.checkedButton()
        if not checked_button:
            QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ninguna transformación.")
            self.hide_status_bar()
            return
        
        transform_type = checked_button.text()
        
        try:
            if transform_type == "Rotación":
                # Aplicamos rotación a las imágenes ya transformadas
                self.transformed_axial, self.transformed_sagittal, self.transformed_coronal = self.apply_rotation(current_axial, current_sagittal, current_coronal)
            elif transform_type == "Traslación":
                self.transformed_axial, self.transformed_sagittal, self.transformed_coronal = (self.apply_translation(current_axial, current_sagittal, current_coronal))
            elif transform_type == "Escalamiento":
                self.transformed_axial, self.transformed_sagittal, self.transformed_coronal = (self.apply_scaling(current_axial, current_sagittal, current_coronal))
            elif transform_type == "Inclinación":
                self.transformed_axial, self.transformed_sagittal, self.transformed_coronal = (self.apply_shearing(current_axial, current_sagittal, current_coronal))
            self.display_transformed_slices()
        finally:
            self.hide_status_bar()

    def display_transformed_slices(self):
        """Muestra los slices transformados en las vistas correspondientes"""
        if self.transformed_axial is not None:
            # La axial no necesita transformaciones adicionales
            self.display_slice(self.transformed_axial, self.axial_renderer, self.axial_vtk_widget)
        
        if self.transformed_sagittal is not None:
            # La sagital ya fue transformada en apply_rotation
            self.display_slice(self.transformed_sagittal, self.sagittal_renderer, self.sagittal_vtk_widget)
        
        if self.transformed_coronal is not None:
            # La coronal ya fue transformada en apply_rotation
            self.display_slice(self.transformed_coronal, self.coronal_renderer, self.coronal_vtk_widget)


    # Aplicar Rotación
    def apply_rotation(self, axial_slice, sagittal_slice, coronal_slice):
        """Aplica rotación a las imágenes ya transformadas sin usar funciones especializadas"""
        angle_deg = self.angle_input.value()
        
        def rotate_image(img_array, angle_degrees):
            angle = -np.radians(angle_degrees)  # signo negativo para rotar en sentido horario
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)

            T = np.array([
                [cos_theta, -sin_theta],
                [sin_theta,  cos_theta]
            ])

            h, w = img_array.shape

            corners = np.array([
                [0, 0],
                [0, w],
                [h, 0],
                [h, w]
            ])
            rotated_corners = np.dot(corners, T.T)
            min_coords = rotated_corners.min(axis=0)
            max_coords = rotated_corners.max(axis=0)

            new_h = int(np.ceil(max_coords[0] - min_coords[0]))
            new_w = int(np.ceil(max_coords[1] - min_coords[1]))

            rotated_img = np.full((new_h, new_w), self.borders[0], dtype=img_array.dtype)

            offset = -min_coords

            for y in range(h):
                for x in range(w):
                    new_coords = np.dot(T, np.array([y, x])) + offset
                    new_y, new_x = np.round(new_coords).astype(int)
                    if 0 <= new_y < new_h and 0 <= new_x < new_w:
                        rotated_img[new_y, new_x] = img_array[y, x]

            return rotated_img
        
        # Rotar cada slice
        rotated_axial = rotate_image(axial_slice, angle_deg)
        rotated_sagittal = rotate_image(sagittal_slice, angle_deg)
        rotated_coronal = rotate_image(coronal_slice, angle_deg)
        
        return rotated_axial, rotated_sagittal, rotated_coronal


    # Aplicar Traslación
    def apply_translation(self, axial_slice, sagittal_slice, coronal_slice):
        """Aplica traslación a los slices actuales según los valores especificados"""
        dx = self.h_translation_input.value()
        dy = self.v_translation_input.value()
            
        def translate_slice(img_array, tx_h, theta):
            h, w = img_array.shape

            # Imagen de salida del mismo tamaño que la original, con relleno en -1000
            translated_img = np.full((h, w), self.borders[0], dtype=img_array.dtype)

            for y in range(h):
                for x in range(w):
                    new_y = int(y + tx_h)
                    new_x = int(x + theta)
                    # Solo asigna si la nueva posición cae dentro de la imagen de salida
                    if 0 <= new_y < h and 0 <= new_x < w:
                        translated_img[new_y, new_x] = img_array[y, x]

            return translated_img
        
        translated_axial = translate_slice(axial_slice, dy, dx)
        translated_sagittal = translate_slice(sagittal_slice, dy, dx)
        translated_coronal = translate_slice(coronal_slice, dy, dx)
        
        return translated_axial, translated_sagittal, translated_coronal
    
    
    def apply_scaling(self, axial_slice, sagittal_slice, coronal_slice):
        """Aplica escalamiento a los slices actuales según los factores especificados"""
        scale_x = self.h_scaling_input.value()
        scale_y = self.v_scaling_input.value()


        def bilinear_interpolation(img_array):
            """ Rellena los valores -1000 en la imagen usando interpolación bilineal """
            h, w = img_array.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img_array[y, x] == -1000:  # Si el píxel está vacío
                        vecinos = []
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w and img_array[ny, nx] != -1000:
                                    vecinos.append(img_array[ny, nx])
                        if vecinos:
                            img_array[y, x] = np.mean(vecinos)  # Promedio de vecinos válidos

            return img_array
        
        def scale_slice(img_array, scale_y, scale_x):
            h, w = img_array.shape

            # Nuevas dimensiones según los factores de escala
            new_h = round(h * scale_y)
            new_w = round(w * scale_x)

            # Inicializar la imagen escalada con -1000
            scaled_img = np.full((new_h, new_w), self.borders[0], dtype=img_array.dtype)

            # Escalado directo
            for y in range(h):
                for x in range(w):
                    new_y = round(y * scale_y)
                    new_x = round(x * scale_x)
                    if 0 <= new_y < new_h and 0 <= new_x < new_w:
                        scaled_img[new_y, new_x] = img_array[y, x]

            return scaled_img
        
        scaled_axial = bilinear_interpolation(scale_slice(axial_slice, scale_y, scale_x))
        scaled_sagittal = bilinear_interpolation(scale_slice(sagittal_slice, scale_y, scale_x))
        scaled_coronal = bilinear_interpolation(scale_slice(coronal_slice, scale_y, scale_x))
        
        return scaled_axial, scaled_sagittal, scaled_coronal

    def apply_shearing(self, axial_slice, sagittal_slice, coronal_slice):
        """Aplica inclinación a los slices actuales según los factores especificados"""
        shear_x = self.h_shearing_input.value()
        shear_y = self.v_shearing_input.value()
        
        def shear_slice(img_array, shear_y_deg, shear_x_deg):
            h, w = img_array.shape

            # Convertir ángulos a radianes
            shear_y = np.tan(np.radians(shear_y_deg))
            shear_x = np.tan(np.radians(shear_x_deg))

            # Matriz de proyección (shear)
            H = np.array([
                [1, shear_y, 0],
                [shear_x, 1, 0],
                [0, 0, 1]
            ])

            # Esquinas de la imagen para determinar tamaño necesario
            corners = np.array([
                [0, 0, 1],
                [0, w, 1],
                [h, 0, 1],
                [h, w, 1]
            ])
            projected = np.dot(H, corners.T).T
            max_y, max_x = projected[:, 0].max(), projected[:, 1].max()
            min_y, min_x = projected[:, 0].min(), projected[:, 1].min()

            new_h = int(np.ceil(max_y - min_y))
            new_w = int(np.ceil(max_x - min_x))

            # Imagen de salida inicializada
            projected_img = np.full((new_h, new_w), self.borders[0], dtype=img_array.dtype)

            offset_y = -int(np.floor(min_y))
            offset_x = -int(np.floor(min_x))

            for y in range(h):
                for x in range(w):
                    original_coord = np.array([y, x, 1])
                    new_coord = H @ original_coord
                    new_y, new_x = np.round(new_coord[0] + offset_y), np.round(new_coord[1] + offset_x)
                    new_y, new_x = int(new_y), int(new_x)
                    if 0 <= new_y < new_h and 0 <= new_x < new_w:
                        projected_img[new_y, new_x] = img_array[y, x]

            return projected_img
        
        sheared_axial = shear_slice(axial_slice, shear_y , shear_x)
        sheared_sagittal = shear_slice(sagittal_slice, shear_y , shear_x)
        sheared_coronal = shear_slice(coronal_slice, shear_y , shear_x)
        
        return sheared_axial, sheared_sagittal, sheared_coronal
    #=====================================================================================================

    #==========================================MODIFICAION DE RESOLUCION==================================
    def create_spatial_resolution_group(self, parent_layout):
        """Grupo para Resolución Espacial con estilo similar a transformación"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # RadioButton
        self.spatial_resolution_rb = QRadioButton("Resolución Espacial")
        self.spatial_resolution_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.resolution_options.addButton(self.spatial_resolution_rb)
        layout.addWidget(self.spatial_resolution_rb)
        
        # Controles
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["Submuestreo", "Sobremuestreo"])
        self.sampling_combo.setEnabled(False)
        
        self.percentage_input = QSpinBox()
        self.percentage_input.setRange(0, 50)
        self.percentage_input.setSingleStep(10)
        self.percentage_input.setValue(0)
        self.percentage_input.setSuffix("%")
        self.percentage_input.setEnabled(False)
        
        # Layout para controles
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Tipo:"))
        controls_layout.addWidget(self.sampling_combo)
        controls_layout.addWidget(QLabel("Porciento:"))
        controls_layout.addWidget(self.percentage_input)
        
        layout.addLayout(controls_layout)
        parent_layout.addWidget(container)

    def create_radiometric_resolution_group(self, parent_layout):
        """Grupo para Resolución Radiométrica con estilo similar a transformación"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # RadioButton
        self.radiometric_resolution_rb = QRadioButton("Resolución Radiométrica")
        self.radiometric_resolution_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.resolution_options.addButton(self.radiometric_resolution_rb)
        layout.addWidget(self.radiometric_resolution_rb)
        
        # Controles
        self.bits_input = QSpinBox()
        self.bits_input.setRange(1, 16)
        self.bits_input.setValue(8)
        self.bits_input.setEnabled(False)
        
        layout.addWidget(QLabel("No. de Bits:"))
        layout.addWidget(self.bits_input)
        parent_layout.addWidget(container)

    def create_temporal_resolution_group(self, parent_layout):
        """Grupo para Resolución Temporal con estilo similar a transformación"""
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # RadioButton
        self.temporal_resolution_rb = QRadioButton("Resolución Temporal")
        self.temporal_resolution_rb.setStyleSheet("""
            QRadioButton {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.resolution_options.addButton(self.temporal_resolution_rb)
        layout.addWidget(self.temporal_resolution_rb)
        
        # Controles
        self.movement_combo = QComboBox()
        self.movement_combo.addItems(["Movimiento Vertical", "Movimiento Horizontal"])
        self.movement_combo.setEnabled(False)
        
        self.movement_intensity_input = QSpinBox()  # Cambiar a QSpinBox
        self.movement_intensity_input.setRange(0, 100)  # Rango de 0 a 100
        self.movement_intensity_input.setValue(0)
        self.movement_intensity_input.setSingleStep(10)  # Paso de 1
        self.movement_intensity_input.setSuffix("%")
        self.movement_intensity_input.setEnabled(False)
        
        # Layout para controles
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Tipo de Movimiento:"))
        controls_layout.addWidget(self.movement_combo)
        controls_layout.addWidget(QLabel("Intensidad:"))
        controls_layout.addWidget(self.movement_intensity_input)
        
        layout.addLayout(controls_layout)
        parent_layout.addWidget(container)

    def update_resolution_controls(self):
        """Habilita solo los controles correspondientes a la opción seleccionada"""
        if not hasattr(self, 'resolution_options') or not self.resolution_options.buttons():
            return
        
        # Obtener el RadioButton seleccionado
        selected_rb = self.resolution_options.checkedButton()
        
        # Deshabilitar todos los controles primero
        self.sampling_combo.setEnabled(False)
        self.percentage_input.setEnabled(False)
        self.bits_input.setEnabled(False)
        self.movement_combo.setEnabled(False)
        self.movement_intensity_input.setEnabled(False)
        
        # Habilitar los controles correspondientes a la opción seleccionada
        if selected_rb == self.spatial_resolution_rb:
            self.sampling_combo.setEnabled(True)
            self.percentage_input.setEnabled(True)
        elif selected_rb == self.radiometric_resolution_rb:
            self.bits_input.setEnabled(True)
        elif selected_rb == self.temporal_resolution_rb:
            self.movement_combo.setEnabled(True)
            self.movement_intensity_input.setEnabled(True)

    def apply_resolution_changes(self):
        self.show_status_bar("Cargando...")
        QApplication.processEvents()  # Ensure UI updates immediately
        """Aplica los cambios de resolución seleccionados"""
        if self.images is None:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas para modificar.")
            self.hide_status_bar()
            return
        
        selected_rb = self.resolution_options.checkedButton()
        
        try:
            if selected_rb == self.spatial_resolution_rb:
                sampling_type = self.sampling_combo.currentText()
                percentage = self.percentage_input.value()
                self.apply_spatial_resolution(sampling_type, percentage)
                
            elif selected_rb == self.radiometric_resolution_rb:
                # Aplicar cambios de resolución radiométrica
                bits = self.bits_input.value()
                self.apply_radiometric_resolution(bits)
                
            elif selected_rb == self.temporal_resolution_rb:
                # Aplicar cambios de resolución temporal
                movement_type = self.movement_combo.currentText()
                intensity = self.movement_intensity_input.value()
                self.apply_temporal_resolution(movement_type, intensity)
        finally:
            self.hide_status_bar()

    def apply_spatial_resolution(self, sampling_type, percentage):
        """Aplica submuestreo o sobremuestreo a las imágenes visibles"""

        # Obtener slices actuales
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])

        def resample_image(img_array, sampling_type, percentage):

            h, w = img_array.shape

            if sampling_type == "Submuestreo":
                factor = 1 + (percentage / 100) * 10
                factor = int(np.clip(round(factor), 1, min(h, w)))
                reduced = img_array[::factor, ::factor]
                return reduced

            elif sampling_type == "Sobremuestreo":
                scale = 1 + (percentage / 100)  # 0.3 → 1.3x

                new_h = int(h * scale)
                new_w = int(w * scale)

                # Crear la nueva imagen vacía
                resized = np.zeros((new_h, new_w), dtype=np.float32)

                # Coordenadas en la imagen original
                row_idx = np.linspace(0, h - 1, new_h)
                col_idx = np.linspace(0, w - 1, new_w)

                for i, y in enumerate(row_idx):
                    y0 = int(np.floor(y))
                    y1 = min(y0 + 1, h - 1)
                    wy = y - y0

                    for j, x in enumerate(col_idx):
                        x0 = int(np.floor(x))
                        x1 = min(x0 + 1, w - 1)
                        wx = x - x0

                        # Interpolación bilineal
                        top = (1 - wx) * img_array[y0, x0] + wx * img_array[y0, x1]
                        bottom = (1 - wx) * img_array[y1, x0] + wx * img_array[y1, x1]
                        interpolated = (1 - wy) * top + wy * bottom

                        resized[i, j] = interpolated

                return resized

        # Aplicar la resolución espacial a cada slice
        self.transformed_axial = resample_image(current_axial, sampling_type, percentage)
        self.transformed_sagittal = resample_image(current_sagittal, sampling_type, percentage)
        self.transformed_coronal = resample_image(current_coronal, sampling_type, percentage)

        # Mostrar los resultados
        self.display_transformed_slices()

    def display_transformed_slices(self):
        """Muestra los slices transformados en las vistas correspondientes"""
        if self.transformed_axial is not None:
            self.display_slice(self.transformed_axial, self.axial_renderer, self.axial_vtk_widget)
        
        if self.transformed_sagittal is not None:
            self.display_slice(self.transformed_sagittal, self.sagittal_renderer, self.sagittal_vtk_widget)
        
        if self.transformed_coronal is not None:
            self.display_slice(self.transformed_coronal, self.coronal_renderer, self.coronal_vtk_widget)


    # Aplicar Resolución Radiométrica --------------------------------------------------
    def apply_radiometric_resolution(self, bits):
        """Aplica la reducción de bits a las tres vistas"""
        if self.images is None:
            return
        
        # Obtener los slices actuales (similar a transformación de coordenadas)
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])
        
        def reduce_bit_depth(image, bits):
            """Reduce la profundidad de bits de la imagen"""
            if bits >= 16:
                return image  # No hacer nada si ya es 16 bits o más
            
            # Normalizar a [0,1], aplicar reducción de bits, y volver a escala original
            min_val = image.min()
            max_val = image.max()
            
            if max_val == min_val:  # Evitar división por cero
                return image
            
            # Normalizar
            normalized = (image - min_val) / (max_val - min_val)
            
            # Reducir bits
            max_val_reduced = (1 << bits) - 1
            reduced = np.round(normalized * max_val_reduced) / max_val_reduced
            
            # Volver a la escala original
            return reduced * (max_val - min_val) + min_val
        
        # Aplicar reducción de bits a cada slice
        self.transformed_axial =reduce_bit_depth(current_axial, bits)
        self.transformed_sagittal =reduce_bit_depth(current_sagittal, bits)
        self.transformed_coronal =reduce_bit_depth(current_coronal, bits)
        
        # Mostrar los resultados
        self.display_transformed_slices()


    # Aplicar Resolución Temporal --------------------------------------------------
    def apply_temporal_resolution(self, movement_type, intensity):
        """Aplica blur de movimiento según el tipo y la intensidad especificados"""
        if self.images is None:
            return

        # Obtener slices actuales
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])

        def apply_temporal_motion(img_array, motion_strength, is_vertical):

            motion_strength = motion_strength / 100.0  # Convertir a rango [0, 1]
            """
            Aplica desenfoque por movimiento simulado sin alterar la escala de intensidades DICOM.
            """
            img = img_array.astype(np.float32)
            h, w = img.shape

            def get_normalized_toeplitz_vector(size, strength):
                if strength <= 0.0:
                    return np.zeros(size)

                n = np.arange(0.01, 3.01, 0.01)
                t = np.exp(-n / (strength ** 2))
                t_sum = t.sum()

                if t_sum == 0:  # evitar división por cero
                    return np.zeros(size)

                t /= t_sum
                vec = np.zeros(size)
                vec[:len(t)] = t
                return vec

            if is_vertical:
                vec = get_normalized_toeplitz_vector(h, motion_strength)
                move = toeplitz(vec) if motion_strength > 0 else np.eye(h, dtype=np.float32)
                result = move @ img
            else:
                vec = get_normalized_toeplitz_vector(w, motion_strength)
                move = toeplitz(vec) if motion_strength > 0 else np.eye(w, dtype=np.float32)
                result = img @ move

            return result  # Misma escala que la imagen original

        # Determinar el tipo de movimiento
        is_vertical = movement_type == "Movimiento Vertical"

        # Aplicar blur de movimiento
        self.transformed_axial = apply_temporal_motion(current_axial, intensity, is_vertical)
        self.transformed_sagittal = apply_temporal_motion(current_sagittal, intensity, is_vertical)
        self.transformed_coronal = apply_temporal_motion(current_coronal, intensity, is_vertical)

        self.display_transformed_slices()

    
    #=====================================================================================================

    #==========================================MODIFICAION DE RESOLUCION==================================
    # Implementación de filtrado frecuencial

    def apply_frequency_filter_to_current_image(self):
        """Aplica el filtro frecuencial a la imagen actual según los parámetros seleccionados."""
        if self.images is None:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas para filtrar.")
            return
        

        # Obtener slices actuales
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])


        # Obtener parámetros del filtro
        filter_name = self.frequency_window_combo.currentText()
        filter_type = "low" if self.frequency_type_combo.currentText() == "Pasa Bajas" else "high"
        sigma = self.frequency_radius_dimension_input.value()


        def apply_frequency_filter(image, filter_type, filter_name, sigma, K=0.0001, n=2):
            K = sigma
            n = 2
            border_factor = self.frequency_input.value()
            M, N = image.shape
            #H = np.zeros((M, N))
            #H = np.full((M, N), self.borders[0], dtype=np.float32)
            H = np.zeros((M, N), dtype=np.float32)

            for h in range(M):
                dx = (h - M / 2) / (M / 2)
                for k in range(N):
                    dy = (k - N / 2) / (N / 2)
                    dxy = np.sqrt(dx**2 + dy**2)

                    if filter_name == "Coseno":
                        if abs(dx) < sigma and abs(dy) < sigma:
                            H[h, k] = np.cos((np.pi * dx) / (2 * sigma)) * np.cos((np.pi * dy) / (2 * sigma))
                    elif filter_name == "Gaussiana Mod":
                        H[h, k] = np.exp(-(dxy**n) / K)
                    elif filter_name == "Barlett":
                        if 0 <= dxy / sigma <= 1:
                            H[h, k] = 1 - (dxy / sigma)
                    elif filter_name == "Hanning":
                        if dxy / sigma < np.pi and dxy < sigma:
                            H[h, k] = 0.5 * (np.cos((np.pi * dxy) / sigma) + 1)
                    elif filter_name == "Gaussiana":
                        H[h, k] = np.exp(-(dxy**2) / (2 * sigma**2))
                    else:
                        raise ValueError(f"Filtro '{filter_name}' no reconocido.")

            if filter_type == "high":
                H = 1 - H
                H *= border_factor

            else:
                H = H ** border_factor
                

            return H
        
        def filter_image(image, H):
            F = fftshift(fft2(image))
            G = F * H
            img_filtered = np.real(ifft2(ifftshift(G)))
            return img_filtered


        # Aplicar el filtro
        transformed_axial = apply_frequency_filter(current_axial, filter_type,filter_name, sigma)
        transformed_sagittal = apply_frequency_filter(current_sagittal, filter_type,filter_name, sigma)
        transformed_coronal = apply_frequency_filter(current_coronal, filter_type,filter_name, sigma)

        self.transformed_axial = filter_image(current_axial, transformed_axial)
        self.transformed_sagittal = filter_image(current_sagittal, transformed_sagittal)
        self.transformed_coronal = filter_image(current_coronal, transformed_coronal)



        # Mostrar resultados
        self.display_transformed_slices()

    def apply_spatial_filter_to_current_image(self):
        """Aplica el filtro espacial a la imagen actual según los parámetros seleccionados."""
        if self.images is None:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas para filtrar.")
            return
        
        
        # Obtener slices actuales
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])

        


        def apply_spatial_filter(image, filter_type, filter_name, factor=1.0, kernel_size=3):

        
            if filter_type == "low":
                # Filtro pasabajos (suavizado)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
                filtered = convolve2d(image, kernel, mode='same', boundary='symm')
                
                # Mezcla con la imagen original para controlar el efecto
                # factor=1: solo filtro, factor=0: solo imagen original
                filtered = factor * filtered + (1 - factor) * image
                
            elif filter_type == "high":
                kernels = {
                    # Sobel (8 direcciones)
                    "Sobel-N":    np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]]),
                    "Sobel-NE":   np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]]),
                    "Sobel-E":    np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]]),
                    "Sobel-SE":   np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]]),
                    "Sobel-S":    np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]),
                    "Sobel-SW":   np.array([[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]]),
                    "Sobel-W":    np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]]),
                    "Sobel-NW":   np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]]),
                    
                    # Prewitt (8 direcciones)
                    "Prewitt-N":  np.array([[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]]),
                    "Prewitt-NE": np.array([[ 0,  1,  1], [-1,  0,  1], [-1, -1,  0]]),
                    "Prewitt-E":  np.array([[-1,  0,  1], [-1,  0,  1], [-1,  0,  1]]),
                    "Prewitt-SE": np.array([[-1, -1,  0], [-1,  0,  1], [ 0,  1,  1]]),
                    "Prewitt-S":  np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]]),
                    "Prewitt-SW": np.array([[ 0, -1, -1], [ 1,  0, -1], [ 1,  1,  0]]),
                    "Prewitt-W":  np.array([[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]]),
                    "Prewitt-NW": np.array([[ 1,  1,  0], [ 1,  0, -1], [ 0, -1, -1]]),
                    
                    # Laplace (3 variantes)
                    "Laplace":    np.array([[ 0,  1,  0], [ 1, -4,  1], [ 0,  1,  0]]),
                    "Laplace-8":  np.array([[ 1,  1,  1], [ 1, -8,  1], [ 1,  1,  1]]),
                    "Laplace-D":  np.array([[ 1,  0,  1], [ 0, -4,  0], [ 1,  0,  1]])
                }

                if filter_name not in kernels:
                    raise ValueError(f"Filtro no reconocido: {filter_name}")
                    
                kernel = kernels[filter_name]
                filtered = convolve2d(image, kernel, mode='same', boundary='symm')
                
                # Aplicar factor de intensidad
                filtered = filtered * factor
                
                
            else:
                raise ValueError("Tipo de filtro inválido (usa 'low' o 'high')")


            return filtered
        

        #Obtener parámetros del filtro
        filter_name = self.spatial_window_combo.currentText()
        filter_type = "low" if self.spatial_type_combo.currentText() == "Pasa Bajas" else "high"
        kernel_size = self.kernel_size_input.value()
        factor = self.spatial_input.value()
        
        self.transformed_axial = apply_spatial_filter(current_axial, filter_type, filter_name, factor, kernel_size)
        self.transformed_sagittal = apply_spatial_filter(current_sagittal, filter_type, filter_name, factor, kernel_size)
        self.transformed_coronal = apply_spatial_filter(current_coronal, filter_type, filter_name, factor, kernel_size)

        # Mostrar resultados
        self.display_transformed_slices()
        
        return

        

    def apply_filters(self):
        self.show_status_bar("Cargando...")
        QApplication.processEvents()  # Ensure UI updates immediately
        """Ejecuta las funciones de filtrado espacial y frecuencial según las opciones seleccionadas."""
        try:
            if self.frequency_domain_rb.isChecked():
                self.apply_frequency_filter_to_current_image()
            elif self.spatial_domain_rb.isChecked():
                self.apply_spatial_filter_to_current_image()
        finally:
            self.hide_status_bar()

    #=====================================================================================================

    #==========================================MEJORAMIENTO ESPACIAL======================================

    def apply_spatial_improvement(self):
        self.show_status_bar("Cargando...")
        QApplication.processEvents()  # Ensure UI updates immediately
        """Aplica el mejoramiento espacial a las imágenes visibles."""
        if self.images is None:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas para mejorar.")
            self.visualization_combo.setEnabled(False)  # Desactiva el ComboBox si no hay imágenes
            self.hide_status_bar()
            return
        else:
            self.visualization_combo.setEnabled(True)  # Activa el ComboBox si hay imágenes

        # Obtener parámetros de entrada
        dispersion_width = self.dispersion_width_input.value()
        method = self.visualization_combo.currentText()

        # Obtener slices actuales
        current_axial = self.images[self.current_axial, :, :].copy()
        current_sagittal = self.prepare_sagittal_slice(self.images[:, :, self.current_sagittal])
        current_coronal = self.prepare_coronal_slice(self.images[:, self.current_coronal, :])

        def restaurar_imagen(imagen_original, tipo='LS', K=10, N0=0.1, alpha=0.1, m1=0.3):
            """
            Aplica un algoritmo de reconstrucción (LS, CLS, WCLS o BMR) a una imagen degradada.

            Parámetros:
            - imagen_original: ndarray 2D (imagen en escala de grises, normalizada entre 0 y 1)
            - tipo: str, uno de 'LS', 'CLS', 'WCLS', 'BMR'
            - K: int, ancho de la función de dispersión (matriz del sistema S)
            - N0: float, varianza del ruido
            - alpha: float, parámetro de regularización para CLS y WCLS
            - m1: float, parámetro de ponderación para WCLS
            """
            if tipo not in ['LS', 'CLS', 'WCLS', 'BMR']:
                raise ValueError("Tipo debe ser uno de: 'LS', 'CLS', 'WCLS', 'BMR'")

            V = imagen_original
            M, N = V.shape

            # Generar ruido gaussiano
            ruido = np.random.randn(M, N) * np.sqrt(N0)

            # Construcción de la matriz S usando |sinc(x)|
            a = np.zeros(M)
            for i in range(1, 2 * (K // 2) + 1):
                a[i] = abs(np.sinc(i / (K // 2)))
            a[0] = 1
            S = toeplitz(a)
            S /= np.sum(S[M // 2, :])  # Normalización

            # Imagen degradada
            U = S @ V + ruido

            # Estimaciones
            if tipo == 'LS':
                restaurada = np.linalg.inv(S.T @ S) @ S.T @ U

            elif tipo == 'CLS':
                restaurada = np.linalg.inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U

            elif tipo == 'WCLS':
                Mu = (1 / N0) * np.eye(M)
                b = np.zeros(M)
                b[0], b[1] = 2, -1
                Mv = toeplitz(b)
                Mv[0, 0] = Mv[-1, -1] = 1
                Mv = np.eye(M) + m1 * Mv
                mv = np.linalg.inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U
                W = np.linalg.inv(S.T @ Mu @ S + alpha * Mv) @ S.T @ Mu
                restaurada = mv + W @ (U - S @ mv)

            elif tipo == 'BMR':
                mv = np.linalg.inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U
                Rn = N0 * np.eye(M)
                mean_v = np.mean(V)
                Rv = (V - mean_v) @ (V - mean_v).T + 0.1 * np.eye(M)
                Mu_BMR = np.linalg.inv(Rn)
                Mv_BMR = np.linalg.inv(Rv)
                W3 = np.linalg.inv(S.T @ Mu_BMR @ S + Mv_BMR) @ S.T @ Mu_BMR
                restaurada = mv + W3 @ (U - S @ mv)

            return restaurada, U
        
        def calcular_metricas(original, reconstruida):
            """
            Calcula PSNR, IOSNR, MAE y SSIM entre dos imágenes.
            Normaliza ambas imágenes al rango [0, 1] antes de calcular las métricas.
            """
            norm = lambda img: (img - img.min()) / (img.max() - img.min()) if img.max() != img.min() else img
            original_norm = norm(original)
            reconstruida_norm = norm(reconstruida)

            psnr_val = peak_signal_noise_ratio(original_norm, reconstruida_norm, data_range=1.0)
            iosnr_val = 10 * np.log10(np.mean(original_norm**2) / np.mean((original_norm - reconstruida_norm)**2))
            mae_val = np.mean(np.abs(original_norm - reconstruida_norm))
            ssim_val = structural_similarity(original_norm, reconstruida_norm, data_range=1.0)

            return np.array([psnr_val, iosnr_val, mae_val, ssim_val])




        self.cls_axial, U_axial = restaurar_imagen(current_axial, tipo='CLS', K=int(dispersion_width), N0=0.1)
        self.cls_sagital, U_sagital = restaurar_imagen(current_sagittal, tipo='CLS', K=int(dispersion_width), N0=0.1)
        self.cls_coronal, U_coronal = restaurar_imagen(current_coronal, tipo='CLS', K=int(dispersion_width), N0=0.1)

        self.wcls_axial, U_axial = restaurar_imagen(current_axial, tipo='WCLS', K=int(dispersion_width), N0=0.1)
        self.wcls_sagital, U_sagital = restaurar_imagen(current_sagittal, tipo='WCLS', K=int(dispersion_width), N0=0.1)
        self.wcls_coronal, U_coronal = restaurar_imagen(current_coronal, tipo='WCLS', K=int(dispersion_width), N0=0.1)

        self.bmr_axial, U_axial = restaurar_imagen(current_axial, tipo='BMR', K=int(dispersion_width), N0=0.1)
        self.bmr_sagital, U_sagital = restaurar_imagen(current_sagittal, tipo='BMR', K=int(dispersion_width), N0=0.1)
        self.bmr_coronal, U_coronal = restaurar_imagen(current_coronal, tipo='BMR', K=int(dispersion_width), N0=0.1)

        def actualizar_vistas(method):

            print(f"Aplicando método: {method}")
        
            if method == "CLS":
                self.transformed_axial = self.cls_axial
                self.transformed_sagittal = self.cls_sagital
                self.transformed_coronal = self.cls_coronal
            elif method == "WCLS":
                self.transformed_axial = self.wcls_axial
                self.transformed_sagittal = self.wcls_sagital
                self.transformed_coronal = self.wcls_coronal
            elif method == "BMR":
                self.transformed_axial = self.bmr_axial
                self.transformed_sagittal = self.bmr_sagital
                self.transformed_coronal = self.bmr_coronal
            elif method == "Imagen Degradada":
                self.transformed_axial = U_axial
                self.transformed_sagittal = U_sagital
                self.transformed_coronal = U_coronal
            elif method == "Imagen Original":
                self.transformed_axial = current_axial
                self.transformed_sagittal = current_sagittal
                self.transformed_coronal = current_coronal
            else:
                self.transformed_axial = current_axial
                self.transformed_sagittal = current_sagittal
                self.transformed_coronal = current_coronal

        # Actualizar vistas según el método seleccionado
        actualizar_vistas(method)

        metricas_axial_cls = calcular_metricas(current_axial, self.cls_axial)
        metricas_sagittal_cls = calcular_metricas(current_sagittal,self.cls_sagital )
        metricas_coronal_cls = calcular_metricas(current_coronal,self.cls_coronal)

        metricas_axial_wcls = calcular_metricas(current_axial, self.wcls_axial)
        metricas_sagittal_wcls = calcular_metricas(current_sagittal, self.wcls_sagital)
        metricas_coronal_wcls = calcular_metricas(current_coronal, self.wcls_coronal)

        metricas_axial_bmr = calcular_metricas(current_axial, self.bmr_axial)
        metricas_sagittal_bmr = calcular_metricas(current_sagittal, self.bmr_sagital)
        metricas_coronal_bmr = calcular_metricas(current_coronal, self.bmr_coronal)
        

        # Mostrar métricas en la tabla
        def actualizar_tabla_metricas(fila_inicio, metricas_axial, metricas_sagittal, metricas_coronal):
            """Actualiza las métricas en la tabla para un método específico."""
            def crear_celda_centrada(valor):
                """Crea un QTableWidgetItem con texto centrado."""
                item = QTableWidgetItem(f"{valor}")
                item.setTextAlignment(Qt.AlignCenter)
                return item

            self.metrics_table.setItem(fila_inicio, 0, crear_celda_centrada(f"{metricas_axial[0]:.2f}"))
            self.metrics_table.setItem(fila_inicio, 1, crear_celda_centrada(f"{metricas_sagittal[0]:.2f}"))
            self.metrics_table.setItem(fila_inicio, 2, crear_celda_centrada(f"{metricas_coronal[0]:.2f}"))

            self.metrics_table.setItem(fila_inicio + 1, 0, crear_celda_centrada(f"{metricas_axial[1]:.2f}"))
            self.metrics_table.setItem(fila_inicio + 1, 1, crear_celda_centrada(f"{metricas_sagittal[1]:.2f}"))
            self.metrics_table.setItem(fila_inicio + 1, 2, crear_celda_centrada(f"{metricas_coronal[1]:.2f}"))

            self.metrics_table.setItem(fila_inicio + 2, 0, crear_celda_centrada(f"{metricas_axial[2]:.4f}"))
            self.metrics_table.setItem(fila_inicio + 2, 1, crear_celda_centrada(f"{metricas_sagittal[2]:.4f}"))
            self.metrics_table.setItem(fila_inicio + 2, 2, crear_celda_centrada(f"{metricas_coronal[2]:.4f}"))

            self.metrics_table.setItem(fila_inicio + 3, 0, crear_celda_centrada(f"{metricas_axial[3]:.4f}"))
            self.metrics_table.setItem(fila_inicio + 3, 1, crear_celda_centrada(f"{metricas_sagittal[3]:.4f}"))
            self.metrics_table.setItem(fila_inicio + 3, 2, crear_celda_centrada(f"{metricas_coronal[3]:.4f}"))

        # Actualizar métricas para CLS
        actualizar_tabla_metricas(0, metricas_axial_cls, metricas_sagittal_cls, metricas_coronal_cls)

        # Actualizar métricas para WCLS
        actualizar_tabla_metricas(4, metricas_axial_wcls, metricas_sagittal_wcls, metricas_coronal_wcls)

        # Actualizar métricas para BMR
        actualizar_tabla_metricas(8, metricas_axial_bmr, metricas_sagittal_bmr, metricas_coronal_bmr)

        # Mostrar resultados
        self.display_transformed_slices()
        self._actualizar_vistas_func = actualizar_vistas
        self.hide_status_bar()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())




