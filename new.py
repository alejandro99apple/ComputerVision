import os
import vtk
import numpy as np
from vtk.util import numpy_support

class DicomViewer:
    def __init__(self):
        self.reader = None
        self.current_axial_slice = 0
        self.current_coronal_slice = 0
        self.current_sagittal_slice = 0

    def cargar_imagenes_dicom(self, directorio):
        """Carga imágenes DICOM desde un directorio especificado."""
        if not os.path.isdir(directorio):
            raise ValueError(f"El directorio {directorio} no existe.")
        
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(directorio)
        self.reader.Update()
        
        if self.reader.GetOutput().GetPointData().GetScalars() is None:
            raise ValueError("No se encontraron imágenes DICOM válidas en el directorio.")
        
        dims = self.reader.GetOutput().GetDimensions()
        self.total_slices = dims[2]
        self.current_slice = self.total_slices // 2

    def convertir_a_matriz_3d(self):
        """Convierte las imágenes DICOM a una matriz numpy 3D."""
        if self.reader is None:
            raise ValueError("No se han cargado imágenes DICOM.")
            
        vtk_data = self.reader.GetOutput().GetPointData().GetScalars()
        dims = self.reader.GetOutput().GetDimensions()
        
        numpy_data = numpy_support.vtk_to_numpy(vtk_data)
        numpy_data = numpy_data.reshape(dims[2], dims[1], dims[0])
        
        return numpy_data

    def mostrar_informacion_imagen(self):
        """Muestra información sobre las imágenes cargadas."""
        if self.reader is None:
            raise ValueError("No se han cargado imágenes DICOM.")
            
        dims = self.reader.GetOutput().GetDimensions()
        spacing = self.reader.GetOutput().GetSpacing()
        numpy_data = self.convertir_a_matriz_3d()
        
        print("\nInformación de la imagen DICOM:")
        print(f"Dimensiones (ancho, alto, slices): {dims}")
        print(f"Espaciado (mm) entre píxeles (x, y, z): {spacing}")
        print(f"Forma de la matriz numpy (slices, alto, ancho): {numpy_data.shape}")
        print(f"Tipo de datos: {numpy_data.dtype}")
        print(f"Valor mínimo: {np.min(numpy_data)}")
        print(f"Valor máximo: {np.max(numpy_data)}")
        print(f"Número total de slices: {self.total_slices}")

    def Axial_View(self, matrix_3d, current_slice):
        """Devuelve la vista axial (2D) correspondiente al slice actual."""
        return matrix_3d[current_slice, :, :]

    def Sagital_View(self, matrix_3d, current_slice):
        """Devuelve la vista sagital (2D) correspondiente al slice actual."""
        return matrix_3d[:, :, current_slice]

    def Coronal_View(self, matrix_3d, current_slice):
        """Devuelve la vista coronal (2D) correspondiente al slice actual."""
        return matrix_3d[:, current_slice, :]

def procesar_dicom(directorio_dicom):
    """Función principal que procesa las imágenes DICOM."""
    try:
        viewer = DicomViewer()
        viewer.cargar_imagenes_dicom(directorio_dicom)
        viewer.mostrar_informacion_imagen()
        
        # Convertir a matriz 3D
        matriz_3d = viewer.convertir_a_matriz_3d()
        
        # Obtener y mostrar las vistas
        axial = viewer.Axial_View(matriz_3d, viewer.current_axial_slice)
        sagittal = viewer.Sagital_View(matriz_3d, viewer.current_sagittal_slice)
        coronal = viewer.Coronal_View(matriz_3d, viewer.current_coronal_slice)
        
        print("\nVista Axial (matriz 2D):")
        print(axial)
        print("\nVista Sagital (matriz 2D):")
        print(sagittal)
        print("\nVista Coronal (matriz 2D):")
        print(coronal)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    directorio_ejemplo = "Img"
    procesar_dicom(directorio_ejemplo)