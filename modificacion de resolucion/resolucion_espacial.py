import numpy as np
from PIL import Image

def apply_spatial_resolution_dicom(img_array, factor):
    """
    Reduce la resolución espacial de una imagen DICOM 2D.
    - img_array: imagen 2D (grayscale)
    - factor: int >= 1, salto entre píxeles (por ejemplo, 8)
    
    Retorna: imagen reducida en resolución (misma escala radiométrica)
    """
    if factor < 1:
        raise ValueError("El factor debe ser >= 1")
    
    # Submuestreo directo
    reduced_img = img_array[::factor, ::factor].copy()
    return reduced_img


img_dicom = Image.open(r'C:/Users/gabyt/Downloads/credencial Alejandro.jpg').convert('L')
img_array = np.asarray(img_dicom)
# Supongamos que 'img_dicom' es una imagen DICOM 2D cargada
img_reduced = apply_spatial_resolution_dicom(img_dicom, factor=8)

# Mostrar ambas con el mismo rango
vmin, vmax = img_dicom.min(), img_dicom.max()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_dicom, cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Imagen Original")

plt.subplot(1, 2, 2)
plt.imshow(img_reduced, cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Resolución Espacial Reducida (n=8)")
plt.show()