import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os

# Función de filtrado espacial (como te la pasé antes)
def apply_spatial_filter(image, filter_type, filter_name, factor=1.0, kernel_size=3):
    image = image.astype(np.float32)

    if filter_type == "low":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    elif filter_type == "high":
        kernels = {
            "sobel-x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "sobel-y": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
            "prewitt-x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "prewitt-y": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
            "laplace": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        }

        if filter_name not in kernels:
            raise ValueError(f"Filtro no reconocido: {filter_name}")
        
        kernel = kernels[filter_name]
    else:
        raise ValueError("Tipo de filtro inválido (usa 'low' o 'high')")

    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            filtered[i, j] = np.sum(region * kernel)

    if filter_type == "low":
        filtered = filtered ** factor
    else:
        filtered = filtered * factor

    filtered = np.clip(filtered, 0, 1)
    return filtered

# === Cargar imagen DICOM ===
def load_dicom_image(filename):
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array.astype(np.float32)

    # Normalizar a [0, 1]
    image -= np.min(image)
    image /= np.max(image)
    return image

# === Ruta al archivo DICOM en la misma carpeta ===
for file in os.listdir():
    if file.endswith(".dcm"):
        dicom_file = file
        break

# Cargar imagen y aplicar filtro
image = load_dicom_image(dicom_file)

# Cambia estos parámetros según lo que quieras probar
filter_type = "low"              # "low" o "high"
filter_name = "laplace"           # Para high: "sobel-x", "sobel-y", "prewitt-x", "prewitt-y", "laplace", "sharpen"
factor = 1.0                      # Factor de borde o ruido
kernel_size = 7                   # Ignorado si es pasa altas

filtered = apply_spatial_filter(image, filter_type, filter_name, factor, kernel_size)

# === Mostrar resultados ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(filtered, cmap='gray')
plt.title(f"Filtrado: {filter_type} - {filter_name}")
plt.show()
