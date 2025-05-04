import numpy as np
import pydicom
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import os

def apply_frequency_filter(image, filter_type, filter_name, sigma, K=0.0001, n=2):
    M, N = image.shape
    H = np.zeros((M, N))

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

    return H

def filter_image(image, H):
    F = fftshift(fft2(image))
    G = F * H
    img_filtered = np.real(ifft2(ifftshift(G)))
    return img_filtered

# Cargar imagen DICOM desde el mismo directorio
def main():
    # Busca el primer archivo .dcm en el directorio actual
    for file in os.listdir():
        if file.lower().endswith('.dcm'):
            dicom_file = file
            break
    else:
        print("No se encontró ningún archivo DICOM en el directorio.")
        return

    ds = pydicom.dcmread(dicom_file)
    image = ds.pixel_array.astype(float)

    # Normalizar imagen a [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Crear el filtro
    H = apply_frequency_filter(image, filter_type="low", filter_name="Gaussiana", sigma=0.1)

    # Aplicar el filtro
    filtered = filter_image(image, H)

    # Mostrar resultados
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(H, cmap='viridis')
    plt.title("Filtro frecuencial")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(filtered, cmap='gray')
    plt.title("Imagen filtrada")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
