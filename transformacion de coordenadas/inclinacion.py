import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def shear_image(img_array, shear_y_deg, shear_x_deg):
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

    # Imagen de salida inicializada con -1000
    projected_img = np.full((new_h, new_w), -1000, dtype=img_array.dtype)

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


# Cargar imagen en escala de grises desde la ruta original
img = Image.open(r'C:/Users/gabyt/Downloads/credencial Alejandro.jpg').convert('L')
img_array = np.asarray(img)

# Aplicar inclinación de 15° vertical y 30° horizontal
sheared = shear_image(img_array, shear_y_deg=30, shear_x_deg=30)

# Visualización
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(sheared, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Inclinada (Proyección)')
plt.show()
