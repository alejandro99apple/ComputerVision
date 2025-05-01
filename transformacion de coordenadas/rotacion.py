import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rotate_image(img_array, angle_degrees):
    angle = np.radians(angle_degrees)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Matriz de rotación
    T = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    # Dimensiones de la imagen original
    h, w = img_array.shape

    # Coordenadas de las esquinas para calcular nueva dimensión
    corners = np.array([
        [0, 0],
        [0, w],
        [h, 0],
        [h, w]
    ])
    rotated_corners = np.dot(corners, T.T)
    min_coords = rotated_corners.min(axis=0)
    max_coords = rotated_corners.max(axis=0)

    # Tamaño de la imagen de salida
    new_h = int(np.ceil(max_coords[0] - min_coords[0]))
    new_w = int(np.ceil(max_coords[1] - min_coords[1]))

    # Imagen de salida
    rotated_img = np.full((new_h, new_w), -1000, dtype=img_array.dtype)

    # Offset para evitar índices negativos
    offset = -min_coords

    for y in range(h):
        for x in range(w):
            new_coords = np.dot(T, np.array([y, x])) + offset
            new_y, new_x = np.round(new_coords).astype(int)
            if 0 <= new_y < new_h and 0 <= new_x < new_w:
                rotated_img[new_y, new_x] = img_array[y, x]

    return rotated_img

# Carga de imagen y prueba
img = Image.open('C:/Users/gabyt/Downloads/credencial Alejandro.jpg').convert('L')
img_array = np.asarray(img)  # normalización
rotated = rotate_image(img_array, 350)  # ejemplo con 60 grados

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img_array, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Rotada')
plt.imshow(rotated, cmap='gray')
plt.show()
