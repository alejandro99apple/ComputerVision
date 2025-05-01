import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale_image(img_array, scale_y, scale_x):
    h, w = img_array.shape

    # Nuevas dimensiones según los factores de escala
    new_h = round(h * scale_y)
    new_w = round(w * scale_x)

    # Inicializar la imagen escalada con -1000
    scaled_img = np.full((new_h, new_w), -1000, dtype=img_array.dtype)

    # Escalado directo
    for y in range(h):
        for x in range(w):
            new_y = round(y * scale_y)
            new_x = round(x * scale_x)
            if 0 <= new_y < new_h and 0 <= new_x < new_w:
                scaled_img[new_y, new_x] = img_array[y, x]

    return scaled_img

# Cargar la imagen (escala de grises)
img_path = r'C:/Users/gabyt/Downloads/credencial Alejandro.jpg'
img = Image.open(img_path).convert('L')
img_array = np.asarray(img)

# Parámetros de escalado
scale_y = 1  # Escalado vertical
scale_x = 2  # Escalado horizontal

# Escalar imagen
scaled_img = scale_image(img_array, scale_y, scale_x)

# Visualización de resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen Escalada')
# Visualizamos con valores limitados entre 0-255 para ignorar los -1000
plt.imshow(np.clip(scaled_img, 0, 255), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
