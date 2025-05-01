import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def translate_image(img_array, tx_h, theta):
    h, w, depth = img_array.shape

    # Matriz de traslación (no se usa directamente como matriz homogénea, pero sirve de guía)
    H = np.array([
        [1, 0, tx_h],
        [0, 1, theta],
        [0, 0, 1]
    ])

    # Dimensiones de la imagen trasladada
    new_h = h + abs(tx_h)
    new_w = w + abs(theta)

    # Imagen de salida inicializada (relleno con -1000 como antes)
    translated_img = np.full((new_h, new_w, depth), -1000, dtype=img_array.dtype)

    for z in range(depth):
        for y in range(h):
            for x in range(w):
                uv = np.array([y, x, 1])
                t_xy = H @ uv
                new_y, new_x = t_xy[0], t_xy[1]
                # Validación para que no se salga de los límites
                if 0 <= new_y < new_h and 0 <= new_x < new_w:
                    translated_img[new_y, new_x, z] = img_array[y, x, z]

    return translated_img

# Ejemplo de uso
img = Image.open('C:/Users/gabyt/Downloads/credencial Alejandro.jpg').convert('RGB')
img_array = np.asarray(img)

translated = translate_image(img_array, tx_h=50, theta=200)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
# Para visualizar correctamente ignoramos los -1000 (matplotlib los trata como negros)
plt.imshow(np.clip(translated, 0, 255).astype(np.uint8))
plt.title('Imagen Trasladada')
plt.show()
