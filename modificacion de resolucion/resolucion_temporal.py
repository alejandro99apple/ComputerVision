import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import toeplitz  # Esta es parte de SciPy, ¿quieres que lo haga a mano?


import numpy as np
from scipy.linalg import toeplitz

def apply_temporal_motion_gray(img_array, motion_strength_v, motion_strength_h):
    """
    Aplica desenfoque simulando movimiento sobre imagen en escala de grises (2D).
    motion_strength_v: fuerza vertical (0 a 1)
    motion_strength_h: fuerza horizontal (0 a 1)
    """
    img = img_array.astype(np.float32) / 255.0
    h, w = img.shape

    def get_toeplitz_vector(size, strength):
        if strength <= 0.0:
            return np.zeros(size)
        n = np.arange(0.01, 3.01, 0.01)
        t = np.exp(-n / (strength ** 2))
        vec = np.concatenate([t, np.zeros(size - len(t))])
        return vec

    vecH = get_toeplitz_vector(w, motion_strength_h)
    vecV = get_toeplitz_vector(h, motion_strength_v)

    moveH = toeplitz(vecH) if motion_strength_h > 0 else np.eye(w)
    moveV = toeplitz(vecV) if motion_strength_v > 0 else np.eye(h)

    # Aplicar desenfoque: M * I * N
    result = moveV @ img @ moveH

    # Normalización para visualización
    result_max = result.max()
    if result_max > 0:
        result /= result_max
    else:
        result[:] = 0

    return result



# Cargar imagen en escala de grises
img = Image.open(r'C:/Users/gabyt/Downloads/credencial Alejandro.jpg').convert('L')
img_array = np.asarray(img)

# Aplicar desenfoque por movimiento horizontal fuerte, vertical nulo
motion_img = apply_temporal_motion_gray(img_array, motion_strength_v=1, motion_strength_h=0)

# Visualización
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title("Imagen Original")

plt.subplot(1, 2, 2)
plt.imshow(motion_img, cmap='gray')
plt.title("Movimiento H=0.8, V=0.0")
plt.show()