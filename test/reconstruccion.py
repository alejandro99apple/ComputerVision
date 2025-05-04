import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, inv

def restaurar_imagen(imagen_original, tipo='LS', K=10, N0=0.01, alpha=0.1, m1=0.3):
    """
    Aplica un algoritmo de reconstrucción (LS, CLS, WCLS o BMR) a una imagen degradada.

    Parámetros:
    - imagen_original: ndarray 2D (imagen en escala de grises, normalizada entre 0 y 1)
    - tipo: str, uno de 'LS', 'CLS', 'WCLS', 'BMR'
    - K: int, ancho de la función de dispersión (matriz del sistema S)
    - N0: float, varianza del ruido
    - alpha: float, parámetro de regularización para CLS y WCLS
    - m1: float, parámetro de ponderación para WCLS

    Muestra:
    - Gráficas de la imagen original, imagen degradada y restaurada.
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
        restaurada = inv(S.T @ S) @ S.T @ U

    elif tipo == 'CLS':
        restaurada = inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U

    elif tipo == 'WCLS':
        Mu = (1 / N0) * np.eye(M)
        b = np.zeros(M)
        b[0], b[1] = 2, -1
        Mv = toeplitz(b)
        Mv[0, 0] = Mv[-1, -1] = 1
        Mv = np.eye(M) + m1 * Mv
        mv = inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U
        W = inv(S.T @ Mu @ S + alpha * Mv) @ S.T @ Mu
        restaurada = mv + W @ (U - S @ mv)

    elif tipo == 'BMR':
        mv = inv(S.T @ S + alpha * np.eye(M)) @ S.T @ U
        Rn = N0 * np.eye(M)
        mean_v = np.mean(V)
        Rv = (V - mean_v) @ (V - mean_v).T + 0.1 * np.eye(M)
        Mu_BMR = inv(Rn)
        Mv_BMR = inv(Rv)
        W3 = inv(S.T @ Mu_BMR @ S + Mv_BMR) @ S.T @ Mu_BMR
        restaurada = mv + W3 @ (U - S @ mv)

    # Mostrar resultados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(V, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(U, cmap='gray')
    plt.title('Imagen Degradada')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(restaurada, cmap='gray')
    plt.title(f'Restauración {tipo}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


from skimage import io, color, img_as_float

# Cargar y convertir la imagen
imagen = img_as_float(color.rgb2gray(io.imread('C:/Users/gabyt/Downloads/credencial Alejandro.jpg')))

# Llamar la función
restaurar_imagen(imagen, tipo='CLS', K=10)
