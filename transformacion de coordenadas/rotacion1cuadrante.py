import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float

# Read and preprocess the image
X = io.imread('C:/Users/gabyt/Downloads/credencial Alejandro.jpg')
X = img_as_float(X)
X = color.rgb2gray(X)

# Rotation
angle = np.pi / 9
T = np.array([[np.cos(angle), -np.sin(angle), 0],
              [np.sin(angle), np.cos(angle), 0],
              [0, 0, 0]])

M, N = X.shape
O = 1  # Grayscale image has single channel

Corr1 = np.array([[1], [N], [1]])
CuvCorr = np.ceil(T @ Corr1).astype(int)

# Initialize output image (you might need to adjust the size based on rotation)
max_dim = int(np.ceil(np.sqrt(M**2 + N**2)))  # Maximum possible dimension after rotation
Y = np.zeros((max_dim, max_dim))

for ky in range(M):
    for kx in range(N):
        Cxy = np.array([[ky], [kx], [1]])
        Cuv = np.round(T @ Cxy).astype(int)
        Cuv[0] = Cuv[0] - CuvCorr[0] + 1
        try:
            Y[Cuv[0, 0], Cuv[1, 0]] = X[ky, kx]
        except IndexError:
            pass  # Skip if the coordinates are out of bounds

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(Y, cmap='gray')
plt.title('Rotated Image')
plt.axis('off')

plt.tight_layout()
plt.show()