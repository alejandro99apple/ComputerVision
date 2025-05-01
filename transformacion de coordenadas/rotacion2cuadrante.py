import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float

# Clear workspace and close figures (equivalent to MATLAB's clear all, clc, close all)
plt.close('all')

# Read and preprocess the image
X = io.imread('C:/Users/gabyt/Downloads/credencial Alejandro.jpg')
X = img_as_float(X)
X = color.rgb2gray(X)

# Rotation parameters
angle = 2 * np.pi / 3  # 120 degrees in radians
T = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 0]
])

# Get image dimensions (O is not needed for grayscale)
M, N = X.shape

# Calculate correction vectors
Corr1 = np.array([[M], [N], [1]])
CuvCorr1 = np.ceil(T @ Corr1).astype(int)

Corr2 = np.array([[M], [1], [1]])
CuvCorr2 = np.ceil(T @ Corr2).astype(int)

# Determine output image size
# We'll calculate maximum possible dimensions to avoid clipping
max_dim = int(np.ceil(np.sqrt(M**2 + N**2)))
Y = np.zeros((max_dim, max_dim))

# Apply transformation
for ky in range(M):
    for kx in range(N):
        Cxy = np.array([[ky + 1], [kx + 1], [1]])  # +1 for MATLAB to Python indexing
        Cuv = np.round(T @ Cxy).astype(int)
        Cuv[0] = Cuv[0] - CuvCorr1[0] + 1
        Cuv[1] = Cuv[1] + CuvCorr2[1] - 1
        
        # Ensure coordinates are within bounds
        if (0 <= Cuv[0] < max_dim) and (0 <= Cuv[1] < max_dim):
            Y[Cuv[0, 0], Cuv[1, 0]] = X[ky, kx]

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Y, cmap='gray')
plt.title('Rotated Image (120°)')
plt.axis('off')

plt.tight_layout()
plt.show()