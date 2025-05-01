import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float

# Clear workspace
plt.close('all')

# Read and preprocess image
X = io.imread('C:/Users/gabyt/Downloads/credencial Alejandro.jpg')
X = img_as_float(X)
X = color.rgb2gray(X)

# Rotation parameters (216 degrees)
angle = 6 * np.pi / 5
T = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 0]
])

# Get image dimensions
M, N = X.shape

# Calculate correction vectors
Corr1 = np.array([[M], [1], [1]])    # Bottom-left corner
CuvCorr1 = np.ceil(T @ Corr1).astype(int)

Corr2 = np.array([[M], [N], [1]])    # Bottom-right corner
CuvCorr2 = np.ceil(T @ Corr2).astype(int)

# Estimate output size (add padding to prevent clipping)
max_dim = int(2 * np.sqrt(M**2 + N**2))
Y = np.zeros((max_dim, max_dim))

# Apply transformation with offset adjustment
for ky in range(M):
    for kx in range(N):
        Cxy = np.array([[ky + 1], [kx + 1], [1]])  # +1 for MATLAB to Python index conversion
        Cuv = np.round(T @ Cxy).astype(int)
        
        # Apply corrections (+2 as in original MATLAB code)
        Cuv[0] = Cuv[0] - CuvCorr1[0] + 2
        Cuv[1] = Cuv[1] - CuvCorr2[1] + 2
        
        # Add center offset (to handle 3rd quadrant placement)
        center_offset = max_dim // 2
        new_y = Cuv[0, 0] + center_offset
        new_x = Cuv[1, 0] + center_offset
        
        # Check bounds before assignment
        if 0 <= new_y < max_dim and 0 <= new_x < max_dim:
            Y[new_y, new_x] = X[ky, kx]

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Y, cmap='gray', vmin=0, vmax=1)  # Ensure consistent scale
plt.title(f'Rotated Image ({np.degrees(angle):.1f}°)')
plt.axis('off')

plt.tight_layout()
plt.show()