import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float

# Clear workspace
plt.close('all')

# Read and preprocess image
X = io.imread('C:/Users/gabyt/Downloads/credencial Alejandro.jpg')
X = img_as_float(X)
X = color.rgb2gray(X)

# Rotation parameters (300 degrees)
angle = 5 * np.pi / 3
T = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 0]
])

# Get image dimensions
M, N = X.shape

# Calculate correction vectors
Corr1 = np.array([[M], [N], [1]])  # Bottom-right corner
CuvCorr1 = np.ceil(T @ Corr1).astype(int)

Corr2 = np.array([[M], [1], [1]])  # Bottom-left corner
CuvCorr2 = np.ceil(T @ Corr2).astype(int)

# Estimate output size with padding
max_dim = int(2 * np.sqrt(M**2 + N**2))
Y = np.zeros((max_dim, max_dim))

# Apply transformation with 4th quadrant adjustment
for ky in range(M):
    for kx in range(N):
        Cxy = np.array([[ky + 1], [kx + 1], [1]])  # MATLAB to Python index adjustment
        Cuv = np.round(T @ Cxy).astype(int)
        
        # Apply 4th quadrant correction (only y-coordinate adjusted)
        # Note: The x-correction (+2) is applied as in your MATLAB code
        Cuv[1] = Cuv[1] - CuvCorr2[1] + 2
        
        # Add offset to position in 4th quadrant
        offset_x = max_dim // 2
        offset_y = max_dim // 2
        new_x = Cuv[1, 0] + offset_x
        new_y = Cuv[0, 0] + offset_y
        
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
plt.imshow(Y, cmap='gray', vmin=0, vmax=1)
plt.title(f'Rotated Image ({np.degrees(angle):.1f}°) - 4th Quadrant')
plt.axis('off')

plt.tight_layout()
plt.show()