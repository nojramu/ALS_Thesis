import os
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

import matplotlib.pyplot as plt

# Generate random 3D data
np.random.seed()  # Ensure randomness each run
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=np.random.rand(100), cmap='viridis', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Random 3D Scatter Plot')

# Ensure image directory exists
image_dir = os.path.join(os.path.dirname(__file__), '..', 'image')  # Save to /workspaces/ALS_Thesis/image
os.makedirs(image_dir, exist_ok=True)

# Save with timestamp to avoid overwrite
filename = f"random_3d_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
filepath = os.path.join(image_dir, filename)
plt.savefig(filepath)
plt.close()

print(f"Plot saved to {filepath}")