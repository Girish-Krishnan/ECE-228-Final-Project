import numpy as np
import matplotlib.pyplot as plt

# Load data
velocity = np.load('CurveFault_A/vel2_1_0.npy')
seismic_data = np.load('CurveFault_A/seis2_1_0.npy')

print(f'Velocity map size: {velocity.shape}')
print(f'Seismic data size: {seismic_data.shape}')

# Select a sample to visualize
sample_idx = 10

# Plot velocity map
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(velocity[sample_idx, 0, :, :], cmap='jet')
ax.set_xticks(range(0, 70, 10))
ax.set_xticklabels(range(0, 700, 100))
ax.set_yticks(range(0, 70, 10))
ax.set_yticklabels(range(0, 700, 100))
ax.set_xlabel('Offset (m)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_title('km/s', fontsize=8)
plt.show()

# Plot seismic data channels
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, axis in enumerate(axes):
    axis.imshow(
        seismic_data[sample_idx, i, :, :],
        extent=[0, 70, 1000, 0],
        aspect='auto',
        cmap='gray',
        vmin=-0.5,
        vmax=0.5
    )
    axis.set_xticks(range(0, 70, 10))
    axis.set_xticklabels(range(0, 700, 100))
    axis.set_yticks([0, 1000])
    axis.set_yticklabels([0, 1])  # Time (s)
    axis.set_xlabel('Offset (m)', fontsize=12)
    if i == 0:
        axis.set_ylabel('Time (s)', fontsize=12)
plt.tight_layout()
plt.show()
