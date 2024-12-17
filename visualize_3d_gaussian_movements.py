import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load positions log
positions_log = torch.load("logs/gaussian_positions_log.pt")  # List of tensors [num_iterations, num_gaussians, 3]

# Number of snapshots to plot (e.g., first, middle, last)
snapshot_indices = [0, len(positions_log) // 2, len(positions_log) - 1]  # Start, middle, end

# Set up Matplotlib 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Colors for different snapshots
colors = ['blue', 'green', 'red']
labels = ['Start', 'Middle', 'End']

# Plot each snapshot
for i, idx in enumerate(snapshot_indices):
    positions = positions_log[idx]  # Shape: [num_gaussians, 3]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[i], s=5, label=f'{labels[i]} (Iter {idx * log_frequency})')

# Example: Plot trajectories for first K Gaussians
K = 10  # Number of Gaussians to track
for i in range(K):
    trajectory = torch.stack([pos[i] for pos in positions_log])  # Gather positions for Gaussian i
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=1, label=f"Trajectory {i}")

# Add labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Gaussian Movement Over Iterations with Trajectories")
ax.legend()

# Save the visualization
plt.savefig("logs/gaussian_movement_with_trajectories.png", dpi=300)
print("Gaussian movement plot with trajectories saved to logs/gaussian_movement_with_trajectories.png")
