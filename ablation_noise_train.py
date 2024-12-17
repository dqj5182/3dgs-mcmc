import os
import matplotlib.pyplot as plt

# Define noise levels to test
noise_levels = [0.1, 0.5, 1.0, 1.5, 2.0]
train_psnrs = []
test_psnrs = []

# Run train.py with different noise levels
for noise in noise_levels:
    print(f"Running train.py with noise level: {noise}")
    os.system(f"python train.py --source_path datasets/tandt/train --config configs/train.json --eval --noise_level {noise}")
    
    # Parse the PSNR results from the standardized log file
    log_file = f"logs/psnr_results_noise_{noise:.2f}.txt"

    with open(log_file, "r") as f:
        lines = f.readlines()
        test_psnr = float(lines[2].split(":")[1].strip())  # Test PSNR
        train_psnr = float(lines[3].split(":")[1].strip())  # Train PSNR
        test_psnrs.append(test_psnr)
        train_psnrs.append(train_psnr)

# Plot and save the PSNR graph
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, train_psnrs, marker='o', label="Train PSNR")
plt.plot(noise_levels, test_psnrs, marker='s', label="Test PSNR")
plt.xlabel("Noise Level")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Noise Level (3DGS-MCMC)")
plt.legend()
plt.grid()
plt.savefig("psnr_vs_noise_level.png")
print("Graph saved as 'psnr_vs_noise_level.png'")
