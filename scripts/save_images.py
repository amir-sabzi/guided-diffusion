import os
import numpy as np
import matplotlib.pyplot as plt

# Set the source and destination directories
src_dir = "/tmp/openai-2024-06-04-13-54-32-042762"
dst_dir = os.path.join("/home/sabzi/workspace/guided-diffusion/saved_images", os.path.basename(src_dir))



# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Load the samples from the .npz file
samples_file = os.path.join(src_dir, "samples_10x64x64x3.npz")
samples = np.load(samples_file)["arr_0"]

# Plot and save each sample
for i, sample in enumerate(samples):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the sample
    ax.imshow(sample)
    ax.axis("off")

    # Save the plot
    plot_file = os.path.join(dst_dir, f"sample_{i}.png")
    fig.savefig(plot_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

print(f"Plots saved to {dst_dir}")