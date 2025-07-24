import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from transformers import CLIPModel, CLIPProcessor

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Move model to GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Input: Image and Sentence
image_path = "bacchus-and-ariadne.jpg"  # Replace with your image path
text_input = "Bacchus' chariot is normally drawn by tigers or panthers, but Alfonso d'Este is known to have had a menagerie at the palace in which he kept a cheetah or a cheetah-like member of the cat family."  # Replace with your sentence
threshold = 0.7  # Adjust based on experimentation
image = Image.open(image_path)

# Preprocess the inputs
inputs = processor(text=[text_input], images=image, return_tensors="pt", padding=True)
inputs = {key: value.to(device) for key, value in inputs.items()}


### Bargraph for text
# Hook function to capture gradients
gradients = None


def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]  # Save the gradients for text embeddings


# Register the hook on the text embeddings
hook_handle = model.text_model.embeddings.register_full_backward_hook(save_gradients)

# Enable gradient tracking
inputs["pixel_values"].requires_grad_(True)

# Forward pass to compute logits
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image

# Compute the gradient of the logits with respect to the image
logit_for_text = logits_per_image[
    0, 0
]  # Select the logit corresponding to the input text
logit_for_text.backward()

# Process the captured gradients
# print(f"Gradients shape: {gradients.shape}")  # Debug: Check gradients shape

# Compute the L2 norm for each token's embedding gradient
token_gradients = gradients.norm(dim=-1).cpu().detach().numpy()

# Ensure token_gradients is a 1D array
if token_gradients.ndim > 1:
    token_gradients = token_gradients.flatten()

# Debug: Print the shape of token_gradients
# print(f"Token Gradients shape: {token_gradients.shape}")

# Map token IDs back to words
tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Verify the number of tokens matches the gradient scores
assert len(tokens) == len(
    token_gradients
), "Mismatch between token count and gradient scores"

# Aggregate contributions for subwords
aggregated_contributions = {}
current_word = ""
for token, grad in zip(tokens, token_gradients):
    if token == "<|startoftext|>" or token == "<|endoftext|>":  # Skip special tokens
        continue
    if token.endswith("</w>"):  # Full word detected (marked by </w>)
        if current_word:  # If building a word, append and finalize
            current_word += token.replace("</w>", "")
            aggregated_contributions[current_word] = (
                aggregated_contributions.get(current_word, 0) + grad
            )
            current_word = ""  # Reset current word
        else:  # Standalone word
            word = token.replace("</w>", "")
            aggregated_contributions[word] = (
                aggregated_contributions.get(word, 0) + grad
            )
    else:  # Subword token without a word boundary
        current_word += token  # Append to the current word

# Handle any remaining subword
if current_word:
    aggregated_contributions[current_word] = aggregated_contributions.get(
        current_word, 0
    )

# Prepare data for plotting
aggregated_words = list(aggregated_contributions.keys())
aggregated_gradients = list(aggregated_contributions.values())

# Debug: Check aggregated words and gradients
# print(f"Aggregated Words: {aggregated_words}")
# print(f"Aggregated Gradients: {aggregated_gradients}")
# Clean up: Remove the hook
hook_handle.remove()


### Heatmap for image
# Extract gradients and compute attention map
grads = inputs["pixel_values"].grad  # Gradients with respect to pixel values
grads = grads.mean(dim=1).squeeze().cpu().numpy()  # Average across channels

# Apply logarithmic scaling to enhance contrast
grads = np.log1p(grads)  # Use natural logarithm for better contrast
grads -= grads.min()
grads /= grads.max()

# Apply thresholding to emphasize high-attention areas
grads[grads < threshold] = 0

# Compute dynamic grid size for gradients
num_patches = grads.size  # Total number of patches
grid_size = int(np.sqrt(num_patches))  # Grid size (assuming square grid)

if grid_size**2 != num_patches:
    raise ValueError(f"Gradient size {num_patches} is not a perfect square.")

# Reshape gradients to match the grid size
grads_resized = grads.reshape((grid_size, grid_size))

# Apply Gaussian blur to smooth the gradients
grads_smoothed = gaussian_filter(grads_resized, sigma=2)  # Adjust sigma for smoothing

# Upscale gradient grid for finer details
grads_upscaled = zoom(grads_smoothed, zoom=32)  # Higher zoom factor for more detail
grads_upscaled -= grads_upscaled.min()
grads_upscaled /= grads_upscaled.max()

# Resize the image to match the upscaled gradient
image_resized = image.resize(
    grads_upscaled.shape[::-1]
)  # Reverse shape for (width, height)


# Create a GridSpec layout with a 2x3 grid
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], figure=fig)

# Plot 1: Aggregated Word Contributions (spanning all three columns on the top row)
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(range(len(aggregated_words)), aggregated_gradients, color="skyblue")
ax1.set_xticks(range(len(aggregated_words)))
ax1.set_xticklabels(aggregated_words, rotation=45, ha="right", fontsize=10)
ax1.set_title("Word Contributions Based on Gradient Magnitudes")
ax1.set_xlabel("Words")
ax1.set_ylabel("Gradient Magnitude")
ax1.grid(axis="y")

# Plot 2: Original Image (bottom left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(image_resized)
ax2.set_title("Original Image")
ax2.axis("off")

# Plot 3: Overlay of Heat Map on Image (bottom middle)
ax3 = fig.add_subplot(gs[1, 1])
alpha_map = (
    grads_upscaled / grads_upscaled.max()
)  # Transparency based on attention values
ax3.imshow(image_resized, alpha=0.6)
ax3.imshow(grads_upscaled, cmap="viridis", alpha=0.4)
ax3.set_title("Heat Map Overlay")
ax3.axis("off")

# Plot 4: Heat Map Alone (bottom right)
ax4 = fig.add_subplot(gs[1, 2])
heatmap = ax4.imshow(grads_upscaled, cmap="viridis")
ax4.set_title("Heat Map Only")
ax4.axis("off")

# Add a color bar to the last subplot
cbar = fig.colorbar(heatmap, ax=ax4, fraction=0.046, pad=0.04, label="grads_upscaled")

# Adjust layout to ensure no overlaps
plt.subplots_adjust(hspace=0.4, wspace=0.1)

# Generate a unique filename using the image name, first three words of the text input, and current time (HH-MM-SS)
base_name = image_path.split("/")[-1].split(".")[
    0
]  # Extract the base name of the image file
text_start = "_".join(
    text_input.split()[:3]
)  # Get the first three words of the text input
timestamp = time.strftime("%H-%M-%S")  # Get the current time in HH-MM-SS format
output_path = f"{base_name}_{text_start}_{timestamp}_combined.png"

# Save the figure
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)

print(f"Results saved to {output_path}")
