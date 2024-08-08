import cupy as cp
import cv2
import matplotlib.pyplot as plt
from cuml.cluster import KMeans
from PIL import Image

# Load the image
image_path = '/mnt/data/272896DC-268D-49D1-80D0-53D6FD576CFA.png'
image = Image.open(image_path)
image = cp.array(image)

# Extract only the RGB channels (ignore the alpha channel)
rgb_image = image[:, :, :3]

# Optional: Resize image to reduce computational load
resize_factor = 0.5
resized_image = cv2.resize(cp.asnumpy(rgb_image), (int(rgb_image.shape[1] * resize_factor), int(rgb_image.shape[0] * resize_factor)))
resized_image = cp.array(resized_image)

# Reshape the image to a 2D array of pixels
rgb_pixels = resized_image.reshape(-1, 3)

# Perform K-means clustering on the new image
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(rgb_pixels)
clustered = kmeans.cluster_centers_[kmeans.labels_]

# Reshape the clustered image back to the resized image shape
clustered_image = clustered.reshape(resized_image.shape).astype(cp.uint8)

# Identify the cluster corresponding to the green color (algal bloom)
cluster_centers = kmeans.cluster_centers_

# Find the cluster index closest to the green color (algal bloom)
green = cp.array([0, 255, 0])
distances = cp.linalg.norm(cluster_centers - green, axis=1)
algal_bloom_cluster = cp.argmin(distances)

# Create the mask for the identified algal bloom cluster
algal_bloom_mask = (kmeans.labels_.reshape(resized_image.shape[:2]) == algal_bloom_cluster)

# Initialize intensity mask with zeros
algal_bloom_intensity = cp.zeros_like(kmeans.labels_, dtype=float)

# Assign intensity values based on cluster distance from the darkest cluster center
distances_to_darkest = cp.linalg.norm(rgb_pixels - cluster_centers[algal_bloom_cluster], axis=1)
algal_bloom_intensity[kmeans.labels_ == algal_bloom_cluster] = distances_to_darkest[kmeans.labels_ == algal_bloom_cluster]

# Reshape the intensity mask to the original image shape
algal_bloom_intensity_image = algal_bloom_intensity.reshape(resized_image.shape[:2])

# Apply a logarithmic function to the intensity values to emphasize strong areas and de-emphasize weak areas
adjusted_intensity_log = cp.log1p(algal_bloom_intensity_image)

# Convert the results back to NumPy arrays for plotting
resized_image_np = cp.asnumpy(resized_image)
clustered_image_np = cp.asnumpy(clustered_image)
algal_bloom_mask_np = cp.asnumpy(algal_bloom_mask)
adjusted_intensity_log_np = cp.asnumpy(adjusted_intensity_log)

# Display the original and clustered images with the algal bloom detection and heatmap with logarithmic adjustment
fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

axes[0].imshow(resized_image_np)
axes[0].set_title("Original Image")

axes[1].imshow(clustered_image_np)
axes[1].set_title("K-means Clustered Image")

axes[2].imshow(algal_bloom_mask_np, cmap='gray')
axes[2].set_title("Algal Bloom Detection")

im = axes[3].imshow(adjusted_intensity_log_np, cmap='viridis')
axes[3].set_title("Intensity Chart", pad=20)

# Adjusting colorbar to match the height of the heatmap
cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
