import time
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
from lightglue import LightGlue, SIFT, viz2d
from lightglue.utils import load_image, rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the feature extractor and matcher
extractor = SIFT(max_num_keypoints=4096).eval().to(device)
matcher = LightGlue(features="sift").eval().to(device)

# Parameters for tiling
N_x, N_y = 1, 1  # Number of tiles along x and y axes (for example, a 2x2 grid)
image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0075.JPG"  # Replace with the path to your image

# Load and split the image
image = Image.open(image_path)
image_np = np.array(image)
height, width, channels = image_np.shape

# Calculate tile sizes
tile_height = height // N_y
tile_width = width // N_x

# Create patches and offsets
patches = []
x_offsets = []
y_offsets = []

for i in range(N_y):
    for j in range(N_x):
        # Calculate coordinates for each tile
        y_start, y_end = i * tile_height, (i + 1) * tile_height
        x_start, x_end = j * tile_width, (j + 1) * tile_width

        # Extract the patch and save it to the list
        patch = image_np[y_start:y_end, x_start:x_end]
        patches.append(torch.from_numpy(patch).float().permute(2, 0, 1))
        # patches.append(torch.from_numpy(patch).float())

        # Calculate and store offsets
        x_offsets.append(x_start)
        y_offsets.append(y_start)

# Perform feature extraction and adjust coordinates
features_kp = []
features_scales = []
features_oris = []
features_descriptors = []
features_keypoint_scores = []
features_image_size = torch.tensor([height, width])

for i, patch in enumerate(patches):
    # Extract features for each patch
    feats = extractor.extract(patch.to(device))

    # Adjust keypoints with offsets
    x_offset = x_offsets[i]
    y_offset = y_offsets[i]
    offset_tensor = torch.tensor([x_offset, y_offset]).to(device)

    # Apply offsets and collect features
    features_kp.append(feats["keypoints"] + offset_tensor)
    features_scales.append(feats["scales"])
    features_oris.append(feats["oris"])
    features_descriptors.append(feats["descriptors"])
    features_keypoint_scores.append(feats["keypoint_scores"])

# Concatenate all features from patches
combined_features = {
    "keypoints": torch.cat(features_kp, dim=1),
    "scales": torch.cat(features_scales, dim=1),
    "oris": torch.cat(features_oris, dim=1),
    "descriptors": torch.cat(features_descriptors, dim=1),
    "keypoint_scores": torch.cat(features_keypoint_scores, dim=1),
    "image_size": features_image_size
}

# Load a second image for matching
image0_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/template_source_DJI_0049.JPG__a02b70954a14ab0a16eb97ff187e44ecc9c0e8f0437a1b33ce56e5e0cce1f413__1280.jpg"  # Replace with the path to your second image
image0 = load_image(image0_path).to(device)
feats0 = extractor.extract(image0)

# Perform matching between the full-image features and combined patch features
matches01 = matcher({"image0": feats0, "image1": combined_features})
feats0, combined_features, matches01 = [
    rbd(x) for x in [feats0, combined_features, matches01]
]

# Visualize the results
kpts0, kpts1, matches = feats0["keypoints"], combined_features["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

image = load_image(image_path)
axes = viz2d.plot_images([image0, image])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

plt.show()

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

plt.show()

time.sleep(1)