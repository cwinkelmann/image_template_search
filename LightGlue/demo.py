from pathlib import Path

from matplotlib import pyplot as plt

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)
images = Path("assets")

from lightglue import SIFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SIFT(max_num_keypoints=5).eval().to(device)  # load the extractor

matcher = LightGlue(features="sift").eval().to(device)




image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0075.JPG"

from PIL import Image
import numpy as np

# Load the image
# image_path = "DJI_0075.JPG"
image = Image.open(image_path)
image_np = np.array(image)  # Convert to NumPy array

# Get the dimensions of the image
height, width, channels = image_np.shape

# Calculate half of height and width for patch size
half_height = height // 2
half_width = width // 2

# Define the 4 patches by slicing the NumPy array
patch1 = image_np[0:half_height, 0:half_width]          # Top-left
patch2 = image_np[0:half_height, half_width:width]       # Top-right
patch3 = image_np[half_height:height, 0:half_width]      # Bottom-left
patch4 = image_np[half_height:height, half_width:width]  # Bottom-right

x_offsets = [0, half_width, 0, half_width]
y_offsets = [0, 0, half_height, half_height]

# Convert patches back to images and save them
Image.fromarray(patch1).save("DJI_0075_patch_1.jpg")
Image.fromarray(patch2).save("DJI_0075_patch_2.jpg")
Image.fromarray(patch3).save("DJI_0075_patch_3.jpg")
Image.fromarray(patch4).save("DJI_0075_patch_4.jpg")


image0 = load_image("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/template_source_DJI_0049.JPG__a02b70954a14ab0a16eb97ff187e44ecc9c0e8f0437a1b33ce56e5e0cce1f413__1280.jpg")

image0 = load_image \
    ("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/template_source_DJI_0049.JPG__a02b70954a14ab0a16eb97ff187e44ecc9c0e8f0437a1b33ce56e5e0cce1f413__1280.jpg")

image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0075.JPG"
image1 = load_image(image_path)

image1_1 = load_image("DJI_0075_patch_1.jpg")
image1_2 = load_image("DJI_0075_patch_2.jpg")
image1_3 = load_image("DJI_0075_patch_3.jpg")
image1_4 = load_image("DJI_0075_patch_4.jpg")

patches = [image1_1, image1_2, image1_3, image1_4]

features_kp = []
features_scales = []
features_oris = []
features_descriptors = []
features_keypoint_scores = []
features_image_size = torch.tensor([height, width])

for i, p in enumerate(patches):
    feats1 = extractor.extract(p.to(device))
    # Define the offsets
    x_offset = x_offsets[i]
    y_offset = y_offsets[i]

    # Create an offset tensor with the same shape as the coordinates tensor
    offsets = torch.tensor([x_offset, y_offset])

    features_kp.append(feats1["keypoints"] + offsets)
    features_scales.append(feats1["scales"])
    features_oris.append(feats1["oris"])
    features_descriptors.append(feats1["descriptors"])
    features_keypoint_scores.append(feats1["keypoint_scores"])


# concatenate the keypoints
feats1["keypoints"] = torch.cat(features_kp, dim=1)
feats1["scales"] = torch.cat(features_scales, dim=1)
feats1["oris"] = torch.cat(features_oris, dim=1)
feats1["descriptors"] = torch.cat(features_descriptors, dim=1)
feats1["keypoint_scores"] = torch.cat(features_keypoint_scores, dim=1)
feats1["image_size"] = features_image_size


feats0 = extractor.extract(image0.to(device))


matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)


plt.show()