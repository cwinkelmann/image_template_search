from pathlib import Path

from matplotlib import pyplot as plt

from demo import feats1
# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from lightglue import SIFT
from PIL import Image
import numpy as np

def image_patcher(image_np: np.array, N_x: int, N_y: int):
    # Get the dimensions of the image
    height, width, channels = image_np.shape

    # Calculate tile sizes
    tile_height = height // N_y
    tile_width = width // N_x

    patches = []
    x_offsets = []
    y_offsets = []

    for i in range(N_y):
        for j in range(N_x):
            # Calculate coordinates for each tile
            y_start, y_end = i * tile_height, (i + 1) * tile_height
            x_start, x_end = j * tile_width, (j + 1) * tile_width

            patch = image_np[y_start:y_end, x_start:x_end]
            # patches.append(torch.from_numpy(patch).float())

            # Calculate and store offsets
            x_offsets.append(x_start)
            y_offsets.append(y_start)

            # TODO use a temporary diretory / extract the feature directly
            patch_name = f"DJI_0075_patch_{i}_{j}.jpg"
            Image.fromarray(patch).save(patch_name)

            patches.append(patch_name)

    return patches, x_offsets, y_offsets

def combine_method(patches, height, width, x_offsets, y_offsets):
    """ Combine the patches

    Args:
        :param patches:
        :param height:
        :param width:
        :param x_offsets:
        :param y_offsets:
        :return:
    """
    feats1 = dict()

    features_kp = []
    features_scales = []
    features_oris = []
    features_descriptors = []
    features_keypoint_scores = []
    features_image_size = torch.tensor([height, width])

    for i, patched_image in enumerate(patches):
        p = load_image(patched_image)
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

    return feats1

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    images = Path("assets")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SIFT(max_num_keypoints=6096).eval().to(device)  # load the extractor
    matcher = LightGlue(features="sift").eval().to(device)


    N_x, N_y = 2, 2

    image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0075.JPG"
    # Load the image
    # image_path = "DJI_0075.JPG"
    image = Image.open(image_path)
    image_np = np.array(image)  # Convert to NumPy array
    height, width, channels = image_np.shape

    # TODO bundle this into patchted extractor
    patches, x_offsets, y_offsets = image_patcher(image_np, N_x, N_y)
    feats1 = combine_method(patches, height, width, x_offsets, y_offsets)

    # extracting the features
    image0 = load_image("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/template_source_DJI_0049.JPG__a02b70954a14ab0a16eb97ff187e44ecc9c0e8f0437a1b33ce56e5e0cce1f413__1280.jpg")
    feats0 = extractor.extract(image0.to(device))
    image1 = load_image(image_path)




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