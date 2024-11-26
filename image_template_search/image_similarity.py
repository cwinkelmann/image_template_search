"""

"""
import copy
import gc
import math
import random
import typing
from pathlib import Path
from typing import Tuple

import PIL.Image
import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch
from PIL import Image
from loguru import logger
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from conf.config_dataclass import CacheConfig
from image_template_search.image_rasterization import tile_large_image
from image_template_search.opencv_findobject_homography import _cached_detect_and_compute, _matcher
from image_template_search.util.HastyAnnotationV2 import ImageLabel
from image_template_search.util.util import get_similarity_cache_to_disk, get_image_dimensions
# Import your existing functions and classes
from lightglue import LightGlue
from lightglue import SIFT
from lightglue import viz2d
from lightglue.utils import load_image, rbd, Extractor





# logger = logging.getLogger(__name__)

def get_similarity_tiled(template_image: Path, image1: Path) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Get the similarity between a template image and a larger frame image by extracting features
    from non-overlapping tiles of the frame and matching them to the template features.

    :param template_image: Path to the template image.
    :param image1: Path to the frame image.
    :return: Tuple containing the similarity score, matched keypoints in the template,
             and matched keypoints in the frame image.
    """
    image_template_T = load_image(template_image)  # Shape: (C, H, W)
    image1_T = load_image(image1)  # Shape: (C, H, W)

    img_norm = image1_T / 255.0 if image1_T.max() > 1 else image1_T
    black_pixels = torch.all(img_norm < 0.1, dim=0).sum()
    white_pixels = torch.all(img_norm > 0.9, dim=0).sum()
    total_pixels = img_norm.shape[1] * img_norm.shape[2]

    if (image1_T.shape[1] > 1000 and image1_T.shape[2] > 1000 and not torch.all(image1_T == 0)
            and (black_pixels + white_pixels) / total_pixels < 0.5):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

        # Extract features from the template image
        extractor = SIFT(max_num_keypoints=5000).eval().to(device)
        extractor_s = SIFT(max_num_keypoints=500).eval().to(device)

        image_template_T = image_template_T.unsqueeze(0)  # Add batch dimension (B, C, H, W)
        feats0 = extractor.extract(image_template_T.to(device))
        # No need to remove batch dimension here

        # Prepare tiling parameters
        tile_width = image_template_T.shape[3] * 2  # Adjust as needed
        tile_height = image_template_T.shape[2] * 2  # Adjust as needed
        frame_width, frame_height = image1_T.shape[2], image1_T.shape[1]

        # Lists to store features from all tiles
        all_kpts1 = []
        all_descs1 = []
        all_scores1 = []
        all_scales1 = []
        all_oris1 = []

        # Iterate over non-overlapping tiles
        for y in range(0, frame_height, tile_height):
            for x in range(0, frame_width, tile_width):
                # Crop the tile from the frame image
                x_end = min(x + tile_width, frame_width)
                y_end = min(y + tile_height, frame_height)
                tile_image = image1_T[:, y:y_end, x:x_end]

                # Check if the tile is valid
                if tile_image.shape[1] < tile_height // 2 or tile_image.shape[2] < tile_width // 2:
                    continue  # Skip tiles that are too small

                # Add batch dimension to tile_image
                tile_image = tile_image.unsqueeze(0)  # Shape: (1, C, H, W)

                # Extract features from the tile
                feats_tile = extractor_s.extract(tile_image.to(device))
                # feats_tile is a dictionary with batch dimension

                # Adjust keypoint coordinates to global frame coordinates
                kpts_tile = feats_tile["keypoints"][0]  # Remove batch dimension
                kpts_tile[:, 0] += x  # x-coordinate
                kpts_tile[:, 1] += y  # y-coordinate

                # Append adjusted keypoints and descriptors
                all_kpts1.append(kpts_tile)
                all_descs1.append(feats_tile["descriptors"][0])  # Remove batch dimension
                all_scores1.append(feats_tile["keypoint_scores"][0])  # Remove batch dimension
                all_scales1.append(feats_tile["scales"][0])  # Remove batch dimension
                all_oris1.append(feats_tile["oris"][0])  # Remove batch dimension

        if not all_kpts1:
            logger.warning("No keypoints found in any tile of the frame.")
            return 0, torch.tensor([]), torch.tensor([])

        # Concatenate features from all tiles
        kpts1 = torch.cat(all_kpts1, dim=0).unsqueeze(0)  # Add batch dimension back
        descs1 = torch.cat(all_descs1, dim=0).unsqueeze(0)  # Add batch dimension back
        scores1 = torch.cat(all_scores1, dim=0).unsqueeze(0)  # Add batch dimension back
        scales1 = torch.cat(all_scales1, dim=0).unsqueeze(0)  # Add batch dimension back
        oris1 = torch.cat(all_oris1, dim=0).unsqueeze(0)  # Add batch dimension back

        # Create combined features dictionary
        feats1 = {
            "keypoints": kpts1,  # Shape: (1, N, 2)
            "descriptors": descs1,  # Shape: (1, N, D)
            "keypoint_scores": scores1,  # Shape: (1, N)
            "image_size": torch.tensor([[frame_width, frame_height]], dtype=torch.float32),
            # Use the last tile's image size
            "scales": scales1,  # Shape: (1, N)
            "oris": oris1,  # Shape: (1, N)
        }

        # Perform matching
        matcher = LightGlue(features="sift").eval().to(device)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        # matches01 is a dictionary with batch dimension

        # Remove batch dimension from matches
        matches = matches01["matches"][0]
        # Similarly for the confidence scores if needed

        if matches.size()[0] == 0:
            logger.warning("No matching keypoints found between template and frame.")
            return 0, torch.tensor([]), torch.tensor([])

        normalised_sim = matches.size()[0] / feats0["keypoints"].size()[1]

        # Get matched keypoints only
        m_kpts0 = feats0["keypoints"][0][matches[..., 0]]  # Shape: (M, 2)
        m_kpts1 = feats1["keypoints"][0][matches[..., 1]]  # Shape: (M, 2)

        del image1_T
        del image_template_T
        gc.collect()

        return normalised_sim, m_kpts0, m_kpts1

    else:
        logger.warning(f"Image too small or not proper: {image1_T.shape}")
        return 0, torch.tensor([]), torch.tensor([])


def extractor_wrapper(image_path: Path, max_num_keypoints=8000) -> typing.Tuple[dict, torch.Tensor]:
    """
    Extract features from an image using a given extractor.
    :param image_path: Path.
    :param extractor: Feature extractor.
    :return: Extracted features.
    """

    extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(CacheConfig.device)
    # extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(CacheConfig.device)
    image = load_image(image_path)
    feats = extractor.extract(image.to(CacheConfig.device))

    return feats, image

def matcher_wrapper(feats0, feats1, feature_type="sift") -> torch.Tensor:
    logger.info(f"Start matching")
    matcher = LightGlue(features=feature_type).eval().to(CacheConfig.device)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    logger.info(f"Done matching")

    return matches01


def visualise_matches(feats0, feats1, matches01, image0, image1):
    """

    :param feats0:
    :param feats1:
    :param matches01:
    :param image0:
    :param image1:
    :return:
    """

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    ## Display the matches
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

    plt.show()


### TODO readd this
@get_similarity_cache_to_disk()
def get_similarity(template_image: Path,
                   image1: Path) -> (float, torch.Tensor, torch.Tensor):
    """
    get the similarity between two images
    :param template_image:
    :param image1:
    :param image2:
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else CacheConfig.device)  # 'mps', 'cpu'
    torch.set_grad_enabled(False)
    # TODO get the image_size
    with Image.open(template_image) as img:
        template_width, template_height = img.size

    PIL.Image.MAX_IMAGE_PIXELS = 5223651122
    with Image.open(image1) as img:
        img1_width, img1_height = img.size
        # img = img.convert("RGB")

    max_num_keypoints = CacheConfig.max_num_keypoints

    N_x = N_y = 2  # TODO evaluate this
    if img1_width - 1000 > template_width and img1_height - 1000 > template_height:
        N_x = img1_width // template_width
        N_y = img1_height // template_height

    logger.info(f"Using {N_x} x {N_y} tiles")

    logger.info(f"START extracting features from {template_image.name}")
    feats0, _ = extractor_wrapper(image_path=template_image, max_num_keypoints=max_num_keypoints)
    logger.info(f"DONE extracting features from {template_image.name}")

    logger.info(f"START extracting features from {image1.name}")
    e = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)  # load the extractor
    # e = SuperPoint(max_num_keypoints=max_num_keypoints ).eval().to(device)  # load the extractor
    extractor = TiledExtractor(extractor=e)
    # TODO bundle this into patched extractor

    feats1 = extractor.extract(image_path=image1, N_x=N_x, N_y=N_y)
    logger.info(f"DONE extracting features from {image1.name}")

    # feats0, image1_T  = extractor_wrapper(image_path=template_image, max_num_keypoints=10000)
    # image1_T = load_image(image1)

    # matches01 = matcher_wrapper(feats0=feats0, feats1=feats1, feature_type="superpoint")
    matches01 = matcher_wrapper(feats0=feats0, feats1=feats1, feature_type="sift")

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    # TODO correct this calculation if the images have a different size
    normalised_sim = matches.size()[0] / kpts0.size()[0]

    # get matched keypoints only
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


    if CacheConfig.visualise_matching:
        ## Display the matches
        image0 = load_image(template_image)
        image1 = load_image(image1)
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

        plt.show()

    return normalised_sim, m_kpts0, m_kpts1


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
            logger.info(f"Patch tile {i}, {j}")
            y_start, y_end = i * tile_height, (i + 1) * tile_height
            x_start, x_end = j * tile_width, (j + 1) * tile_width

            patch = image_np[y_start:y_end, x_start:x_end]
            # Does the patch contain more 50% black pixels?
            # Get the total number of values in the array (H * W * 3 for RGB)
            total_pixels = patch.shape[0] * patch.shape[1]
            zero_pixels = (patch == [0, 0, 0]).all(axis=-1)
            non_zero_pixels = np.count_nonzero((patch != [0, 0, 0]).all(axis=-1))

            # Count the number of zero pixels
            zero_pixel_count = np.count_nonzero(zero_pixels)
            if zero_pixel_count > (0.9 * total_pixels):
                logger.warning(f"Patch {i}, {j} contains more than 50% black pixels")
                continue
            # Calculate and store offsets
            x_offsets.append(x_start)
            y_offsets.append(y_start)

            # TODO use a temporary directory / extract the feature directly
            patch_name = f"temp_patch_{i}_{j}.jpg"
            Image.fromarray(patch).save(patch_name)

            patches.append(patch_name)

    return patches, x_offsets, y_offsets


class TiledExtractor:
    """
    Little Wrapper to allow for a lightglue matching of very small images to larger images
    """

    def __init__(self, extractor: Extractor):
        self.extractor = extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else CacheConfig.device)  # 'mps', 'cpu'

    def combine_method(self, patches, height, width, x_offsets, y_offsets):
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
            logger.info(f"Loading {patched_image}")
            feats1 = self.extractor.extract(p.to(self.device))
            # Define the offsets
            x_offset = x_offsets[i]
            y_offset = y_offsets[i]

            # Create an offset tensor with the same shape as the coordinates tensor
            offsets = torch.tensor([x_offset, y_offset])

            features_kp.append(feats1["keypoints"] + offsets)
            if isinstance(self.extractor, SIFT):
                features_scales.append(feats1["scales"])
                features_oris.append(feats1["oris"])

            features_descriptors.append(feats1["descriptors"])
            features_keypoint_scores.append(feats1["keypoint_scores"])

        # concatenate the keypoints
        feats1["keypoints"] = torch.cat(features_kp, dim=1)
        if isinstance(self.extractor, SIFT):
            feats1["scales"] = torch.cat(features_scales, dim=1)
            feats1["oris"] = torch.cat(features_oris, dim=1)
        feats1["descriptors"] = torch.cat(features_descriptors, dim=1)
        feats1["keypoint_scores"] = torch.cat(features_keypoint_scores, dim=1)
        feats1["image_size"] = features_image_size

        return feats1

    # def extract(self, image_np, N_x, N_y):
    def extract(self, image_path, N_x, N_y):
        # Load the image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            image_np = np.array(image)
            # alpha_zero_mask = image[:, :, 3] == 0
            # # Count the non-zero values in the alpha channel
            # nonzero_alpha_count = np.count_nonzero(alpha_zero_mask)
            # image_np[alpha_zero_mask, :3] = 0
        else:
            image_np = np.array(image)  # Convert to NumPy array

        height, width, channels = image_np.shape
        assert channels == 3, "Image must have 3 channels (RGB)"
        patches, x_offsets, y_offsets = image_patcher(image_np, N_x, N_y)
        feats = self.combine_method(patches, height, width, x_offsets, y_offsets)

        return feats


def find_rotation_gen_kornia(m_kpts0: np.ndarray,
                      m_kpts1: np.ndarray,
                      image_name: typing.Union[str, Path]) -> (np.ndarray, np.ndarray, shapely.Polygon):
    """
    Find the footprint of the template in the image using kornia for homography
    WARNING! on a ARM (M1,M2,... ) macbook this consumes a lot of memory
    :param m_kpts0: Matching keypoints in the template image.
    :param m_kpts1: Matching keypoints in the target image.
    :param image_name: Path to the target image.
    :return: Homography matrix, mask of inliers, and the footprint polygon.
    """
    logger.warning("on a ARM (M1,M2,... ) macbook this consumes a lot of memory and is not practical for large images")
    # Convert image path if necessary
    if isinstance(image_name, str):
        image_name = Path(image_name)

    # Parameters
    ransac_reproj_threshold = CacheConfig.ransac_reproj_threshold
    logger.info(f"RANSAC Threshold: {CacheConfig.ransac_reproj_threshold}")

    # Convert keypoints to tensors and match shape for kornia
    kpts0 = torch.from_numpy(m_kpts0).float().unsqueeze(0)
    kpts1 = torch.from_numpy(m_kpts1).float().unsqueeze(0)

    # Find homography using kornia
    M = K.geometry.find_homography_dlt(kpts0, kpts1).squeeze().numpy()

    w, h = get_image_dimensions(image_name)
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Transform the footprint points using the homography
    footprint = cv2.perspectiveTransform(pts, M)
    footprint = np.int32(footprint.reshape(4, 2))
    footprint_polygon = Polygon(footprint.reshape(4, 2))

    # Kornia does not return a mask, so setting it to None
    mask = None

    return M, mask, footprint_polygon


def find_rotation_gen_cv2(m_kpts0: np.ndarray,
                          m_kpts1: np.ndarray,
                          image_name: str | Path) -> (np.ndarray, np.ndarray, shapely.Polygon):
    """
    find the footprint of the template in the image
    :param m_kpts0:
    :param m_kpts1:
    :param image_name:
    :return:
    """

    if isinstance(image_name, str):
        image_name = Path(image_name)
    logger.info(f"RANSAC Threshold: {CacheConfig.ransac_reproj_threshold}")
    M, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, CacheConfig.ransac_reproj_threshold)

    # TODO integrate kornia findHomography in here
    img1 = cv2.imread(str(image_name), cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    footprint = cv2.perspectiveTransform(pts, M)
    footprint = np.int32(footprint.reshape(4, 2))
    footprint = shapely.Polygon(footprint.reshape(4, 2))

    return M, mask, footprint


def find_patch(template_path: Path,
               large_image_path: Path,
               output_path=Path("./output")):
    """
    Find the template in the large image using LightGlue https://github.com/cvg/LightGlue and SIFT
    TODO: when the template is too small it is not working well. There is no method of identifying if a match is right or not
    :param template_path:
    :param large_image_path:
    :param output_path:
    :return:
    """

    normalised_sim, m_kpts0, m_kpts1 = get_similarity(template_path, Path(large_image_path))
    # normalised_sim, m_kpts0, m_kpts1 = get_similarity_tiled(template_path, Path(large_image_path))
    logger.info(f"normalised_sim: {normalised_sim}")

    fx = 1  # TODO clarify if this is needed
    fy = 1

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    template_identifier = template_path.stem
    large_image_identifier = large_image_path.stem

    M, mask, footprint = find_rotation_gen_cv2(m_kpts0.cpu().numpy(),
                                           m_kpts1.cpu().numpy(),
                                           image_name=large_image_path)

    img1 = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)  # train
    img1 = cv2.resize(img1, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1,
                                                                               2)  # This is the box of the query image

    dst = cv2.perspectiveTransform(pts, M)

    new_image_footprint = dst
    # draw the matches outer box
    footprint = np.int32(dst.reshape(4, 2))
    ## TODO create the footprint from that

    ## plot the footprint of the template on the base image
    footprint = shapely.Polygon(footprint.reshape(4, 2))
    bounds = footprint.bounds

    fig = plt.figure(figsize=(20, 20))
    large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
    large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

    large_image_l = cv2.polylines(large_image, [np.int32(dst)], True, 255, 53, cv2.LINE_AA)
    large_image_l = cv2.resize(large_image_l, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    plt.imshow(large_image_l, 'gray')
    fig.savefig(output_path / f"t_{template_identifier}_b{large_image_identifier}_large_image_footprint.jpg")
    plt.show()

    # TODO class variable
    theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
    print(f"The camera rotated: {round(theta, 2)} degrees")

    # TODO class variable
    M_ = np.linalg.inv(M)
    M_

    large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
    large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
    template_image = cv2.imread(str(template_path))
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)

    rotated_cropped_image_bbox = cv2.warpPerspective(large_image, M_,
                                                     (template_image.shape[1], template_image.shape[0]))

    fig, axes = plt.subplots(1, sharey=True, figsize=(13, 12))
    # Display the result
    plt.imshow(rotated_cropped_image_bbox)
    # plt.axis('off')  # Hide axis
    plt.show()
    rotated_cropped_image_bbox_path = output_path / f"t_{template_identifier}_b{large_image_identifier}_rotated_cropped_image_bbox.jpg"
    cv2.imwrite(str(rotated_cropped_image_bbox_path), cv2.cvtColor(rotated_cropped_image_bbox, cv2.COLOR_RGB2BGR))

    return rotated_cropped_image_bbox, footprint





def find_patch_stacked(template_path, large_image_paths, output_path,
                       tile_path, cache_path, MIN_MATCH_COUNT=50,
                       tile_size_x=1500, tile_size_y=1500,
                       visualise=False):
    """
    find a crop in multiple other images

    :param template_path:
    :param large_image_paths:
    :param output_path:
    :param cache_path:
    :return:
    """

    ## TODO use at least multiprocessing or dask to parallize this
    crops = []

    for large_image_path in large_image_paths:
        logger.info(f"finding patch in {large_image_path}")
        crop = find_patch_tiled(template_path,
                                large_image_path,
                                output_path=output_path,
                                cache_path=cache_path,
                                MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                                tile_size_x=tile_size_x, tile_size_y=tile_size_y, visualise=False)

        if isinstance(crop, np.ndarray):
            crops.append(crop)
            im = PIL.Image.fromarray(crop)
            im.save(output_path / f"crop_{large_image_path.stem}_t_{template_path.stem}.jpeg")
        else:
            # no match
            pass

    return crops


def project_bounding_box(label: typing.Union[Polygon, ImageLabel], M: np.ndarray) -> typing.Union[Polygon, ImageLabel]:
    """
    Project a Shapely bounding box from Image A to Image B using the cv2 perspectiveTransform function.
    # TODO project this so we can project points, lines and polygons
    :param bbox_a: Bounding box in Image A (as a Shapely Polygon)
    :param M: 3x3 perspective transformation matrix
    :return: Transformed bounding box in Image B (as a Shapely Polygon)
    """
    # Extract the bounding box coordinates (minx, miny, maxx, maxy)
    if isinstance(label, ImageLabel):
        bbox_a = label.bbox_polygon
    else:
        bbox_a = label

    # TODO use the corners directly if the object is rectangular
    minx, miny, maxx, maxy = bbox_a.bounds

    # Define the four corners of the bounding box in Image A
    corners_a = np.array([
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy]
    ], dtype=np.float32).reshape(-1, 1, 2)  # Reshape for cv2.perspectiveTransform (Nx1x2)

    # Apply the perspective transformation to the corners
    corners_b = cv2.perspectiveTransform(corners_a, M)

    # Convert the transformed points back to a list of tuples (x, y)
    transformed_corners = corners_b.reshape(-1, 2)

    # Create a new polygon with the transformed corners in Image B
    bbox_b = Polygon(transformed_corners)

    if isinstance(label, ImageLabel):
        label.bbox_polygon = bbox_b
        return label

    return bbox_b


def project_annotations_to_crop(buffer: shapely.Polygon,
                                imagelabels: list[ImageLabel]):
    """
    TODO this is pretty much the code in 'project_bounding_box' but with a different signature
    :param pc:
    :param patch_size:
    :param image:
    :return:
    """
    # create a buffer around the centroid of the polygon
    assert isinstance(buffer, shapely.Polygon)
    assert all([isinstance(il, ImageLabel) for il in imagelabels])

    minx, miny, maxx, maxy = buffer.bounds

    obj_in_crop = [copy.copy(il) for il in imagelabels if il.centroid.within(buffer)]  # all objects withing the buffer
    cropped_annotations = [l for l in obj_in_crop if buffer.contains(l.centroid)]

    a, b, d, e = 1.0, 0.0, 0.0, 1.0  # Scale and rotate
    xoff, yoff = -minx, -miny  # Translation offsets

    # Apply the affine transformation to the polygon to reproject into image coordinates
    transformation_matrix = [a, b, d, e, xoff, yoff]

    for ca in cropped_annotations:
        ca.bbox_polygon = affine_transform(ca.bbox_polygon, transformation_matrix)

    return cropped_annotations, buffer


class ImagePatchFinderLG(object):
    """
    find a patch on an image in another image and calculate the homography between the two
    """

    footprint: shapely.Polygon
    template_path: Path
    template_polygon: shapely.Polygon
    proj_template_polygon: shapely.Polygon
    large_image_path: Path

    large_image: np.ndarray
    large_image_shape: tuple
    warped_image_B: np.ndarray

    M_: np.ndarray
    M: np.ndarray
    mask: np.ndarray
    theta: float

    def __init__(self, template_path, large_image_path):
        """

        :param template_path: The image which is to be found in the large image
        :param template_polygon: The footprint of the template image
        :param large_image_path:
        """
        self.template_image = None
        self.warped_image_B = None
        self.proj_template_polygon = None
        self.footprint = None
        self.M_ = None
        self.M = None
        self.mask = None

        self.template_path = template_path

        width, height = get_image_dimensions(template_path)
        template_polygon = Polygon([(0, 0), (width, 0), (width, height), (0, height)])

        self.template_polygon = template_polygon
        self.large_image_path = large_image_path
        self.matched_templates = []

    def __call__(self, *args, **kwargs):
        return self.find_patch()

    def find_patch(self,
                   output_path=Path("./output"),
                   similarity_threshold=0.05):
        """
        Find the template in the large image using LightGlue https://github.com/cvg/LightGlue and SIFT
        TODO: when the template is too small it is not working well. There is no method of identifying if a match is right or not
        :param similarity_threshold:
        :param template_path:
        :param large_image_path:
        :param output_path:
        :return:
        """
        normalised_sim, m_kpts0, m_kpts1 = get_similarity(self.template_path,
                                                          Path(self.large_image_path))

        if len(m_kpts0) < 10 or len(m_kpts1) < 10:
            logger.warning(f"The template {self.template_path.stem} is not in the image {self.large_image_path.stem}")
            logger.info(f"normalised_sim: {normalised_sim}, len(m_kpts0): {len(m_kpts0)}, len(m_kpts1): {len(m_kpts1)}")

            raise ValueError(f"The template {self.template_path.stem} is not in the image {self.large_image_path.stem}")

        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        logger.info(f"normalised_sim: {normalised_sim}, len(m_kpts0): {len(m_kpts0)}, len(m_kpts1): {len(m_kpts1)}")
        M, mask, footprint = find_rotation_gen_cv2(m_kpts0.cpu().numpy(),
                                               m_kpts1.cpu().numpy(),
                                               image_name=self.large_image_path)

        self.M = M
        self.M_ = np.linalg.inv(M)
        self.mask = mask

        self.proj_template_polygon = project_bounding_box(self.template_polygon, M)

        self.footprint = footprint

        # calculate the rotation of the camera
        self.theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
        logger.info(f"The camera rotated: {round(self.theta, 2)} by degrees")

        return True

    def project_image(self, output_path):
        """
        Project large image to the footprint of the template image
        :return:
        """

        template_image = cv2.imread(str(self.template_path))
        self.template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)

        large_image = cv2.imread(str(self.large_image_path), cv2.IMREAD_COLOR)
        self.large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
        self.large_image_shape = self.large_image.shape

        # warp the other large image to the template image
        self.warped_image_B = cv2.warpPerspective(self.large_image,
                                                  self.M_,
                                                  dsize=(
                                                      self.template_image.shape[1], self.template_image.shape[0]))

        matched_source_image = f"warped_source_{self.template_path.stem}_match_{self.large_image_path.stem}.jpg"

        Path(output_path).mkdir(exist_ok=True, parents=True)
        warped_other_image_path = output_path / matched_source_image
        cv2.imwrite(str(warped_other_image_path), cv2.cvtColor(self.warped_image_B, cv2.COLOR_RGB2BGR))

        return warped_other_image_path
