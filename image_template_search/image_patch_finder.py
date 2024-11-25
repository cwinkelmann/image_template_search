import gc
import math
import random
import typing
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from loguru import logger
from shapely.geometry import Polygon

from image_template_search.image_rasterization import tile_large_image
from image_template_search.image_similarity import project_bounding_box
from image_template_search.opencv_findobject_homography import _cached_detect_and_compute, _matcher
from image_template_search.util.util import get_image_dimensions


# Import your existing functions and classes


def _cached_tiled_keypoints_and_descriptors_extraction(detector,
                                                       tile_size_x,
                                                       tile_size_y,
                                                       overlap_x,
                                                       overlap_y,
                                                       tile_base_path,
                                                       cache_path: Path = None,
                                                       large_image: np.ndarray = None,
                                                       large_image_path: Path = None,
                                                       ):
    """
    Extract keypoints and descriptors from a large image by tiling it
    :param detector:
    :param large_image:
    :param tile_size_x:
    :param tile_size_y:
    :param overlap_x:
    :param overlap_y:
    :param tile_base_path:
    :param cache_path:
    :return:
    """

    assert large_image is not None or large_image_path is not None, "Either large_image or large_image_path must be provided"
    if large_image_path is not None:
        logger.info(f"Loading large image from {large_image_path}")
        large_image = Image.open(large_image_path).convert("L")
        large_image = np.array(large_image)
    kp2s = []
    des2s = []
    tiles = []

    for y in range(0, large_image.shape[0], tile_size_y - overlap_y):
        for x in range(0, large_image.shape[1], tile_size_x - overlap_x):
            try:
                tile_path = tile_large_image(x, y, tile_size_x, tile_size_y,
                                             large_image,
                                             tile_base_path,
                                             prefix=large_image_path.stem)
                tiles.append((tile_path, x, y))
            except RuntimeError as e:
                logger.error(f"error processing tile: {x}, {y}, {e}")

    del large_image

    for tile_path, x, y in tiles:

        img2 = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)

        kp2, des2 = _cached_detect_and_compute(detector, img=img2,
                                               img_path=tile_path, cache_path=cache_path)

        # we need to modify the points to match the original image
        kp_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]

        # add the global coordinates offset to the keypoints location
        kp2 = [cv2.KeyPoint(x=pt[0] + x, y=pt[1] + y, size=pt[2], angle=pt[3], response=pt[4], octave=pt[5],
                            class_id=pt[6]) for pt in kp_list]
        logger.info(f"Found {len(kp2)} keypoints in tile x:{x} y:{y}")
        # the keypoints and descriptors are stored in a list
        if len(kp2) > 0:
            kp2s.append(kp2)
        if des2 is not None:
            des2s.append(des2)
        gc.collect()

    # assemble all keypoints and descriptors from the tiles
    kp2 = [item for sublist in kp2s for item in sublist]
    des2 = [item for sublist in des2s for item in sublist]

    del (des2s)
    del (kp2s)
    gc.collect()

    return kp2, des2


def project_image(M, template_path, large_image_path, output_path,
                  visualise=False,
                  buffer=0) -> Path:
    """
    :param M:
    :param template_path:
    :param large_image_path:
    :param output_path:
    :param visualise:
    :return:
    """
    # TODO class variable
    M_ = np.linalg.inv(M)

    template_image = Image.open(
        template_path)  # Replace with your image file path
    if template_image.mode != "RGB":
        template_image = template_image.convert("RGB")
    template_image = np.array(template_image)

    large_image = Image.open(large_image_path)  # Replace with your image file path
    if large_image.mode != "RGB":
        large_image = large_image.convert("RGB")
    large_image = np.array(large_image)

    rotated_cropped_image_bbox = cv2.warpPerspective(large_image, M_,
                                                     (template_image.shape[1]+buffer, template_image.shape[0]+buffer))
    if visualise:
        fig, axes = plt.subplots(1, sharey=True, figsize=(13, 12))
        # Display the result
        plt.imshow(rotated_cropped_image_bbox)
        # plt.axis('off')  # Hide axis
        plt.show()
    rotated_cropped_image_bbox_path = output_path / f"mt_{template_path.stem}_{large_image_path.stem}.jpg"
    cv2.imwrite(str(rotated_cropped_image_bbox_path), cv2.cvtColor(rotated_cropped_image_bbox, cv2.COLOR_RGB2BGR))

    return rotated_cropped_image_bbox_path


def find_patch_tiled(template_path: Path,
                     large_image_path: Path,
                     output_path=Path("./output"),
                     tile_size_x=1200,
                     tile_size_y=1200,
                     tile_base_path=Path("./"),
                     cache_path=Path("./cache"),
                     MIN_MATCH_COUNT=50, visualise=False) -> typing.Tuple[np.ndarray, np.array]:
    """
    TODO refactor the parameters
    TODO tile from the center as well
    Find the template in the large image by tiling the large image, then iterating through the tiles

    :param tile_size_y:
    :param tile_size_x:
    :param template_path:
    :param large_image_path:
    :param output_path:
    :param tile_base_path:
    :param cache_path:
    :return:
    """

    tile_base_path.mkdir(exist_ok=True, parents=True)
    if cache_path is not None:
        cache_path.mkdir(exist_ok=True, parents=True)
    output_path.mkdir(exist_ok=True, parents=True)

    fx = 1
    fy = 1
    # keep 70% of the keypoints and descriptors
    keepers = 100

    # SIFT Based Feature Extractor
    detector = cv2.SIFT_create()

    overlap_x = 0
    overlap_y = 0

    # get the keypoints and descriptors of the template image
    template_image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)  # train
    template_image = cv2.resize(template_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    kp1, des1 = _cached_detect_and_compute(detector, img=template_image,
                                           img_path=template_path,
                                           cache_path=cache_path)

    kp2, des2 = _cached_tiled_keypoints_and_descriptors_extraction(detector,
                                                                   tile_size_x,
                                                                   tile_size_y,
                                                                   overlap_x,
                                                                   overlap_y,
                                                                   tile_base_path,
                                                                   large_image_path=large_image_path,
                                                                   cache_path=cache_path)

    gc.collect()

    ### match the keypoints and descriptors of both images

    num_to_keep = len(kp2) // keepers
    indices_to_keep = sorted(random.sample(range(len(kp2)), num_to_keep))

    kp2 = [kp2[i] for i in indices_to_keep]
    des2 = [des2[i] for i in indices_to_keep]
    des2 = np.array(des2)

    if len(kp1) > 10 and len(kp2) > 10:
        ## SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = _matcher(flann, des1, des2, template_path, Path(large_image_path), k=2)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
    except Exception as e:
        logger.error("Not enough matches are found - {}".format(len(matches)))

    if len(good) > MIN_MATCH_COUNT:
        logger.info(f"{len(good)} good matches found.")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        ## we take the shape of the first image which was transformed
        ## then we create a box around the transformed image and project it new image
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1,
                                                                                   2)  # This is the box of the query image

        dst = cv2.perspectiveTransform(pts, M)

        new_image_footprint = dst
        # draw the matches outer box
        footprint = np.int32(dst.reshape(4, 2))

        if False:
            ## TODO create the footprint from that
            footprint = shapely.Polygon(footprint.reshape(4, 2))

            fig = plt.figure(figsize=(20, 20))
            # large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
            # large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

            large_image = Image.open(
                large_image_path)  # Replace with your image file path
            large_image = np.array(large_image)

            large_image = cv2.polylines(large_image, [np.int32(dst)], True, 255, 53, cv2.LINE_AA)
            large_image = cv2.resize(large_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

            plt.imshow(large_image, 'gray')
            fig.savefig(output_path / f"{large_image_path}_large_image_footprint.jpg")  # TODO give it a good name
            plt.show()

            draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                               singlePointColor=None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            fig = plt.figure(figsize=(20, 20))
            img_matches = cv2.drawMatches(template_image, kp1, large_image, kp2, good, None, **draw_params)
            plt.imshow(img_matches, 'gray')
            fig.savefig(output_path / f"{large_image_path}_large_image_match.jpg")
            plt.show()

        theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
        print(f"The camera rotated: {round(theta, 2)} degrees")

        return M, footprint



    else:
        logger.error("Not enough good matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        return False


class ImagePatchFinderCV(object):
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
        self.cache_path = None

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
                   similarity_threshold=0.1):
        """
        Find the template in the large image using OpenCV with SIFT and a FLANN Matcher
        :param template_path:
        :param large_image_path:
        :param output_path:
        :return:
        """

        M, footprint = find_patch_tiled(
            template_path=self.template_path,
            large_image_path=self.large_image_path,
            output_path=output_path,
            tile_size_x=6000,
            tile_size_y=4000,
            tile_base_path=output_path / "tiles",
            cache_path=self.cache_path,
            MIN_MATCH_COUNT=50, visualise=False
        )

        self.M = M
        self.M_ = np.linalg.inv(M)

        self.proj_template_polygon = project_bounding_box(self.template_polygon, M)

        self.footprint = footprint

        # calculate the rotation of the camera
        self.theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
        logger.info(f"The camera rotated: {round(self.theta, 2)} by degrees")

        return True
