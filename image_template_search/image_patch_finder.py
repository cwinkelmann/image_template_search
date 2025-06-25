import gc
import math
import random
import typing
from pathlib import Path

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.geometry import Polygon

from conf.config_dataclass import CacheConfig
from image_template_search.image_rasterization import tile_large_image
from image_template_search.image_similarity import get_similarity, find_rotation_gen_cv2
from image_template_search.util.projection import project_bounding_box
from image_template_search.opencv_findobject_homography import _cached_detect_and_compute, _matcher
from image_template_search.util.util import get_image_dimensions
from image_template_search.types.exceptions import NoMatchError, DetailedNoMatchError


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

        if visualise:
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
        logger.error(f"Not enough good matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        raise DetailedNoMatchError(f"Not enough good matches are found - {len(good)}/{MIN_MATCH_COUNT}", large_image_path=large_image_path, template_path=template_path)


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
                   similarity_threshold=0.05) :
        """
        Find the template in the large image using LightGlue https://github.com/cvg/LightGlue and SIFT
        when the template is too small it is not working well. There is no method of identifying if a match is right or not
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

            raise NoMatchError(f"The template {self.template_path.stem} is not in the image {self.large_image_path.stem}")

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

        return self.proj_template_polygon

    def project_image(self, output_path) -> Path:
        """
        Project large image to the footprint of the template image
        :return:
        """
        assert self.template_path.exists()
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
    logger.warning("This method is deprecated. Use ImagePatchFinderLG or ImageFinderCV instead")
    normalised_sim, m_kpts0, m_kpts1 = get_similarity(template_path,
                                                      Path(large_image_path),
                                                        max_num_keypoints=CacheConfig.max_num_keypoints)

    logger.info(f"normalised_sim: {normalised_sim}")

    if normalised_sim > 0.15:
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
    else:
        logger.error(f"Template {template_path} not found in {large_image_path}")
        raise DetailedNoMatchError("Template not in image.", template_path, large_image_path)


def find_patch_stacked(template_path,
                       large_image_paths,
                       output_path):
    """
    find a crop in multiple other images.

    :param template_path:
    :param large_image_paths:
    :param output_path:
    :param cache_path:
    :return:
    """

    ## TODO use at least multiprocessing or dask to parallize this
    crops = []
    failed_images = []
    for large_image_path in large_image_paths:
        logger.info(f"finding patch in {large_image_path}")

        try:
            ipf = ImagePatchFinderLG(template_path=template_path, large_image_path=large_image_path)
            ipf.find_patch(output_path=output_path)
            ipf.project_image(output_path=output_path)


            # crop = find_patch_tiled(template_path,
            #                         large_image_path,
            #                         output_path=output_path,
            #                         cache_path=cache_path,
            #                         MIN_MATCH_COUNT=MIN_MATCH_COUNT,
            #                         tile_size_x=tile_size_x, tile_size_y=tile_size_y, visualise=False)

            if isinstance(ipf.warped_image_B, np.ndarray):
                crops.append(ipf.warped_image_B)
                im = PIL.Image.fromarray(ipf.warped_image_B)
                im.save(output_path / f"crop_{large_image_path.stem}_t_{template_path.stem}.jpeg")
            else:
                # no match
                pass
        except NoMatchError as e:
            logger.warning(f"patch not found in {large_image_path}. Error: {str(e)}")
            failed_images.append(large_image_path)

    if failed_images:
        logger.info(f"Failed to find patches in the following images: {', '.join(map(str, failed_images))}")
    return crops
