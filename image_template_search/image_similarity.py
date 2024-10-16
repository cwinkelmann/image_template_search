"""

"""
import copy
import gc
import math
import pickle
import random
import typing
from pathlib import Path

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch
from loguru import logger
from shapely.affinity import affine_transform

from image_template_search.image_rasterization import tile_large_image
from image_template_search.opencv_findobject_homography import _cached_detect_and_compute, _cached_matcher, \
    persist_descriptors, persist_keypoints
from image_template_search.util.HastyAnnotationV2 import ImageLabel
from image_template_search.util.util import get_image_id, cache_to_disk
from lightglue import LightGlue, SIFT
from lightglue.utils import load_image, rbd


def _cached_tiled_keypoints_and_descriptors_extraction(detector, large_image,
                                                       tile_size_x, tile_size_y,
                                                       overlap_x, overlap_y,
                                                       tile_base_path, cache_path):
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

    if cache_path is not None:
        img_id = get_image_id(image=large_image)
        keypooints_cache_path = cache_path / f"{img_id}_keypoints.pkl"
        descriptors_cache_path = cache_path / f"{img_id}_descriptors.pkl"

    if cache_path is not None and keypooints_cache_path.exists() and descriptors_cache_path.exists():
        # TODO write a function for this
        with open(keypooints_cache_path, 'rb') as f:
            kp2 = pickle.load(f)
            kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=pt[2], angle=pt[3], response=pt[4], octave=pt[5],
                                class_id=pt[6]) for pt in kp2]
        with open(descriptors_cache_path, 'rb') as f:
            des2 = pickle.load(f)

    else:

        kp2s = []
        des2s = []

        for y in range(0, large_image.shape[0], tile_size_y - overlap_y):
            for x in range(0, large_image.shape[1], tile_size_x - overlap_x):
                try:
                    tile_path = tile_large_image(x, y, tile_size_x, tile_size_y,
                                                 large_image,
                                                 tile_base_path,
                                                 prefix=tile_base_path.name)

                    img2 = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)  # train

                    kp2, des2 = _cached_detect_and_compute(detector, img=img2,
                                                           img_path=tile_path, cache_path=None)

                    # we need to modify the points to match the original image
                    kp_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in
                               kp2]

                    kp2 = [cv2.KeyPoint(x=pt[0] + x, y=pt[1] + y, size=pt[2], angle=pt[3], response=pt[4], octave=pt[5],
                                        class_id=pt[6]) for pt in kp_list]

                    # the keypoints and descriptors are stored in a list
                    if len(kp2) > 0:
                        kp2s.append(kp2)
                    if des2 is not None:
                        des2s.append(des2)

                except RuntimeError as e:
                    logger.error(f"error processing tile: {x}, {y}, {e}")

        # assemble all keypoints and descriptors from the tiles
        kp2 = [item for sublist in kp2s for item in sublist]
        des2 = [item for sublist in des2s for item in sublist]
        del (des2s)
        del (kp2s)
        gc.collect()

        if cache_path is not None:
            persist_descriptors(des=des2, descriptors_cache_path=descriptors_cache_path)
            persist_keypoints(kp=kp2, keypooints_cache_path=keypooints_cache_path)

    return kp2, des2


@cache_to_disk(cache_dir="similarity_cache")
def get_similarity(image0: Path, image1: Path) -> (float, torch.Tensor, torch.Tensor):
    """
    get the similarity between two images
    :param image1:
    :param image2:
    :return:
    """

    image0_T = load_image(image0)
    image1_T = load_image(image1)

    img_norm = image1_T / 255.0 if image1_T.max() > 1 else image1_T
    black_pixels = torch.all(img_norm < 0.1, dim=0).sum()
    white_pixels = torch.all(img_norm > 0.9, dim=0).sum()
    total_pixels = img_norm.shape[1] * img_norm.shape[2]

    if (image1_T.shape[1] > 1000 and image1_T.shape[2] > 1000 and not torch.all(image1_T == 0)
            and (black_pixels + white_pixels) / total_pixels < 0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        torch.set_grad_enabled(False)

        extractor = SIFT(max_num_keypoints=4096).eval().to(device)  # load the extractor
        matcher = LightGlue(features="sift").eval().to(device)

        feats0 = extractor.extract(image0_T.to(device))
        feats1 = extractor.extract(image1_T.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

        normalised_sim = matches.size()[0] / kpts0.size()[0]

        # get matched keypoints only
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        del (image1_T)
        del (image0_T)
        gc.collect()
        return normalised_sim, m_kpts0, m_kpts1

    else:
        logger.warning(f"Image too small or not propper: {image1_T.shape}")
        return 0, torch.tensor([]), torch.tensor([])


def find_rotation_gen(m_kpts0: np.ndarray,
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

    M, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
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
    logger.info(f"normalised_sim: {normalised_sim}")

    fx = 1  # TODO clarify if this is needed
    fy = 1

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    template_identifier = template_path.stem
    large_image_identifier = large_image_path.stem

    M, mask, footprint = find_rotation_gen(m_kpts0.cpu().numpy(),
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


def find_patch_tiled(template_path: Path, large_image_path: Path,
                     output_path=Path("./output"),
                     tile_size_x=1200,
                     tile_size_y=1200,
                     tile_base_path=Path("./"),
                     cache_path=Path("./cache"),
                     MIN_MATCH_COUNT=50, visualise=False) -> np.ndarray:
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

    large_image = cv2.imread(large_image_path,
                             cv2.IMREAD_COLOR)
    large_image = cv2.resize(large_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    # SIFT Based Feature Extractor
    detector = cv2.SIFT_create()
    # detector = cv2.BRISK_create()
    norm = cv2.NORM_HAMMING

    overlap_x = 0
    overlap_y = 0
    pad_size = 0

    # get the keypoints and descriptors of the template image
    template_image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)  # train
    template_image = cv2.resize(template_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    kp1, des1 = _cached_detect_and_compute(detector, img=template_image,
                                           img_path=template_path,
                                           cache_path=cache_path)

    # process the bigger image by tiling it tiled big image
    ## Apply the caching here instead of caching the tiles
    ## TODO Caching here

    kp2, des2 = _cached_tiled_keypoints_and_descriptors_extraction(detector, large_image,
                                                                   tile_size_x, tile_size_y, overlap_x, overlap_y,
                                                                   tile_base_path, None)

    # kp2, des2 = _cached_tiled_keypoints_and_descriptors_extraction(detector, large_image,
    #                                                                  tile_size_x, tile_size_y, overlap_x, overlap_y,
    #                                                    tile_base_path, cache_path)
    del (large_image)
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

        # FIXME the caching of matches does not work
        # matches = _cached_matcher(flann, des1, des2, template_path, Path(large_image_path), k=2,
        #                           cache_path=cache_path)
        matches = _cached_matcher(flann, des1, des2, template_path, Path(large_image_path), k=2,
                                  cache_path=None)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
    except Exception as e:
        logger.error("Not enough matches are found - {}".format(len(matches)))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        ## we take the shape of the first image which was transformed
        ## then we create a box around the transformed image and project it new image
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1,
                                                                                   2)  # This is the box of the query image

        dst = cv2.perspectiveTransform(pts, M)

        new_image_footprint = dst
        # draw the matches outer box
        footprint = np.int32(dst.reshape(4, 2))
        ## TODO create the footprint from that
        footprint = shapely.Polygon(footprint.reshape(4, 2))

        fig = plt.figure(figsize=(20, 20))
        large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
        large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

        large_image = cv2.polylines(large_image, [np.int32(dst)], True, 255, 53, cv2.LINE_AA)
        large_image = cv2.resize(large_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

        if visualise:
            plt.imshow(large_image, 'gray')
            fig.savefig(output_path / "large_image_footprint.jpg")  # TODO give it a good name
            plt.show()

            draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                               singlePointColor=None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            fig = plt.figure(figsize=(20, 20))
            img_matches = cv2.drawMatches(template_image, kp1, large_image, kp2, good, None, **draw_params)
            plt.imshow(img_matches, 'gray')
            fig.savefig(output_path / "large_image_match.jpg")
            plt.show()

        theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
        print(f"The camera rotated: {round(theta, 2)} degrees")

        # TODO class variable
        M_ = np.linalg.inv(M)

        # TODO this code is duplicated
        template_image = cv2.imread(str(template_path))
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
        large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

        rotated_cropped_image_bbox = cv2.warpPerspective(large_image, M_,
                                                         (template_image.shape[1], template_image.shape[0]))
        if visualise:
            fig, axes = plt.subplots(1, sharey=True, figsize=(13, 12))
            # Display the result
            plt.imshow(rotated_cropped_image_bbox)
            # plt.axis('off')  # Hide axis
            plt.show()
        rotated_cropped_image_bbox_path = output_path / f"rotated_cropped_image_bbox_{large_image_path.stem}.jpg"
        cv2.imwrite(str(rotated_cropped_image_bbox_path), cv2.cvtColor(rotated_cropped_image_bbox, cv2.COLOR_RGB2BGR))

        return rotated_cropped_image_bbox

    else:
        logger.error("Not enough good matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        return False


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


import cv2
import numpy as np
from shapely.geometry import Polygon


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


class ImagePatchFinder(object):
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

    def __init__(self, template_path, template_polygon, large_image_path):
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
        self.template_polygon = template_polygon
        self.large_image_path = large_image_path
        self.matched_templates = []

    def __call__(self, *args, **kwargs):
        return self.find_patch()

    def find_patch(self,
                   output_path=Path("./output"),
                   similarity_threshold=0.1):
        """
        Find the template in the large image using LightGlue https://github.com/cvg/LightGlue and SIFT
        TODO: when the template is too small it is not working well. There is no method of identifying if a match is right or not
        :param template_path:
        :param large_image_path:
        :param output_path:
        :return:
        """
        normalised_sim, m_kpts0, m_kpts1 = get_similarity(self.template_path, Path(self.large_image_path))

        if normalised_sim < similarity_threshold:
            logger.warning("The template is not in the image")
            return False

        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        M, mask, footprint = find_rotation_gen(m_kpts0.cpu().numpy(),
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

        warped_other_image_path = output_path / matched_source_image
        cv2.imwrite(str(warped_other_image_path), cv2.cvtColor(self.warped_image_B, cv2.COLOR_RGB2BGR))

        return warped_other_image_path
