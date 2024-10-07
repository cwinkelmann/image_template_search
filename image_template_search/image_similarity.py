"""

"""

import gc
import math
import pickle

import shapely
import torch
from loguru import logger
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from image_template_search.image_rasterization import tile_large_image
from image_template_search.opencv_findobject_homography import _cached_detect_and_compute, _cached_matcher, \
    persist_descriptors, persist_keypoints
from image_template_search.util.util import get_image_id
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
        and (black_pixels + white_pixels) / total_pixels < 0.5) :
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

        del(image1_T)
        del(image0_T)
        gc.collect()
        return normalised_sim, m_kpts0, m_kpts1
    else:
        logger.warning(f"Image too small or not propper: {image1_T.shape}")
        return 0, torch.tensor([]), torch.tensor([])


def find_rotation_gen(m_kpts0: np.ndarray,
                      m_kpts1: np.ndarray,
                      image_name:str) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    find the footprint of the template in the image
    :param m_kpts0:
    :param m_kpts1:
    :param image_name:
    :return:
    """
    M, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
    # M2, mask2 = cv2.getAffineTransform(m_kpts0, m_kpts1)
    img1 = cv2.imread(str(image_name), cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1,
                                                                               2)  # This is the box of the query image

    # M = np.linalg.inv(M)
    footprint = cv2.perspectiveTransform(pts, M)
    footprint = np.int32(footprint.reshape(4, 2))

    return M, mask, footprint


def find_patch(template_path: Path, large_image_path: Path, output_path = Path("./output")):
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

    fx = 1 # TODO clarify if this is needed
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

    fig = plt.figure(figsize=(20, 20))
    large_image = cv2.imread(large_image_path, cv2.IMREAD_COLOR)
    large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

    large_image_l = cv2.polylines(large_image, [np.int32(dst)], True, 255, 53, cv2.LINE_AA)
    large_image_l = cv2.resize(large_image_l, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)


    plt.imshow(large_image_l, 'gray')
    fig.savefig(output_path / f"t_{template_identifier}_b{large_image_identifier}_large_image_footprint.jpg")
    plt.show()

    # TODO class variable
    theta = - math.atan2(M[0,1], M[0,0]) * 180 / math.pi
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

    return rotated_cropped_image_bbox


def find_patch_tiled(template_path: Path, large_image_path: Path,
                     output_path=Path("./output"),
                     tile_size_x=1200,
                     tile_size_y=1200,
                     tile_base_path=Path("./"),
                     cache_path=Path("./cache"),
                     MIN_MATCH_COUNT = 50, visualise=False):
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
            fig.savefig(output_path / "large_image_footprint.jpg") # TODO give it a good name
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
                       tile_path, cache_path, MIN_MATCH_COUNT = 50,
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
        crop = find_patch_tiled(template_path,
                                large_image_path,
                                output_path=output_path,
                                cache_path=cache_path,
                                MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                                tile_size_x=tile_size_x, tile_size_y=tile_size_y, visualise=False)

        if isinstance(crop, np.ndarray):
            crops.append(crop)
        else:
            # no match
            pass

    return crops





class ImagePatchFinder(object):

    def __init__(self, template_path, large_image_path):
        self.template_path = template_path
        self.large_image_path = large_image_path

        # TODO integrate the simple functions so the code is rather wrapped and seperated
