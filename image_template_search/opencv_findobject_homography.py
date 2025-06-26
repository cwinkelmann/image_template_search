"""
https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

find a template image in a larger image.
Doesn't work for big images.
rotated or perspective transformed images should work, i.e. cliff flights.
"""
import gc
import pickle

import cv2
from PIL import Image
from loguru import logger
from pathlib import Path

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt, patches
from shapely.geometry.polygon import Polygon

from image_template_search.util.util import get_image_id

MIN_MATCH_COUNT = 10

def tile_large_image(x, y, tile_size_x, tile_size_y,
                     overlap_x, overlap_y,
                     large_image, template_path, tile_base_path, prefix):
    """
    Wrapper function to extract the tile and call match_template.
    Adjusts match coordinates for their position in the original image.
    """

    # tile creation
    tile = large_image[y:y+tile_size_y, x:x+tile_size_x]
    tile_path = tile_base_path / f"{prefix}_tile_{x}_{y}.jpg"
    a = cv2.imwrite(str(tile_path), tile)

    return tile_path



def transform_points(M, point):
    """
    Apply the transformation matrix to the points
    :param M:
    :param points:
    :return:
    """
    transformed_point1_homogeneous = np.matmul(M, point)
    return transformed_point1_homogeneous


def _cached_detect_and_compute(detector, img, img_path, cache_path: Path = None):
    """
    Detect and compute the keypoints and descriptors
    :param detector:
    :param img:
    :return:
    """


    if cache_path is not None:
        logger.info(f"Cache Path: {cache_path}")
        img_id = get_image_id(img_path)
        keypooints_cache_path = cache_path / f"{img_path.stem}_{img_id}_keypoints.pkl"
        descriptors_cache_path = cache_path / f"{img_path.stem}_{img_id}_descriptors.pkl"

    if cache_path is not None and keypooints_cache_path.exists() and descriptors_cache_path.exists():
        with open(keypooints_cache_path, 'rb') as f:
            kp = pickle.load(f)
            kp = [cv2.KeyPoint(x=pt[0], y=pt[1], size=pt[2], angle=pt[3], response=pt[4], octave=pt[5],
                                      class_id=pt[6]) for pt in kp]

        with open(descriptors_cache_path, 'rb') as f:
            des = pickle.load(f)
    else:
        kp, des = detector.detectAndCompute(img, None)

        if cache_path is not None:

            persist_descriptors(des, descriptors_cache_path)
            persist_keypoints(kp, keypooints_cache_path)

    return kp, des


def persist_descriptors(des, descriptors_cache_path):
    """
    Persist the descriptors of opencv
    :param des:
    :param descriptors_cache_path:
    """
    with open(descriptors_cache_path, 'wb') as f:
        pickle.dump(des, f)


def persist_keypoints(kp, keypooints_cache_path):
    """
    Persist the keypoints of opencv
    :param kp:
    :param keypooints_cache_path:
    :return:
    """
    with open(keypooints_cache_path, 'wb') as f:
        keypoints_picklable = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in
                               kp]
        pickle.dump(keypoints_picklable, f)


def _matcher(flann, des1, des2, img_1_path: Path, img_2_path: Path,
             k=2):
    """
    Cache the matches
    :param flann:
    :param des1:
    :param des2:
    :param img_1_path:
    :param img_2_path:
    :param k:
    :param cache_path:
    :return:
    """
    logger.info(f"Computing matches for {img_1_path.name} and {img_2_path.name}")
    matches = flann.knnMatch(des1, des2, k=k)
    logger.info(f"Done Computing matches for {img_1_path.name} and {img_2_path.name}")

    return matches


