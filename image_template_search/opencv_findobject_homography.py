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

# def find_rotation(img1_path: Path, img2_path: Path, cache_path):
#     """
#     estimate the affine transformation matrix == homography between two images
#     :param img1_path:
#     :param img2_path:
#     :return:
#     """
#     img1 = cv.imread(
#         str(img1_path),
#         cv.IMREAD_GRAYSCALE)  # queryImage
#
#     img2 = cv.imread(
#         str(img2_path),
#         cv.IMREAD_GRAYSCALE)  # train
#
#     detector = cv.SIFT_create()
#     # detector = cv.BRISK_create(thresh=70, octaves=0, patternScale=1.0)
#     norm = cv.NORM_HAMMING
#
#     logger.info(f"detector.detectAndCompute")
#     # find the keypoints and descriptors with SIFT
#     # kp1, des1 = detector.detectAndCompute(img1, None)
#     kp1, des1 = _cached_detect_and_compute(detector, img1, img1_path, cache_path=cache_path)
#     # kp2, des2 = detector.detectAndCompute(img2, None)
#     kp2, des2 = _cached_detect_and_compute(detector, img2, img2_path, cache_path=cache_path)
#
#     logger.info(f"Found {len(kp1)} keypoints in img1 {img1_path.name}")
#     logger.info(f"Found {len(kp2)} keypoints in img2 {img2_path.name}")
#
#     if len(kp1) > 10 and len(kp2) > 10:
#         ## BRISK
#         # FLANN_INDEX_KDTREE = 1
#         # index_params = dict(algorithm = 6,
#         #                                table_number = 6, # 12
#         #                                key_size = 12,     # 20
#         #                                multi_probe_level = 1)
#         # # search_params = dict(checks=50)
#         # flann = cv.FlannBasedMatcher(index_params, {})
#
#         ## SIFT
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)
#         flann = cv.FlannBasedMatcher(index_params, search_params)
#         # matches = flann.knnMatch(des1, des2, k=2)
#
#         logger.info(f"flann.knnMatch")
#         matches = _matcher(flann, des1, des2, img1_path, img2_path, k=2, cache_path=cache_path)
#         # matches = flann.knnMatch(des1, des2, k=2)
#     else:
#         logger.error("Not enough keypoints found to be matched")
#         matches = []
#
#
#
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     try:
#         for m, n in matches:
#             if m.distance < 0.7 * n.distance:
#                 good.append(m)
#     except Exception as e:
#         logger.error("Not enough matches are found - {}".format(len(matches)))
#         matchesMask = None
#         M = None
#         new_image_footprint = None
#
#
#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         logger.info(f"cv.findHomography")
#         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()
#
#         ## we take the shape of the first image which was transformed
#         ## then we create a box around the transformed image and project it new image
#         h, w = img1.shape
#         pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) # This is the box of the query image
#         logger.info(f"cv.perspectiveTransform")
#         dst = cv.perspectiveTransform(pts, M)
#         new_image_footprint = dst
#         # draw the matches outer box
#         img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
#
#
#         new_image_footprint = np.int32(dst.reshape(4, 2))
#
#     else:
#         print("Not enough good matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#         matchesMask = None
#         M = None
#         new_image_footprint = None
#
#     draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)
#
#     # img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#
#     plt.imshow(img2, 'gray')
#     plt.show()
#     plt.close()
#
#     return M, matchesMask, new_image_footprint



# def find_rotation_list(query_img_path: Path, img2_paths: [Path], cache_path):
#     """
#     TODO implement this efficiently
#     estimate the affine transformation matrix == homography between a query image and a list of image tiles
#     :param query_img_path:
#     :param img2_path:
#     :return:
#     """
#     # TODO
#     img1 = cv.imread(
#         str(query_img_path),
#         cv.IMREAD_GRAYSCALE)  # queryImage
#
#
#
#     detector = cv.SIFT_create()
#     # detector = cv.BRISK_create(thresh=70, octaves=0, patternScale=1.0)
#     norm = cv.NORM_HAMMING
#
#     logger.info(f"detector.detectAndCompute")
#     # find the keypoints and descriptors with SIFT
#     # kp1, des1 = detector.detectAndCompute(img1, None)
#     kp1, des1 = _cached_detect_and_compute(detector, img1, query_img_path, cache_path=cache_path)
#     # kp2, des2 = detector.detectAndCompute(img2, None)
#     logger.info(f"Found {len(kp1)} keypoints in img1 {query_img_path.name}")
#
#     kp2s = []
#     des2s = []
#     for img2_path in img2_paths:
#         img2 = cv.imread(
#             str(img2_path),
#             cv.IMREAD_GRAYSCALE)  # train
#
#         kp2, des2 = _cached_detect_and_compute(detector, img2, img2_path, cache_path=cache_path)
#         kp2s.append(kp2)
#         des2s.append(des2)
#         logger.info(f"Found {len(kp2)} keypoints in img2 {img2_path.name}")
#
#     ## TODO: now the keypoints have the corrdinates of the part of the image, we need to transform them to the coordinates of the original image
#
#     if len(kp1) > 10 and len(kp2) > 10:
#         ## BRISK
#         # FLANN_INDEX_KDTREE = 1
#         # index_params = dict(algorithm = 6,
#         #                                table_number = 6, # 12
#         #                                key_size = 12,     # 20
#         #                                multi_probe_level = 1)
#         # # search_params = dict(checks=50)
#         # flann = cv.FlannBasedMatcher(index_params, {})
#
#         ## SIFT
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)
#         flann = cv.FlannBasedMatcher(index_params, search_params)
#         # matches = flann.knnMatch(des1, des2, k=2)
#
#         logger.info(f"flann.knnMatch")
#         matches = _matcher(flann, des1, des2, query_img_path, img2_path, k=2, cache_path=cache_path)
#         # matches = flann.knnMatch(des1, des2, k=2)
#     else:
#         logger.error("Not enough keypoints found to be matched")
#         matches = []
#
#
#
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     try:
#         for m, n in matches:
#             if m.distance < 0.7 * n.distance:
#                 good.append(m)
#     except Exception as e:
#         logger.error("Not enough matches are found - {}".format(len(matches)))
#         matchesMask = None
#         M = None
#         new_image_footprint = None
#
#
#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         logger.info(f"cv.findHomography")
#         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()
#
#         ## we take the shape of the first image which was transformed
#         ## then we create a box around the transformed image and project it new image
#         h, w = img1.shape
#         pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) # This is the box of the query image
#         logger.info(f"cv.perspectiveTransform")
#         dst = cv.perspectiveTransform(pts, M)
#         new_image_footprint = dst
#         # draw the matches outer box
#         img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
#
#
#         new_image_footprint = np.int32(dst.reshape(4, 2))
#
#     else:
#         print("Not enough good matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#         matchesMask = None
#         M = None
#         new_image_footprint = None
#
#     draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)
#
#     # img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#
#     #plt.imshow(img2, 'gray')
#     #plt.show()
#     plt.close()
#
#     return M, matchesMask, new_image_footprint

#img1 = Path('/Users/christian/data/2TB/ai-core/data/demo_FMO04_BDII/demo_FMO04_subset/output/ID_9__DJI_0052.JPG')
#img2 = Path('/Users/christian/data/2TB/ai-core/data/demo_FMO04_BDII/demo_FMO04_subset/output/ID_9__DJI_0053.JPG')

#M, mask = find_rotation(img1, img2)
#
# if __name__ == "__main__":
#     base_path = "/Users/christian/data/2TB/ai-core/data/demo_FMO04_BDII/demo_FMO04_subset/output/clipped_fp_25_4"
#     fig, ax = plt.subplots(1, figsize=(5, 5))
#
#     base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
#     drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
#     image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif"
#     interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")
#
#
#     #M, mask, new_image_footprint = find_rotation(img1, img2)
#
#     tile_base_path = interm_path / "tiles"
#     cache_path = interm_path / "cache"
#     output_path = interm_path / "output"
#
#     p_image = Image.open(
#         drone_image)  # Replace with your image file path
#
#     source_image_width, source_image_height = p_image.size
#     template_extent = Polygon(
#         [(0, 0), (source_image_width, 0), (source_image_width, source_image_height), (0, source_image_height)])
#
#     # TODO this is duplicate code
#     kp1, des1, kp2, des2 = tile_opencv_findobject_homography(template_path=drone_image,
#                                                              large_image_path=image_2,
#                                                                      tile_base_path=tile_base_path,
#                                                              cache_path=cache_path, output_path=output_path)
#
#     M, matchesMask, new_image_footprint = opencv_flann_matching(des1, des2, kp1, kp2, query_img_path=drone_image, img2_path=image_2, cache_path=cache_path)
#
#     M_ = np.linalg.inv(M)

