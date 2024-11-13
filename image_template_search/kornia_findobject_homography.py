"""
using kornia to find an a template in very big image


"""
from pathlib import Path

import torch
import kornia as K
import kornia.feature as KF
import cv2
import numpy as np
from matplotlib import pyplot as plt


# def find_small_image_in_large_image(small_img: np.ndarray, large_img: np.ndarray):
#     # Convert images to grayscale if they are RGB
#     if small_img.shape[-1] == 3:
#         small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
#     else:
#         small_img_gray = small_img
#
#     if large_img.shape[-1] == 3:
#         large_img_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
#     else:
#         large_img_gray = large_img
#
#     # Convert images to tensors
#     small_img_tensor = K.image_to_tensor(small_img_gray, keepdim=False).float() / 255.0
#     large_img_tensor = K.image_to_tensor(large_img_gray, keepdim=False).float() / 255.0
#
#     # # Add batch dimension
#     # small_img_tensor = small_img_tensor.unsqueeze(0)
#     # large_img_tensor = large_img_tensor.unsqueeze(0)
#
#     # SIFT feature extractor
#     sift = KF.SIFTFeature(num_features=500)  # Limit to 500 features for efficiency
#
#     # Extract SIFT keypoints and descriptors
#     small_kpts, small_desc = sift(small_img_tensor)
#
#
#     large_kpts, large_desc = sift(large_img_tensor)
#
#     # Match descriptors using SNN (Second Nearest Neighbor) to find matches
#     matcher = KF.DescriptorMatcher('snn', 0.8)  # Ratio test with a threshold of 0.8
#     matches = matcher(small_desc, large_desc)
#
#     # Filter out unmatched keypoints
#     small_matched_kpts = small_kpts[0, matches[0][0, :]]
#     large_matched_kpts = large_kpts[0, matches[0][1, :]]
#
#     # Convert keypoints to numpy arrays for homography estimation
#     small_pts = small_matched_kpts[:, :2].cpu().numpy()
#     large_pts = large_matched_kpts[:, :2].cpu().numpy()
#
#     # Estimate homography using RANSAC
#     H, inliers = cv2.findHomography(small_pts, large_pts, cv2.RANSAC, 5.0)
#
#     return H, inliers

if __name__ == "__main__":
    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
    image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif"
    # This is absolutely unsuitable for large images
    # M, inliers = find_small_image_in_large_image(cv2.imread(str(drone_image)), cv2.imread(str(image_2)))
    M_ = np.linalg.inv(M)

    # estimate the transformation

    template_image = cv2.imread(str(drone_image))
    rotated_cropped_image_bbox = cv2.warpPerspective(cv2.imread(str(image_2)), M_,
                                                     (template_image.shape[1], template_image.shape[0]))
    fig, axes = plt.subplots(1, sharey=True, figsize=(13, 12))
    # Display the result
    plt.imshow(rotated_cropped_image_bbox)
    # plt.axis('off')  # Hide axis
    plt.show()