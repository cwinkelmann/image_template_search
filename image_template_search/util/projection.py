import copy
import typing
from pathlib import Path

import cv2
import numpy as np
import shapely
from PIL import Image
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.affinity import affine_transform

from image_template_search.util.HastyAnnotationV2 import ImageLabel


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
