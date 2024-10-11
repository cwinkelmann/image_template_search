import typing
from pathlib import Path
import hashlib

import numpy as np

from time import sleep

import shapely
from typing import List

import PIL
import copy

import uuid
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
import matplotlib.axis as axis
import matplotlib.patches as patches
import matplotlib.axes as axes
from shapely import Polygon
from shapely.geometry import box, Point
from PIL import Image
from shapely.geometry import Polygon
from typing import List, Tuple

import os
import joblib
import torch
from functools import wraps
from pathlib import Path
import hashlib

def cache_to_disk(cache_dir="cache"):
    """
    Decorator to cache the results of the function to disk.
    :param cache_dir: Directory where cached results will be stored
    """
    def decorator(func):
        @wraps(func)
        def wrapper(image0: Path, image1: Path, *args, **kwargs):
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache file name based on the inputs (hash the paths)
            cache_key = hashlib.md5(f"{image0}{image1}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

            # Check if cache exists
            if os.path.exists(cache_file):
                # Load cached result
                logger.info(f"Loading cached result for {image0} and {image1}")
                return joblib.load(cache_file)

            # Call the function and cache the result
            result = func(image0, image1, *args, **kwargs)
            joblib.dump(result, cache_file)
            logger.info(f"Cached result to {cache_file}")

            return result
        return wrapper
    return decorator


def get_image_id(filename: Path = None, image: np.ndarray = None):
    """
    @ has moved to flight_image_capturing_sim/helper/image.py
    generate an id from the image itself which can be used to find images which are exactly the same
    @param filename:
    @return:
    """

    if filename is not None:
        with open(filename, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash
    if image is not None:
        image_bytes = image.tobytes()
        readable_hash = hashlib.sha256(image_bytes).hexdigest()
        return readable_hash


def visualise_polygons(polygons: List[shapely.Polygon] = (), points: List[shapely.Point] = (),
                       filename=None, show=False, title = None,
                       max_x=None, max_y=None, color="blue", ax:axes.Axes =None, linewidth=0.5, markersize=0.5) -> axes.Axes:
    """
    Visualize a list of polygons
    :param polygons:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    assert isinstance(ax, axes.Axes), f"Expected matplotlib.axes.Axes, got {type(ax)}"

    if max_x:
        plt.xlim(0, max_x)
    if max_y:
        plt.ylim(0, max_y)
    if title:
        plt.title(title)
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)
    for point in points:
        x, y = point.xy
        ax.plot(x, y, marker='o', color=color, linewidth=linewidth, markersize=markersize)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    return ax


def visualise_image(image_path: Path = None,
                    image: typing.Union[PIL.Image, np.ndarray] = None,
                    output_file_name: Path = None,
                    show: bool = False,
                    title: str = "original image",
                    ax: axes.Axes = None, figsize=(20, 15), dpi=150) -> axis:
    """
    :param image:
    :param output_file_name:
    :param show:
    :param figsize:
    :param ax:
    :param title:
    :param image_path:
    :return:
    """

    if image is not None and isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)  # TODO use the shape of imr to get the right ration
    if image_path is not None:
        image = Image.open(image_path)
    imr = np.array(image, dtype=np.uint8)
    ax.imshow(imr)
    ax.set_title(title)

    if output_file_name is not None:
        plt.savefig(output_file_name)

    if show:
        plt.show()
        # sleep(0.1)
    else:
        return ax




def create_box_around_point(center: Point, a: float, b: float) -> box:
    """
    create a bounding box around a point
    """
    x_center, y_center = center.x, center.y

    # Calculate the min and max x, y values for the box
    minx = x_center - a / 2
    maxx = x_center + a / 2
    miny = y_center - b / 2
    maxy = y_center + b / 2

    # Create a box with these bounds
    return box(minx, miny, maxx, maxy)




def crop_objects_from_image(image: typing.Union[Image, np.ndarray], bbox_polygons: List[Polygon]) -> List[Image]:
    """
    Crop the rectangular polygons from the image and return them as a list of cropped images.

    :param image: PIL image from which objects are to be cropped
    :param bbox_polygons: List of rectangular Shapely polygons representing bounding boxes
    :return: List of cropped PIL images
    """
    cropped_images = []
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    for polygon in bbox_polygons:
        # Ensure the polygon is a rectangle
        if not polygon.is_valid or len(polygon.exterior.coords) != 5:
            raise ValueError("One of the polygons is not a valid rectangle.")

        # Get the bounding box coordinates (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = polygon.bounds

        # Ensure the bounding box fits within the image dimensions
        img_width, img_height = image.size
        minx, miny = max(0, minx), max(0, miny)
        maxx, maxy = min(img_width, maxx), min(img_height, maxy)

        # Crop the image using the bounding box
        cropped_image = image.crop((minx, miny, maxx, maxy))

        # Append the cropped image to the list
        cropped_images.append(cropped_image)

    return cropped_images
