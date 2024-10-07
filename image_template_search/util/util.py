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
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import matplotlib.axis as axis
import matplotlib.patches as patches
import matplotlib.axes as axes
from shapely import Polygon

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
                       max_x=None, max_y=None, color="blue", ax:axes.Axes =None) -> axes.Axes:
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
        ax.plot(x, y, color=color, linewidth=0.5)
    for point in points:
        x, y = point.xy
        ax.plot(x, y, marker='o', color=color, linewidth=0.5, markersize=0.5)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    return ax


def visualise_image(image_path: Path = None,
                    image: Image = None,
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
