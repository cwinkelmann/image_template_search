import typing
from datetime import datetime

import shapely
import numpy as np
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
import matplotlib.axis as axis
import matplotlib.axes as axes
from shapely.geometry import box, Point
from PIL import Image as PILImage
from shapely.geometry import Polygon
from typing import List

import os
import joblib
import torch
from functools import wraps
import hashlib
from pathlib import Path
from joblib import Memory

from conf.config_dataclass import CacheConfig, get_config
from image_template_search.util.HastyAnnotationV2 import ImageLabel
from image_template_search.util.georeferenced_image import ExifMetaData, XMPMetaData, ExtendImageMetaData


def feature_extractor_cache():
    """
    Decorator to cache the results of the function to disk.
    :param cache_dir: Directory where cached results will be stored
    """

    def decorator(func):
        cache_dir = CacheConfig.cache_path

        @wraps(func)
        def wrapper(image_path: Path, *args, **kwargs):
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache file name based on the inputs (hash the paths)
            cache_key = hashlib.md5(f"{image_path}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}_feats.pkl")

            # Check if cache exists
            if os.path.exists(cache_file):
                # Load cached result
                logger.info(f"Loading cached {func.__name__} result for {image_path.stem}")
                return joblib.load(cache_file)

            # Call the function and cache the result
            result = func(image_path, *args, **kwargs)
            joblib.dump(result, cache_file)
            logger.info(f"Cached result to {cache_file} for {image_path.stem}")

            return result
        return wrapper
    return decorator


def generic_cache_to_disk():
    """
    Generic caching decorator using Joblib's Memory caching.

    :param cache_dir: Directory where cached results will be stored.
    """

    def decorator(func):
        cache_dir = f"{CacheConfig.cache_path}_{func.__name__}"
        # Set up Joblib memory object
        memory = Memory(cache_dir, verbose=0)
        cached_func = memory.cache(func)  # Cache the function using Joblib
        logger.info(f"Loading cached {func.__name__} from {cache_dir}")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the cached version of the function
            return cached_func(*args, **kwargs)

        return wrapper

    return decorator


def cache_to_disk():
    """
    Decorator to cache the results of the function to disk.
    :param cache_dir: Directory where cached results will be stored
    """

    def decorator(func):
        cache_dir = CacheConfig.cache_path

        @wraps(func)
        def wrapper(image0: Path, image1: Path, *args, **kwargs):
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache file name based on the inputs (hash the paths)
            cache_key = hashlib.md5(f"{image0}{image1}".encode()).hexdigest()
            cache_key_ = hashlib.md5(f"{image1}{image0}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            cache_file_ = os.path.join(cache_dir, f"{cache_key_}.pkl")

            # Check if cache exists
            if os.path.exists(cache_file):
                # Load cached result
                logger.info(f"Loading cached result for {image0.stem} and {image1.stem}")
                return joblib.load(cache_file)

            # Call the function and cache the result
            result = func(image0, image1, *args, **kwargs)
            joblib.dump(result, cache_file)
            logger.info(f"Cached result to {cache_file} for {image0.stem} and {image1.stem}")

            return result
        return wrapper
    return decorator


def get_similarity_cache_to_disk():
    """
    Decorator to cache the results of the function to disk.
    :param cache_dir: Directory where cached results will be stored
    """

    def decorator(func):
        cache_dir = get_config().cache_path

        @wraps(func)
        def wrapper(image0: Path, image1: Path, *args, **kwargs):
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache file name based on the inputs (hash the paths)
            cache_key = hashlib.md5(f"{image0}{image1}".encode()).hexdigest()
            cache_key_ = hashlib.md5(f"{image1}{image0}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"cache_find_similarity_{image0.name}_{image1.name}_{cache_key}.pkl")
            cache_file_ = os.path.join(cache_dir, f"cache_find_similarity_{image1.name}_{image0.name}_{cache_key_}.pkl")

            # Check if cache exists
            if os.path.exists(cache_file) and CacheConfig.caching:
                # Load cached result
                logger.info(f"Loading cached result for {image0.stem} and {image1.stem} from {cache_file}")
                return joblib.load(cache_file)

            # Call the function and cache the result
            result = func(image0, image1, *args, **kwargs)
            normalised_sim, m_kpts0, m_kpts1 = result
            joblib.dump((normalised_sim, m_kpts0, m_kpts1), cache_file)
            joblib.dump((normalised_sim, m_kpts1, m_kpts0), cache_file_)
            logger.info(f"Cached result to {cache_file} for {image0.stem} and {image1.stem}")
            logger.info(f"Cached result to {cache_file_} for {image1.stem} and {image0.stem}")

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
    assert (filename is not None) != (image is not None), "Either filename or image must be provided"

    if filename is not None:
        with open(filename, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()

        return readable_hash
    if image is not None:
        image_bytes = image.tobytes()
        readable_hash = hashlib.sha256(image_bytes).hexdigest()

        return readable_hash


def get_template_id(image_name, combined_hash, patch_size):
    return f"{image_name}__{combined_hash}__{patch_size}"


def hash_objects(objs: list[ImageLabel]) -> str:
    # make the template unique
    hashes = {o.id for o in objs}
    hashes = "_".join(hashes)

    hash_object = hashlib.sha256(hashes.encode())
    # Return a truncated version of the hash (e.g., first 10 characters)
    combined_hash = hash_object.hexdigest()[0:10]

    return combined_hash


def visualise_polygons(polygons: List[shapely.Polygon] = (),
                       points: List[shapely.Point] = (),
                       filename=None, show=False, title=None,
                       max_x=None, max_y=None, color="blue",
                       ax: axes.Axes = None, linewidth=0.5, markersize=0.5, fontsize=22,
                       labels: List[str] = None) -> axes.Axes:
    """
    Visualize a list of polygons
    :param labels:
    :param fontsize:
    :param markersize:
    :param linewidth:
    :param ax:
    :param color:
    :param max_y:
    :param max_x:
    :param title:
    :param show:
    :param filename:
    :param points:
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
    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)

        # Add label for each polygon if labels are provided
        if labels and i < len(labels):
            # Get the centroid of the polygon for labeling
            centroid = polygon.centroid
            ax.text(centroid.x, centroid.y, labels[i], fontsize=fontsize, ha='center', color='red')

    for point in points:
        x, y = point.xy
        ax.plot(x, y, marker='o', color=color, linewidth=linewidth, markersize=markersize)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
        plt.close()

    return ax


def visualise_image(image_path: Path = None,
                    image: typing.Union[PILImage, np.ndarray] = None,
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
        image = PILImage.fromarray(image)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)  # TODO use the shape of imr to get the right ration
    if image_path is not None:
        PILImage.Image.MAX_IMAGE_PIXELS = 5223651122

        image = PILImage.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

    imr = np.array(image, dtype=np.uint8)
    ax.imshow(imr)
    ax.set_title(title)

    if output_file_name is not None:
        plt.savefig(output_file_name)

    if show:
        plt.show()
        return ax
    else:
        return ax




def create_box_around_point(center: Point, a: float, b: float, image_width = None, image_height = None) -> box:
    """
    create a bounding box around a point
    """
    x_center, y_center = center.x, center.y

    # Calculate the min and max x, y values for the box
    minx = x_center - a / 2
    maxx = x_center + a / 2
    miny = y_center - b / 2
    maxy = y_center + b / 2

    if image_width is not None and image_height is not None:
        # Adjust the box position to stay within image boundaries, without changing its dimensions
        if minx < 0:
            minx = 0
            maxx = a
        elif maxx > image_width:
            maxx = image_width
            minx = image_width - a

        if miny < 0:
            miny = 0
            maxy = b
        elif maxy > image_height:
            maxy = image_height
            miny = image_height - b

    # Create a box with these bounds
    return box(minx, miny, maxx, maxy)

def calculate_nearest_border_distance(centroids: list[shapely.Point], frame_width, frame_height):
    """
    Calculate the distance from each centroid to the nearest border of the frame.

    Parameters:
    - centroids: List of tuples [(x1, y1), (x2, y2), ...] representing centroid coordinates.
    - frame_width: Width of the frame (maximum x value).
    - frame_height: Height of the frame (maximum y value).

    Returns:
    - List of distances corresponding to each centroid.
    """
    distances = []
    for idx, p in enumerate(centroids):
        # Distances to each border
        distance_left = p.x  # Distance to x = 0
        distance_right = frame_width - p.x  # Distance to x = frame_width
        distance_top = p.y  # Distance to y = 0
        distance_bottom = frame_height - p.y  # Distance to y = frame_height

        # Nearest distance
        nearest_distance = min(distance_left, distance_right, distance_top, distance_bottom)

        distances.append(nearest_distance)
        # logger.info(f"  Nearest Distance to Border: {nearest_distance}")
    return distances

def crop_image_bounds(image: typing.Union[PILImage, np.ndarray, Path], polygon: Polygon)-> PILImage:
    # Get the bounding box coordinates (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = polygon.bounds

    if image is not None and isinstance(image, Path):
        image = PILImage.open(image)

    # Ensure the bounding box fits within the image dimensions
    img_width, img_height = image.size
    minx, miny = max(0, minx), max(0, miny)
    maxx, maxy = min(img_width, maxx), min(img_height, maxy)

    # Crop the image using the bounding box
    cropped_image = image.crop((minx, miny, maxx, maxy))

    return cropped_image

def crop_templates_from_image(image: typing.Union[PILImage, np.ndarray], bbox_polygons: List[Polygon]) -> List[PILImage]:
    """
    Crop the rectangular polygons from the image and return them as a list of cropped images.

    :param image: PIL image from which objects are to be cropped
    :param bbox_polygons: List of rectangular Shapely polygons representing bounding boxes
    :return: List of cropped PIL images
    """
    cropped_images = []
    if isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)

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


def get_image_dimensions(image_path) -> typing.Tuple[int, int]:
    """
    Get the dimensions of an image file.
    :param image_path:
    :return:
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def decimal_coords(coords, ref):
    """
    calculate the decimal degrees
    :param coords:
    :param ref:
    :return:
    """
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def get_exif_metadata(img_path: Path) -> ExtendImageMetaData:
    """
    load gps coordinates from the image

    https://medium.com/spatial-data-science/how-to-extract-gps-coordinates-from-images-in-python-e66e542af354

    :param img_path:
    :return:
    """
    from exif import Image as ExifImage

    with open(img_path, 'rb') as src:
        img = ExifImage(src)
        image_id = get_image_id(img_path)

    if img.has_exif:
        try:
            latitude = decimal_coords(img.gps_latitude, img.gps_latitude_ref)
            longitude = decimal_coords(img.gps_longitude, img.gps_longitude_ref)

            # print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
            # print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")
            metadata = {"image_id": image_id,
                        "image_name": img_path.name,
                        "latitude": latitude, "longitude": longitude,
                        "datetime_original": img.datetime_original,

                        "filepath": str(img_path)}

            supposedly_available_keys = ['image_width', 'image_height', 'bits_per_sample', 'image_description',
                                         'make', 'model', 'orientation',
                                         'samples_per_pixel', 'x_resolution', 'y_resolution', 'resolution_unit',
                                         'software', 'datetime',
                                         'y_and_c_positioning', '_exif_ifd_pointer', '_gps_ifd_pointer',
                                         'xp_keywords', 'compression',
                                         'jpeg_interchange_format', 'jpeg_interchange_format_length',
                                         'exposure_time', 'f_number',
                                         'exposure_program', 'photographic_sensitivity', 'exif_version',
                                         'datetime_original',
                                         'datetime_digitized', 'exposure_bias_value', 'max_aperture_value',
                                         'metering_mode', 'light_source',
                                         'focal_length', 'color_space', 'pixel_x_dimension',
                                         'pixel_y_dimension', 'exposure_mode',
                                         'white_balance', 'digital_zoom_ratio', 'focal_length_in_35mm_film',
                                         'scene_capture_type',
                                         'gain_control', 'contrast', 'saturation', 'sharpness',
                                         'body_serial_number', 'lens_specification',
                                         'gps_version_id', 'gps_latitude_ref', 'gps_latitude',
                                         'gps_longitude_ref', 'gps_longitude',
                                         'gps_altitude_ref', 'gps_altitude']

            # raw_exif = img.get_all()
            for key in supposedly_available_keys:
                metadata[key] = img.get(key)

            metadata["datetime_digitized"] = str(
                datetime.strptime(metadata["datetime_digitized"], "%Y:%m:%d %H:%M:%S"))

            exif_meta_data = ExtendImageMetaData(**metadata)

        except AttributeError:
            print('No Coordinates')
            image_metadata = {}
            return {}

    else:
        logger.warning(f"The Image {src} has no EXIF information")

    # exif_meta_data = ExifMetaData(**metadata)
    return exif_meta_data

def list_images(path: Path, extension, recursive=False):
    """
    find images in a path

    :param extension:
    :param recursive:
    :return:
    :param path:
    :return:
    """

    assert extension in ["jpg", "jpeg", "png", "tif", "tiff", "JPG", "JPEG"], f"Unsupported image extension: {extension}"

    if recursive:
        images_list = list(path.rglob(f"*.{extension}"))
    else:
        images_list = list(path.glob(f"*.{extension}"))

    # remove hidden files which are especially annoying on a Mac
    images_list = [image_path for image_path in images_list if not str(image_path.name).startswith(".")]

    return images_list
