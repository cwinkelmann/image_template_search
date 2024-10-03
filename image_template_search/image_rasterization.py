from typing import List

import PIL.Image
import cv2
import shapely
from shapely import Polygon


def generate_positions(center, max_value, slice_size):
    """
    Function to generate positions along one axis

    :param center:
    :param max_value:
    :param slice_size:
    :return:
    """
    positions = []
    positions_set = set()
    # Positive direction from center
    offset = 0
    while center + offset < max_value:
        pos = center + offset
        if pos not in positions_set and 0 <= pos <= max_value - slice_size:
            positions.append(pos)
            positions_set.add(pos)
        offset += slice_size
    # Negative direction from center
    offset = -slice_size
    while center + offset >= 0:
        pos = center + offset
        if pos not in positions_set and 0 <= pos <= max_value - slice_size:
            positions.append(pos)
            positions_set.add(pos)
        offset -= slice_size
    return sorted(positions)


def create_regular_raster_grid(max_x: int, max_y: int,
                                           slice_height: int,
                                           slice_width: int) -> [List[Polygon]]:
    """
    Create a regular raster grid of oriented bounding boxes, starting from the center of the image,
    with the center point located at the intersection of four tiles.
    """
    tiles = []
    tile_coordinates = []

    center_x = max_x / 2
    center_y = max_y / 2

    # Generate x and y positions
    x_positions = generate_positions(center_x, max_x, slice_width)
    y_positions = generate_positions(center_y, max_y, slice_height)

    # Generate tiles and calculate distances from the center
    tiles_info = []

    for height_i, y1 in enumerate(y_positions):
        y2 = y1 + slice_height
        for width_j, x1 in enumerate(x_positions):
            x2 = x1 + slice_width

            # Ensure tiles are within image boundaries
            x1_clipped = max(x1, 0)
            y1_clipped = max(y1, 0)
            x2_clipped = min(x2, max_x)
            y2_clipped = min(y2, max_y)

            # Adjust tile size if it goes beyond boundaries
            if x2_clipped - x1_clipped <= 0 or y2_clipped - y1_clipped <= 0:
                continue

            # Calculate the center of the tile
            tile_center_x = (x1_clipped + x2_clipped) / 2
            tile_center_y = (y1_clipped + y2_clipped) / 2

            # Calculate the Euclidean distance from the tile center to the image center
            distance = ((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2) ** 0.5

            # Append tile information
            tiles_info.append((distance, height_i, width_j, x1_clipped, y1_clipped, x2_clipped, y2_clipped))

    # Sort the tiles based on their distance from the image center (closest first)
    tiles_info_sorted = sorted(tiles_info, key=lambda x: x[0])

    # Generate tiles starting from the center
    for info in tiles_info_sorted:
        distance, height_i, width_j, x1, y1, x2, y2 = info

        # Create the polygon for the tile
        pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        # Append the polygon and its coordinates to the lists
        tiles.append(pol)

        tile_coordinates.append({
            "height_i": height_i,
            "width_j": width_j,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return tiles, tile_coordinates


def crop_polygons(image: PIL.Image,
                  rasters: List[shapely.Polygon],
                  ) -> List[PIL.Image]:
    """
    iterate through rasters and crop out the tiles from the image return the new images

    :param rasters:
    :param output_path:
    :param full_images_path:

    """
    images = []

    # sliding window of the image
    for pol in rasters:
        assert isinstance(pol, shapely.Polygon)
        sliced = image.crop(pol.bounds)
        images.append(sliced)

    return images


def tile_large_image(x, y, tile_size_x, tile_size_y,
                     large_image, tile_base_path, prefix):
    """
    Wrapper function to extract the tile and call match_template.
    Adjusts match coordinates for their position in the original image.
    """

    # tile creation
    tile = large_image[y:y+tile_size_y, x:x+tile_size_x]
    tile_path = tile_base_path / f"{prefix}_tile_{x}_{y}.jpg"
    a = cv2.imwrite(str(tile_path), tile)

    return tile_path
