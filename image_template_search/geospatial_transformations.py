import json

import shapely
from pyproj import Proj, transform
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, box
from shapely.geometry.geo import mapping
from shapely.ops import transform
from pyproj import Transformer
from pathlib import Path

import shapely
import rasterio
from pyproj import Transformer
from rasterio import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling


def project_orthomsaic(orthomosaic_path: Path, proj_orthomosaic_path: Path, target_crs = "EPSG:4326"):
    """
    Project an orthomosaic to a different CRS
    :param orthomosaic_path:
    :param proj_orthomosaic_path:
    :param target_crs:
    :return:
    """
    if isinstance(target_crs, str):
        target_crs = CRS({'init': target_crs})

    with rasterio.open(orthomosaic_path) as src:
        # Calculate the transform and dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Update metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Open the output file and reproject each band
        with rasterio.open(proj_orthomosaic_path, "w", **out_meta) as dest:
            for i in range(1, src.count + 1):  # Loop through each band
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dest, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # Use nearest or another method as needed
                )


def convert_point_crs(point: shapely.Point, target_crs: str, source_crs: str = "EPSG:4326") -> shapely.Point:
    """
    Convert a point from one CRS to another
    :param point: shapely.Point
    :param source_crs: str
    :param target_crs: str
    :return: shapely.Point
    """


    """
    Clip an orthomosaic by a location and buffer size

    :param location: tuple (lat, lon)
    :param orthomosaic_path: Path to orthomosaic
    :param buffer_size: Buffer size in meters
    :return: Path to clipped orthomosaic
    """

    # Transformer for EPSG:4326 to EPSG:32715
    transformer_to_32715 = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    transformer_to_4326 = Transformer.from_crs(target_crs, source_crs, always_xy=True)

    # Project the point to EPSG:32715
    projected_point = transform(transformer_to_32715.transform, point)

    return projected_point


def create_buffer_box(projected_point: shapely.Point, buffer_distance: int) -> shapely.geometry.box:
    """

    :param location:
    :param buffer_distance:
    :param local_epsg:
    :return:
    """
    # Create a 100m buffer box around the projected point
    bounding_box = box(
        projected_point.x - buffer_distance,
        projected_point.y - buffer_distance,
        projected_point.x + buffer_distance,
        projected_point.y + buffer_distance
    )

    return bounding_box


def save_polygon_as_geojson(polygon: shapely.geometry.Polygon, file_path: Path, EPSG_code: int):
    """
    Save a polygon as a GeoJSON file
    :param polygon: shapely.geometry.Polygon
    :param file_path: Path
    :return:
    """
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{EPSG_code}"
            }
        },
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {}
            }
        ]
    }

    # Write to GeoJSON file
    with open(file_path, "w") as f:
        json.dump(geojson_data, f)
