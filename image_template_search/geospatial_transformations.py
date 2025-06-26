from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import json
import rasterio
import shapely
import time
import typing
from loguru import logger
from pathlib import Path
from pyproj import Transformer
from pyproj import transform
from rasterio import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.geo import mapping
from shapely.ops import transform
from osgeo import gdal


def project_orthomsaic(orthomosaic_path: Path, proj_orthomosaic_path: Path, target_crs = "EPSG:4326", resampling_method: str = "cubic",
                                 compress: str = "lzw",
                                 tiled: bool = True,
                                 blocksize: int = 512):
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
            "height": height,
            "compress": compress,
            "tiled": tiled,
            "blockxsize": blocksize,
            "blockysize": blocksize
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
                    resampling=Resampling.cubic,
                    num_threads=4 # Use nearest or another method as needed
                )


def convert_point_crs(point: shapely.Point, target_crs: str, source_crs: str = "EPSG:4326") -> shapely.Point:
    """
    Convert a point from one CRS to another
    :param point: shapely.Point
    :param source_crs: str (default: "EPSG:4326")
    :param target_crs: str
    :return: shapely.Point
    """

    # Transformer for source CRS to target CRS
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Project the point to target CRS
    projected_point = transform(transformer.transform, point)

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

def warp_to_epsg(input_file: Path, output_file: Path, target_epsg: str, overwrite: bool = False, overviews=(2, 4, 8, 16, 32)):
    """Reproject raster to target EPSG"""

    if not output_file.exists() or overwrite:

        logger.info(f"Warping {input_file} to {target_epsg}")
        start_ts = time.time()
        gdal.Warp(
            str(output_file),
            str(input_file),
            dstSRS=f"{target_epsg}",
            resampleAlg="bilinear",
            format="GTiff",  # temporary GeoTIFF
            creationOptions=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "BIGTIFF=YES",
                "BLOCKXSIZE=1024",
                "BLOCKYSIZE=1024"
            ]
        )
        logger.info(f"Warped to {output_file} in {time.time() - start_ts:.2f} seconds")

        ds = gdal.Open("output.tif", gdal.GA_Update)  # Must be in update mode
        overview_count = ds.GetRasterBand(1).GetOverviewCount()
        if overview_count == 0 and len(overviews):
            ds.BuildOverviews("NEAREST", overviews )
        ds = None  # Close the dataset
    else:
        logger.info(f"Warped File already exists: {output_file}")


def batch_warp_to_epsg(input_files: typing.List[Path],
                        target_epsg: str,
                       output_files: typing.Optional[typing.List[Path]] = None,
                       output_dir: typing.Optional[Path] = None,

                       overwrite: bool = False,
                       overviews=(2, 4, 8, 16, 32),
                       max_workers=4):

    """Parallelize the reprojection of multiple rasters to target EPSG."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        if output_files is not None:
            for input_file, output_file in zip(input_files, output_files):
                futures.append(executor.submit(warp_to_epsg, input_file, output_file, target_epsg, overwrite, overviews))
        else:
            for input_file in input_files:
                output_file = output_dir / f"{Path(input_file).stem}.tif"
                futures.append(executor.submit(warp_to_epsg, input_file, output_file, target_epsg, overwrite, overviews))

        # Wait for all tasks to complete
        for future in futures:
            future.result()    ,


def convert_to_cog(input_file: Path, output_file: Path, overwrite: bool = False):
    """
    Convert a raster to a Cloud-Optimized COG GeoTIFF
    :param input_file:
    :param output_file:
    :return:
    """
    try:
        if not output_file.exists() or overwrite:

            cog_options = {
                # "BLOCKXSIZE": 1024,  # Tile width
                # "BLOCKYSIZE": 1024,  # Tile height
                # "TILED": True,  # Enable tiling
                "COMPRESS": "LZW",  # Compression type (LZW is common for COGs)
                # "COPY_SRC_OVERVIEWS": True,  # Copy overviews if they exist
                # "BIGTIFF": True  # Use BigTIFF format for large files
            }
            logger.info(f"Converting {input_file} to COG")
            start_ts = time.time()
            # Open the input file
            with rasterio.open(input_file) as src:
                # Copy the input file to a COG with updated options
                copy(
                    src,
                    output_file,
                    driver="COG",
                    **cog_options
                )
            logger.info(f"COG saved to {output_file} in {time.time() - start_ts:.2f} seconds")
        else:
            logger.info(f"COG already exists: {output_file}")

    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Error converting {input_file} to COG: {e}")


def convert_to_tiled_geotiff(input_file: Path, output_file: Path, overwrite: bool = False):
    """
    Convert a raster to a simple tiled GeoTIFF (COG)
    :param input_file:
    :param output_file:
    :return:
    """
    try:
        if not output_file.exists() or overwrite:

            cog_options = {
                "BLOCKXSIZE": 1024,  # Tile width
                "BLOCKYSIZE": 1024,  # Tile height
                "TILED": True,  # Enable tiling
                "COMPRESS": "LZW",  # Compression type (LZW is common for COGs)
                "COPY_SRC_OVERVIEWS": True,  # Copy overviews if they exist
                "BIGTIFF": True  # Use BigTIFF format for large files
            }
            logger.info(f"Converting {input_file} to COG")
            start_ts = time.time()
            # Open the input file
            with rasterio.open(input_file) as src:
                # Copy the input file to a COG with updated options
                copy(
                    src,
                    output_file,
                    driver="GEOTIFF",  # Use GeoTIFF driver for tiled output
                    **cog_options
                )
            logger.info(f"Tiled GEOTIFF saved to {output_file} in {time.time() - start_ts:.2f} seconds")
        else:
            logger.info(f"Tiled GEOTIFF already exists: {output_file}")

    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Error converting {input_file} to Tiled GEOTIFF: {e}")


def batch_convert_to_cog(input_files: typing.List[Path],
                         output_files: typing.Optional[typing.List[Path]] = None,
                         output_dir: typing.Optional[Path] = None, overwrite: bool = False,
                         max_workers=4):
    """Parallelize the conversion of multiple files to COG."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        if output_files is not None:
            for input_file, output_file in zip(input_files, output_files):
                # output_file = output_dir / f"{Path(input_file).stem}.tif"
                futures.append(executor.submit(convert_to_cog, input_file, output_file, overwrite))
        else:
            for input_file in input_files:
                output_file = output_dir / f"{Path(input_file).stem}.tif"
                futures.append(executor.submit(convert_to_cog, input_file, output_file, overwrite))

        # Wait for all tasks to complete
        for future in futures:
            future.result()


def create_universal_grid(extent, cell_size_km, crs="EPSG:32715"):
    """
    Creates a regular grid within a specified extent.

    Args:
        extent (tuple): (xmin, ymin, xmax, ymax) of the bounding box.
        cell_size_km (float): Edge length of the grid cells in kilometers.
        crs (str): Coordinate Reference System (e.g., EPSG:32715 for UTM Zone 15S).

    Returns:
        GeoDataFrame: A GeoDataFrame containing the grid.
    """

    xmin, ymin, xmax, ymax = extent
    cell_size = cell_size_km * 1000  # Convert km to meters

    # Generate grid cells
    grid_cells = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size

    # Create GeoDataFrame
    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=crs)
    return grid




def create_tiles(input_file, output_dir, tile_size=512):
    """Split a large raster into smaller physical tiles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cog_options = {
        "BLOCKXSIZE": 1024,  # Internal tile size for COGs
        "BLOCKYSIZE": 1024,
        "TILED": True,
        "COMPRESS": "LZW",  # Compression for COGs
    }

    with rasterio.open(input_file) as src:
        width, height = src.width, src.height
        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                # Define window for the tile
                window = Window(i, j, tile_size, tile_size)
                transform = src.window_transform(window)

                # Read data and write to a new file
                tile_file = output_dir / f"{input_file.stem}_tile_{i}_{j}.tif"
                meta = src.meta.copy()
                meta.update({
                    "driver": "COG",  # Save directly as COG
                    "height": min(tile_size, height - j),
                    "width": min(tile_size, width - i),
                    "transform": transform,
                    **cog_options,  # Add COG-specific options
                })

                with rasterio.open(tile_file, "w", **meta) as dst:
                    logger.info(f"Writing {tile_file}")
                    dst.write(src.read(window=window))

                    logger.info(f"Adding overviews to {tile_file}")
                    # Add overviews
                    overviews = [2, 4, 8]  # Define overview levels
                    dst.build_overviews(overviews, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")
                logger.info(f"Saved {tile_file}")



def get_gsd(geotiff_path):
    """Calculate the Ground Sampling Distance (GSD) of a GeoTIFF."""
    with rasterio.open(geotiff_path) as dataset:
        # The affine transformation provides the pixel size
        transform = dataset.transform

        # Pixel resolution in x and y direction
        gsd_x = transform.a  # Width of a pixel (East-West direction)
        gsd_y = -transform.e  # Height of a pixel (North-South direction, usually negative)

        # CRS info
        crs = dataset.crs
        print(f"CRS: {crs}")

    return gsd_x, gsd_y


def get_geotiff_compression(geotiff_path):
    """Extract compression type from a GeoTIFF file."""
    with rasterio.open(geotiff_path) as dataset:
        compression = dataset.profile.get('compress', 'None')  # Get compression info
        return compression