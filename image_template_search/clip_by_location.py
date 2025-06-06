"""
Clip the approximate location from an orthomosaic in order to make finding a template easier
"""
from pathlib import Path

import shapely
import rasterio





def clip_orthomoasic_by_location(bounding_box: shapely.Polygon,
                                 orthomosaic_path: Path,
                                 cropped_orthomosaic_path: Path):
    """
    Clip an orthomosaic by a location and buffer size

    :param location: tuple (lat, lon)
    :param orthomosaic_path: Path to orthomosaic
    :param buffer_size: Buffer size in meters
    :return: Path to clipped orthomosaic
    """

    from rasterio.mask import mask

    with rasterio.open(str(orthomosaic_path)) as src:
        crs = src.crs
        crs.to_epsg()
        # Reproject the bounding box to the source CRS (EPSG:4326) to use for masking
        # Convert the bounding box to GeoJSON format for rasterio mask
        bounding_box_geojson = [bounding_box.__geo_interface__]

        # Crop the raster using the reprojected bounding box
        out_image, out_transform = mask(src, bounding_box_geojson, crop=True)

        # Update metadata for the cropped output
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": crs.to_epsg()
        })

    with rasterio.open(cropped_orthomosaic_path, "w", **out_meta) as dest:
        dest.write(out_image)


