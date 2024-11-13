"""
Clip the approximate location from an orthomosaic in order to make finding a template easier
"""


from pathlib import Path

import shapely.geometry

from image_template_search.util.util import get_exif_metadata

def project_orthomsaic(orthomosaic_path: Path, proj_orthomosaic_path: Path, target_crs = "EPSG:4326"):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling


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


def clip_orthomoasic_by_location(location: shapely.Point, orthomosaic_path: Path,
                                 buffer_distance: int, cropped_orthomosaic_path: Path,
                                 local_epsg="EPSG:32715"):
    """
    Clip an orthomosaic by a location and buffer size
    :param location: tuple (lat, lon)
    :param orthomosaic_path: Path to orthomosaic
    :param buffer_size: Buffer size in meters
    :return: Path to clipped orthomosaic
    """
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Point, box
    from shapely.ops import transform
    from pyproj import Transformer

    # Transformer for EPSG:4326 to EPSG:32715
    transformer_to_32715 = Transformer.from_crs("EPSG:4326", local_epsg, always_xy=True)
    transformer_to_4326 = Transformer.from_crs(local_epsg, "EPSG:4326", always_xy=True)

    # Project the point to EPSG:32715
    projected_point = transform(transformer_to_32715.transform, location)

    # Create a 100m buffer box around the projected point
    bounding_box = box(
        projected_point.x - buffer_distance,
        projected_point.y - buffer_distance,
        projected_point.x + buffer_distance,
        projected_point.y + buffer_distance
    )

    with rasterio.open(str(orthomosaic_path)) as src:
        crs = src.crs
        crs.to_epsg()
        # Reproject the bounding box to the source CRS (EPSG:4326) to use for masking
        bounding_box_4326 = transform(transformer_to_4326.transform, bounding_box)

        # Convert the bounding box to GeoJSON format for rasterio mask
        bounding_box_geojson = [bounding_box_4326.__geo_interface__]

        # Crop the raster using the reprojected bounding box
        out_image, out_transform = mask(src, bounding_box_geojson, crop=True)

        # Update metadata for the cropped output
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            # "crs": "EPSG:4326"
            "crs": "EPSG:32715"
        })

    with rasterio.open(cropped_orthomosaic_path, "w", **out_meta) as dest:
        dest.write(out_image)



if __name__ == "__main__":

    base_path = Path("/Users/christian/Downloads")

    drone_image = base_path / "FCD01-02-03/template_images/Fer_FCD01-02-03_20122021_single_images/DJI_0366.JPG"
    image_2 =  base_path / "FCD01-02-03/Metashape_FCD01-02-03-orthomosaic.tif" # metashape
    image_2_crop =  base_path / "FCD01-02-03/Metashape_FCD01-02-03-orthomosaic_cropped.tif" # metashape

    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    local_base_path = Path("/Users/christian/Downloads")

    drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
    image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Pix4D.tiff" # DroneDeploy
    proj_orthomosaic_path = local_base_path / f"{image_2.stem}_proj_4326.tif"

    project_orthomsaic(orthomosaic_path=image_2, proj_orthomosaic_path=proj_orthomosaic_path)

    image_2_crop =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Pix4D_cropped.tif" # metashape

    image_meta_data = get_exif_metadata(drone_image)
    location_long_lat = shapely.Point(image_meta_data.longitude, image_meta_data.latitude)
    clip_orthomoasic_by_location(location=location_long_lat, orthomosaic_path=proj_orthomosaic_path, buffer_distance=100, cropped_orthomosaic_path = image_2_crop)