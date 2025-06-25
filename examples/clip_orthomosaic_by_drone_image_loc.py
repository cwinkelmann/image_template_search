"""
Use the position of a drone image to clip an orthomosaic.
This is useful when you have a drone image and want to clip the orthomosaic to the same area as the drone image.

"""
if __name__ == "__main__":


    from pathlib import Path
    from rasterio import CRS
    import rasterio
    import shapely.geometry
    from loguru import logger

    from image_template_search.clip_by_location import clip_orthomoasic_by_location
    from image_template_search.geospatial_transformations import convert_point_crs, create_buffer_box, \
        save_polygon_as_geojson, project_orthomsaic
    from image_template_search.util.util import get_exif_metadata

    # temporary files base path for output files
    local_base_path = Path("/Users/christian/Downloads")

    # location of the input files
    base_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/San_STJB01_10012023/")
    # drone image for the reference location
    drone_image_path = base_path / "template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
    orthomosaic_path = base_path / "San_STJB01_10012023_orthomosaic_Agisoft.tif"  # Metashape
    # orthomosaic_path = base_path / "Snt_STJB01to05_10012023_DDeploy.tif"  # DroneDeploy


    base_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/Fer_FCD01-02-03_20122021")
    drone_image_path = base_path / "template_images/Fer_FCD01-02-03_20122021_single_images/DJI_0366.JPG"
    orthomosaic_path = base_path / "Metashape_FCD01-02-03-orthomosaic.tif"  # Metashape




    buffer_distance = 100  # buffer distance in meters

    orthomosaic_proj_path = local_base_path  / f"{orthomosaic_path.stem}_proj.tif"

    # clipped orthomosaic
    orthomosaic_crop_path = local_base_path / f"{orthomosaic_path.stem}_cropped.tif"

    # project_orthomsaic(orthomosaic_path, orthomosaic_proj_path, target_crs="EPSG:4326")

    image_meta_data = get_exif_metadata(drone_image_path)
    location_long_lat = shapely.Point(image_meta_data.longitude, image_meta_data.latitude)

    logger.info(f"Drone image location (lon, lat): {location_long_lat}")

    with rasterio.open(str(orthomosaic_path)) as src:
        crs = src.crs
        logger.info(f"Orthomosaic CRS: {crs}")
        epsg = crs.to_epsg()
        logger.info(f"Orthomosaic EPSG: {epsg}")

        if epsg != 4326:
            projected_point = convert_point_crs(location_long_lat, target_crs=crs, source_crs="EPSG:4326")
            buffer = create_buffer_box(projected_point, buffer_distance=buffer_distance)

        else:
            logger.warning(f"The EPSG Code is 4326 we need to assume EPSG:32715 for the Galapagos")
            target_crs = CRS({'init': "EPSG:32715"})
            projected_point = convert_point_crs(location_long_lat, target_crs=target_crs.__str__(), source_crs="EPSG:4326")
            buffer = create_buffer_box(projected_point, buffer_distance=buffer_distance)

            buffer = convert_point_crs(buffer, source_crs=target_crs.__str__(), target_crs="EPSG:4326")



    save_polygon_as_geojson(buffer, local_base_path / f"{drone_image_path.stem}_{buffer_distance}_buffer.geojson", EPSG_code=epsg)

    clip_orthomoasic_by_location(bounding_box=buffer,
                                 orthomosaic_path=orthomosaic_path,
                                 cropped_orthomosaic_path=orthomosaic_crop_path)

    logger.info(f"Clipped orthomosaic saved to {orthomosaic_crop_path}")