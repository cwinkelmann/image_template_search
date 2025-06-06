"""
This is similar to workflow_iguana_deduplication.py

This is a bit different to getting the homography between two non-geospatial images

"""

import cv2
import gc
import numpy as np
import rasterio
import shapely
import tempfile
from loguru import logger
from pathlib import Path
from rasterio.crs import CRS

from clip_by_location import clip_orthomoasic_by_location
from geospatial_transformations import create_buffer_box, convert_point_crs
from image_template_search.image_patch_finder import ImagePatchFinderLG
from image_template_search.util.util import get_exif_metadata, list_images


###
def georeference_image_high_quality(image_path, orthomosaic_path, M, output_path=None):
    """
    High-quality version with optimal settings for best results.

    :param image_path: Path to simple jpg image (drone image)
    :param orthomosaic_path: Path to an orthomosaic image which is georeferenced
    :param M: Homography matrix from image to orthomosaic pixel coordinates
    :param output_path: Optional output path for georeferenced image
    :return: Path to the georeferenced image file
    """
    return georeference_image(
        image_path=image_path,
        orthomosaic_path=orthomosaic_path,
        M=M,
        output_path=output_path,
        interpolation=cv2.INTER_LANCZOS4,  # Best quality interpolation
        anti_aliasing=False,  # Enable anti-aliasing
        supersampling_factor=2  # 2x supersampling for maximum quality
    )


def georeference_image(image_path, orthomosaic_path, M,
                       interpolation,
                       supersampling_factor,
                       anti_aliasing,
                       output_path=None):
    """
    Register an image to an orthomosaic using the provided transformation matrix M.

    :param image_path: Path to simple jpg image (drone image)
    :param orthomosaic_path: Path to an orthomosaic image which is georeferenced
    :param M: Homography matrix from image to orthomosaic pixel coordinates
    :param output_path: Optional output path for georeferenced image. If None, creates temp file.
    :return: Path to the georeferenced image file
    """

    # Read the input image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Get orthomosaic geospatial information
    with rasterio.open(orthomosaic_path) as ortho_src:
        ortho_transform = ortho_src.transform
        ortho_crs = ortho_src.crs
        ortho_bounds = ortho_src.bounds
        ortho_width = ortho_src.width
        ortho_height = ortho_src.height

    # Calculate output dimensions (with optional supersampling)
    output_width = ortho_width * supersampling_factor
    output_height = ortho_height * supersampling_factor
    # Adjust homography matrix for supersampling
    if supersampling_factor > 1:
        scale_matrix = np.array([
            [supersampling_factor, 0, 0],
            [0, supersampling_factor, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        M_scaled = scale_matrix @ M
    else:
        M_scaled = M

    # Apply homography to warp the image to orthomosaic pixel space
    warped_image = cv2.warpPerspective(
        image,
        M_scaled,
        (output_width, output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Downsample if we used supersampling
    if supersampling_factor > 1:
        # Apply anti-aliasing before downsampling
        if anti_aliasing:
            kernel_size = supersampling_factor * 2 + 1
            warped_image = cv2.GaussianBlur(warped_image, (kernel_size, kernel_size),
                                            supersampling_factor * 0.5)

        # Downsample back to target resolution
        warped_image = cv2.resize(warped_image, (ortho_width, ortho_height),
                                  interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB for rasterio
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    # Create alpha channel - transparent where all RGB channels are 0 (black)
    alpha_channel = np.where(
        (warped_image_rgb[:, :, 0] == 0) &
        (warped_image_rgb[:, :, 1] == 0) &
        (warped_image_rgb[:, :, 2] == 0),
        0,  # Transparent (0) where black
        255  # Opaque (255) elsewhere
    ).astype(np.uint8)

    # Combine RGB with alpha to create RGBA
    warped_image_rgba = np.dstack([warped_image_rgb, alpha_channel])

    # Create output path if not provided
    if output_path is None:
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"georeferenced_{Path(image_path).stem}.tif"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"georeferenced_{Path(image_path).stem}.tif"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the georeferenced image
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ortho_height,
            width=ortho_width,
            count=4,  # RGBA channels
            dtype=warped_image_rgba.dtype,
            crs=ortho_crs,
            transform=ortho_transform,
            compress='lzw',  # Optional compression
            tiled=True,  # Optional tiling for better performance
            blockxsize=512,
            blockysize=512,
            nodata=0  # Set nodata value for transparency
    ) as dst:
        # Write each channel (RGB + Alpha)
        for i in range(4):
            dst.write(warped_image_rgba[:, :, i], i + 1)

        # Set alpha band interpretation
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha
        ]

        # Add metadata
        dst.update_tags(
            source_image=str(image_path),
            orthomosaic_reference=str(orthomosaic_path),
            transformation_applied="homography_registration"
        )

    logger.info(f"Georeferenced image saved to: {output_path}")
    return str(output_path)


if __name__ == '__main__':

    # drone_image_path = Path(
    #     "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Marchena/MBN01_06122021/Mar_MBN01_DJI_0918_06122021_Nazca.JPG")
    # drone_image_path = Path(
    #     "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC07_22012021/Flo_FLPC07_DJI_0051_22012021.JPG")

    drone_image_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC07_22012021")
    # drone_image_path = Path(
    #     "/Volumes/G-DRIVE/Iguanas_From_Above/2021 Jan/Photos from drone/Floreana/22.01.21/FPC07/")

    drone_image_paths = list_images(drone_image_path, extension="JPG", recursive=False)

    # orthomosaic_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Marchena/MBN01_06122021/Mar_MBN01_DJI_0919_06122021_Nazca.JPG")
    orthomosaic_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Mar_MBN01_06122021.tif")
    orthomosaic_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC07_22012021.tif")
    # orthomosaic_crop_path = Path(
    #     "'/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/Flo/Flo_FLPC07_22012021.tif'")

    for drone_image_path in drone_image_paths:

        if drone_image_path.name != "Flo_FLPC07_DJI_0051_22012021.JPG":
        # if drone_image_path.name != "DJI_0051.JPG":
            continue
        orthomosaic_crop_path = Path(f"Flo_FLPC07_22012021_cropped_{drone_image_path.stem}.tif")

        output_path = Path(drone_image_path.name).with_suffix(".tiff")
        if output_path.exists():
            logger.info(f"Output path {output_path} already exists. Skipping processing for {drone_image_path}.")
            continue

        image_meta_data = get_exif_metadata(drone_image_path)
        location_long_lat = shapely.Point(
            image_meta_data.longitude, image_meta_data.latitude
        )
        target_crs = CRS({"init": "EPSG:32715"})
        projected_point = convert_point_crs(
            location_long_lat,
            target_crs=target_crs.__str__(),
            source_crs="EPSG:4326",
        )
        buffer = create_buffer_box(
            projected_point, buffer_distance=25
        )

        clip_orthomoasic_by_location(
            bounding_box=buffer,
            orthomosaic_path=orthomosaic_path,
            cropped_orthomosaic_path=orthomosaic_crop_path,
        )

        logger.info(f"Clipped orthomosaic saved to {orthomosaic_crop_path}")

        # visualise_image(
        #     image_path=orthomosaic_crop_path,
        #     show=True,
        #     dpi=75,
        #     title="Cropped and projected Mosaic image",
        #     output_file_name= f"{orthomosaic_path.stem}_cropped_by_location.jpg"
        # )

        ipf = ImagePatchFinderLG(template_path=drone_image_path,
                                 large_image_path=orthomosaic_crop_path)

        ipf.find_patch()
        # ax_i = visualise_image(image_path=ipf.large_image_path, show=False, dpi=150, title=f"Projected {drone_image_path.name} onto Orthomosaic {orthomosaic_path.name} ")
        # visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_i, show=True, color="red", linewidth=4)

        georeference_image_high_quality(image_path=drone_image_path,
                                        orthomosaic_path=orthomosaic_crop_path,
                                        M=ipf.M, output_path=output_path)

        orthomosaic_crop_path.unlink()

        # TODO collect the centroid of the projected polygon and save it to a file
        # these can be used to find the most suitable drone image for a given orthomosaic

        gc.collect()
