"""
Short Snippet: Register a simple image to an orthomosaic using a homography matrix.

This is a bit different to getting the homography between two non-geospatial images

"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS

from image_template_search.image_patch_finder import ImagePatchFinderLG
from image_template_search.util.util import visualise_image, visualise_polygons


def georeference_image(image_path, orthomosaic_path, M, output_path=None):
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

    # Apply homography to warp the image to orthomosaic pixel space
    warped_image = cv2.warpPerspective(
        image,
        M,
        (ortho_width, ortho_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Convert BGR to RGB for rasterio
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    # Create alpha channel - transparent where all RGB channels are 0 (black)
    alpha_channel = np.where(
        (warped_image_rgb[:, :, 0] == 0)
        & (warped_image_rgb[:, :, 1] == 0)
        & (warped_image_rgb[:, :, 2] == 0),
        0,  # Transparent (0) where black
        255,  # Opaque (255) elsewhere
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
        "w",
        driver="GTiff",
        height=ortho_height,
        width=ortho_width,
        count=4,  # RGBA channels
        dtype=warped_image_rgba.dtype,
        crs=ortho_crs,
        transform=ortho_transform,
        compress="lzw",  # Optional compression
        tiled=True,  # Optional tiling for better performance
        blockxsize=512,
        blockysize=512,
        nodata=0,  # Set nodata value for transparency
    ) as dst:
        # Write each channel (RGB + Alpha)
        for i in range(4):
            dst.write(warped_image_rgba[:, :, i], i + 1)

        # Set alpha band interpretation
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]

        # Add metadata
        dst.update_tags(
            source_image=str(image_path),
            orthomosaic_reference=str(orthomosaic_path),
            transformation_applied="homography_registration",
        )

    logger.info(f"Georeferenced image saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    drone_image_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Marchena/MBN01_06122021/Mar_MBN01_DJI_0918_06122021_Nazca.JPG"
    )
    drone_image_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC07_22012021/Flo_FLPC07_DJI_0051_22012021.JPG"
    )
    # orthomosaic_crop_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Marchena/MBN01_06122021/Mar_MBN01_DJI_0919_06122021_Nazca.JPG")
    orthomosaic_crop_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Mar_MBN01_06122021.tif"
    )
    orthomosaic_crop_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC07_22012021.tif"
    )
    # orthomosaic_crop_path = Path(
    #     "'/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/Flo/Flo_FLPC07_22012021.tif'")

    output_path = Path(drone_image_path.name).with_suffix(".tif")

    ipf = ImagePatchFinderLG(
        template_path=drone_image_path, large_image_path=orthomosaic_crop_path
    )

    ipf.find_patch()
    ax_i = visualise_image(
        image_path=ipf.large_image_path,
        show=False,
        dpi=150,
        title="Projected Orthomosaic",
    )
    visualise_polygons(
        polygons=[ipf.proj_template_polygon],
        ax=ax_i,
        show=True,
        color="red",
        linewidth=4,
    )

    georeference_image(
        image_path=drone_image_path,
        orthomosaic_path=orthomosaic_crop_path,
        M=ipf.M,
        output_path=output_path,
    )
