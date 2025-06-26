"""
Georeference a drone image to an orthomosaic using a homography matrix.

This allows for overlaying them on an orthomosaic, which is useful for visualizing the drone image in its geospatial context and the quality difference of both images.
This is similar to workflow_iguana_deduplication.py
"""

import gc
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
import shapely
from loguru import logger
from pyproj import CRS

from image_template_search.clip_by_location import clip_orthomoasic_by_location
from image_template_search.geospatial_transformations import (
    convert_to_cog,
    create_buffer_box,
    convert_point_crs,
    project_orthomsaic,
)
from image_template_search.image_patch_finder import ImagePatchFinderLG
from image_template_search.util.util import (
    get_exif_metadata,
    list_images,
)


def calculate_optimal_output_resolution(
    image_shape, M, ortho_transform, min_resolution_factor=1.0
):
    """
    Calculate optimal output resolution to preserve image quality.

    :param image_shape: (height, width) of source image
    :param M: Homography matrix
    :param ortho_transform: Orthomosaic geotransform
    :param min_resolution_factor: Minimum resolution relative to orthomosaic (1.0 = same, 2.0 = 2x, etc.)
    :return: (output_width, output_height, adjusted_transform)
    """
    img_height, img_width = image_shape[:2]

    # Get corners of source image
    corners = np.array(
        [[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # Transform corners to orthomosaic space
    transformed_corners = cv2.perspectiveTransform(corners, M)
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Calculate bounding box in orthomosaic pixel space
    min_x, min_y = np.min(transformed_corners, axis=0)
    max_x, max_y = np.max(transformed_corners, axis=0)

    # Calculate the scale factor based on transformation
    # This estimates how much the image is scaled during transformation
    src_area = img_width * img_height
    dst_width = max_x - min_x
    dst_height = max_y - min_y
    dst_area = dst_width * dst_height

    scale_factor = np.sqrt(dst_area / src_area)

    # Calculate pixel resolution in meters (from orthomosaic)
    ortho_pixel_size = abs(ortho_transform.a)  # Assuming square pixels

    # Calculate optimal resolution to preserve detail
    # We want at least the same effective resolution as the source image
    optimal_pixel_size = ortho_pixel_size / max(min_resolution_factor, scale_factor)

    # Calculate output dimensions
    output_width = int(dst_width * (ortho_pixel_size / optimal_pixel_size))
    output_height = int(dst_height * (ortho_pixel_size / optimal_pixel_size))

    # Create adjusted transform for the higher resolution output
    adjusted_transform = rasterio.transform.from_bounds(
        min_x * ortho_transform.a + ortho_transform.c,  # west
        (min_y + dst_height) * ortho_transform.e + ortho_transform.f,  # south
        (min_x + dst_width) * ortho_transform.a + ortho_transform.c,  # east
        min_y * ortho_transform.e + ortho_transform.f,  # north
        output_width,
        output_height,
    )

    # Adjust homography matrix for the new output dimensions
    scale_x = output_width / dst_width
    scale_y = output_height / dst_height

    scale_matrix = np.array(
        [[scale_x, 0, -min_x * scale_x], [0, scale_y, -min_y * scale_y], [0, 0, 1]],
        dtype=np.float64,
    )

    adjusted_M = scale_matrix @ M

    return output_width, output_height, adjusted_transform, adjusted_M


def georeference_image_adaptive_resolution(
    image_path,
    orthomosaic_path,
    M,
    interpolation="cubic",
    anti_aliasing=True,
    resolution_factor=2.0,
    output_path=None,
):
    """
    Georeference with adaptive resolution to preserve image quality.

    :param image_path: Path to high-res drone image
    :param orthomosaic_path: Path to orthomosaic
    :param M: Homography matrix
    :param interpolation: Interpolation method
    :param anti_aliasing: Apply anti-aliasing
    :param resolution_factor: Resolution multiplier relative to orthomosaic (2.0 = 2x resolution)
    :param output_path: Output path
    :return: Path to georeferenced image
    """

    # Read the input image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Optional anti-aliasing
    if anti_aliasing:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)

    # Get orthomosaic information
    with rasterio.open(orthomosaic_path) as ortho_src:
        ortho_transform = ortho_src.transform
        ortho_crs = ortho_src.crs

    # Calculate optimal output resolution
    output_width, output_height, adjusted_transform, adjusted_M = (
        calculate_optimal_output_resolution(
            image.shape, M, ortho_transform, min_resolution_factor=resolution_factor
        )
    )

    logger.info(f"Source image: {image.shape[1]}×{image.shape[0]}")
    logger.info(f"Output image: {output_width}×{output_height}")
    logger.info(
        f"Resolution preserved: {(output_width * output_height) / (image.shape[1] * image.shape[0]):.2f}x"
    )

    # Set up interpolation
    interp_methods = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }
    interp_flag = interp_methods.get(interpolation.lower(), cv2.INTER_CUBIC)

    # Apply transformation with optimal resolution
    warped_image_bgr = cv2.warpPerspective(
        image,
        adjusted_M,
        (output_width, output_height),
        flags=interp_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Convert BGR to RGB
    warped_image_rgb = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2RGB)

    # Create alpha channel
    alpha_channel = np.where(
        (warped_image_rgb[:, :, 0] == 0)
        & (warped_image_rgb[:, :, 1] == 0)
        & (warped_image_rgb[:, :, 2] == 0),
        0,
        255,
    ).astype(np.uint8)

    warped_image_rgba = np.dstack([warped_image_rgb, alpha_channel])

    # Create output path
    if output_path is None:
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"georeferenced_hires_{Path(image_path).stem}.tif"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = (
                output_path / f"georeferenced_hires_{Path(image_path).stem}.tif"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with adaptive resolution and transform
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output_height,
        width=output_width,
        count=4,
        dtype=warped_image_rgba.dtype,
        crs=ortho_crs,
        transform=adjusted_transform,  # Use adjusted transform
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=0,
    ) as dst:
        for i in range(4):
            dst.write(warped_image_rgba[:, :, i], i + 1)

        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]

        dst.update_tags(
            source_image=str(image_path),
            orthomosaic_reference=str(orthomosaic_path),
            transformation_applied="homography_registration_adaptive_resolution",
            resolution_factor=str(resolution_factor),
        )

    logger.info(f"High-resolution georeferenced image saved to: {output_path}")
    return str(output_path)


def georeference_image_max_quality(image_path, orthomosaic_path, M, output_path=None):
    """Maximum quality georeferencing - preserves as much resolution as possible."""
    return georeference_image_adaptive_resolution(
        image_path,
        orthomosaic_path,
        M,
        interpolation="cubic",
        anti_aliasing=True,
        resolution_factor=1.0,  # 3x orthomosaic resolution
        output_path=output_path,
    )


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
        interpolation="cubic",  # Best quality interpolation
        anti_aliasing=False,  # Enable anti-aliasing
        supersampling_factor=1,  # 2x supersampling for maximum quality
    )


def georeference_image(
    image_path,
    orthomosaic_path,
    M,
    interpolation,
    supersampling_factor,
    anti_aliasing,
    output_path=None,
):
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
        scale_matrix = np.array(
            [[supersampling_factor, 0, 0], [0, supersampling_factor, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        M_scaled = scale_matrix @ M
    else:
        M_scaled = M

    # Add this mapping at the beginning of the function
    interp_methods = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }

    interp_flag = interp_methods.get(interpolation.lower(), cv2.INTER_CUBIC)

    # Then use interp_flag instead of interpolation
    warped_image = cv2.warpPerspective(
        image,
        M_scaled,
        (output_width, output_height),
        flags=interp_flag,  # Changed this line
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Replace the downsampling section
    if supersampling_factor > 1:
        if anti_aliasing:
            kernel_size = supersampling_factor * 2 + 1
            warped_image = cv2.GaussianBlur(
                warped_image, (kernel_size, kernel_size), supersampling_factor * 0.5
            )

        # Use higher quality downsampling
        warped_image = cv2.resize(
            warped_image, (ortho_width, ortho_height), interpolation=cv2.INTER_CUBIC
        )  # Changed from INTER_AREA

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


def debug_homography_transformation(image_path, orthomosaic_path, M):
    """
    Debug function to check homography transformation results.
    """
    image = cv2.imread(str(image_path))

    with rasterio.open(orthomosaic_path) as ortho_src:
        ortho_width = ortho_src.width
        ortho_height = ortho_src.height

    # Get image corners
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
        -1, 1, 2
    )

    # Transform corners
    transformed_corners = cv2.perspectiveTransform(corners, M)
    transformed_corners = transformed_corners.reshape(-1, 2)

    logger.info(f"Source image size: {w}×{h}")
    logger.info(f"Orthomosaic size: {ortho_width}×{ortho_height}")
    logger.info(f"Transformed corners: {transformed_corners}")
    logger.info(
        f"Corner bounds: x=[{transformed_corners[:, 0].min():.1f}, {transformed_corners[:, 0].max():.1f}], "
        f"y=[{transformed_corners[:, 1].min():.1f}, {transformed_corners[:, 1].max():.1f}]"
    )

    # Check if corners are within orthomosaic bounds
    in_bounds = (
        (transformed_corners[:, 0] >= 0)
        & (transformed_corners[:, 0] < ortho_width)
        & (transformed_corners[:, 1] >= 0)
        & (transformed_corners[:, 1] < ortho_height)
    )
    logger.info(f"Corners in bounds: {in_bounds}")

    return transformed_corners


def save_array_as_jpg(image_array, output_path, debug_info=True):
    """
    Save a numpy array as a regular JPG file for debugging.

    :param image_array: Numpy array (BGR, RGB, or grayscale)
    :param output_path: Path to save JPG
    :param debug_info: Print debug information about the array
    """
    output_path = Path(output_path)

    if debug_info:
        logger.info(f"Array shape: {image_array.shape}")
        logger.info(f"Array dtype: {image_array.dtype}")
        logger.info(f"Array min/max: {image_array.min()}/{image_array.max()}")
        logger.info(f"Non-zero pixels: {np.count_nonzero(image_array)}")
        logger.info(f"Unique values count: {len(np.unique(image_array))}")

    # Handle different array formats
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:  # RGBA
            # Convert RGBA to RGB (remove alpha channel)
            image_rgb = image_array[:, :, :3]
            logger.info("Converted RGBA to RGB")
        elif image_array.shape[2] == 3:  # RGB or BGR
            image_rgb = image_array
        else:
            raise ValueError(f"Unexpected number of channels: {image_array.shape[2]}")
    elif len(image_array.shape) == 2:  # Grayscale
        image_rgb = image_array
    else:
        raise ValueError(f"Unexpected array shape: {image_array.shape}")

    # Ensure correct data type for saving
    if image_rgb.dtype != np.uint8:
        # Normalize to 0-255 if needed
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
            logger.info("Normalized float array to uint8")
        else:
            image_rgb = image_rgb.astype(np.uint8)
            logger.info("Converted to uint8")

    # Save using OpenCV (expects BGR format)
    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
        # If this is RGB, convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_rgb

    success = cv2.imwrite(str(output_path), image_bgr)

    if success:
        logger.info(f"Successfully saved debug image to: {output_path}")
        # Verify the saved file
        file_size = output_path.stat().st_size
        logger.info(f"Saved file size: {file_size} bytes")
    else:
        logger.error(f"Failed to save image to: {output_path}")

    return success


def georeference_image_simple_high_res(
    image_path,
    orthomosaic_path,
    M,
    scale_factor=2.0,
    interpolation="cubic",
    output_path=None,
) -> Path:
    """
    Simple high-resolution georeferencing that works well with cropped orthomosaics.

    :param image_path: Path to drone image
    :param orthomosaic_path: Path to (possibly cropped) orthomosaic
    :param M: Homography matrix from drone image to orthomosaic pixels
    :param scale_factor: Output resolution multiplier (2.0 = 2x orthomosaic resolution)
    :param interpolation: Interpolation method
    :param output_path: Output path
    :return: Path to georeferenced image
    """

    # Read the input image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Get orthomosaic information
    with rasterio.open(orthomosaic_path) as ortho_src:
        ortho_transform = ortho_src.transform
        ortho_crs = ortho_src.crs
        ortho_width = ortho_src.width
        ortho_height = ortho_src.height

    # Calculate output dimensions (simple scaling)
    output_width = int(ortho_width * scale_factor)
    output_height = int(ortho_height * scale_factor)

    # Scale the homography matrix for higher resolution output
    scale_matrix = np.array(
        [[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=np.float64
    )

    M_scaled = scale_matrix @ M

    # Set up interpolation
    interp_methods = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }
    interp_flag = interp_methods.get(interpolation.lower(), cv2.INTER_CUBIC)

    # Apply transformation
    warped_image_bgr = cv2.warpPerspective(
        image,
        M_scaled,
        (output_width, output_height),
        flags=interp_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    logger.info(f"Source image: {image.shape[1]}×{image.shape[0]}")
    logger.info(f"Output image: {output_width}×{output_height}")
    logger.info(f"Non-zero pixels: {np.count_nonzero(warped_image_bgr)}")

    # Convert BGR to RGB
    warped_image_rgb = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2RGB)

    # Create alpha channel
    alpha_channel = np.where(
        (warped_image_rgb[:, :, 0] == 0)
        & (warped_image_rgb[:, :, 1] == 0)
        & (warped_image_rgb[:, :, 2] == 0),
        0,
        255,
    ).astype(np.uint8)

    warped_image_rgba = np.dstack([warped_image_rgb, alpha_channel])

    # Create scaled transform (simple pixel size adjustment)
    scaled_transform = rasterio.transform.Affine(
        ortho_transform.a / scale_factor,  # Smaller pixel size (higher resolution)
        ortho_transform.b,
        ortho_transform.c,
        ortho_transform.d,
        ortho_transform.e / scale_factor,  # Smaller pixel size (higher resolution)
        ortho_transform.f,
    )

    # Create output path
    if output_path is None:
        temp_dir = Path(tempfile.gettempdir())
        output_path = temp_dir / f"georeferenced_simple_{Path(image_path).stem}.tif"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = (
                output_path / f"georeferenced_simple_{Path(image_path).stem}.tif"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save the transformed image as a jpeg for debugging
    debug_image_path = output_path.with_suffix(".jpg")
    save_array_as_jpg(warped_image_rgb, debug_image_path)

    # Write the result
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output_height,
        width=output_width,
        count=4,
        dtype=warped_image_rgba.dtype,
        crs=ortho_crs,
        transform=scaled_transform,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=0,
    ) as dst:
        for i in range(4):
            dst.write(warped_image_rgba[:, :, i], i + 1)

        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]

        dst.update_tags(
            source_image=str(image_path),
            orthomosaic_reference=str(orthomosaic_path),
            transformation_applied="simple_high_resolution_georeferencing",
            scale_factor=str(scale_factor),
        )

    logger.info(f"Georeferenced image saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    EPSG_STRING: str = "EPSG:32715"

    drone_image_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC07_22012021"
    )
    drone_image_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC06_22012021"
    )

    drone_image_paths = list_images(drone_image_path, extension="JPG", recursive=False)

    orthomosaic_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Mar_MBN01_06122021.tif"
    )
    orthomosaic_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC07_22012021.tif"
    )

    orthomosaic_path = Path("/Users/christian/Downloads/Flo_FLPC07_22012021_MS.tif")
    proj_orthomosaic_path = Path(
        "/Users/christian/Downloads/Flo_FLPC07_22012021_MS_32715.tif"
    )

    orthomosaic_path = Path("/Users/christian/Downloads/Flo_FLPC06_22012021_MS.tif")
    proj_orthomosaic_path = Path(
        "/Users/christian/Downloads/Flo_FLPC06_22012021_MS_32715.tif"
    )

    output_base_path = (
        Path("/Volumes/2TB/projected_drone_images_MS") / drone_image_path.name
    )
    output_base_path.mkdir(parents=True, exist_ok=True)

    for drone_image_path in drone_image_paths:
        orthomosaic_crop_path = Path(
            f"{orthomosaic_path.name}_cropped_{drone_image_path.stem}.tif"
        )

        output_path = output_base_path / Path(drone_image_path.name).with_suffix(
            ".tiff"
        )
        output_cog_path = (
            output_base_path / "cog" / Path(drone_image_path.name).with_suffix(".tiff")
        )
        if output_cog_path.exists():
            logger.info(
                f"Output path {output_path} already exists. Skipping processing for {drone_image_path}."
            )
            continue

        image_meta_data = get_exif_metadata(drone_image_path)
        location_long_lat = shapely.Point(
            image_meta_data.longitude, image_meta_data.latitude
        )

        target_crs = CRS({"init": EPSG_STRING})
        if proj_orthomosaic_path.exists():
            orthomosaic_path = proj_orthomosaic_path
        else:
            project_orthomsaic(
                orthomosaic_path=orthomosaic_path,
                proj_orthomosaic_path=proj_orthomosaic_path,
                target_crs=EPSG_STRING,
            )
            orthomosaic_path = proj_orthomosaic_path

        projected_point = convert_point_crs(
            location_long_lat,
            target_crs=target_crs.__str__(),
            source_crs="EPSG:4326",
        )
        buffer = create_buffer_box(projected_point, buffer_distance=25)

        clip_orthomoasic_by_location(
            bounding_box=buffer,
            orthomosaic_path=orthomosaic_path,
            cropped_orthomosaic_path=orthomosaic_crop_path,
        )

        logger.info(f"Clipped orthomosaic saved to {orthomosaic_crop_path}")

        ipf = ImagePatchFinderLG(
            template_path=drone_image_path, large_image_path=orthomosaic_crop_path
        )

        ipf.find_patch()

        debug_homography_transformation(drone_image_path, orthomosaic_crop_path, ipf.M)

        saved_geotiff = georeference_image_simple_high_res(
            image_path=drone_image_path,
            orthomosaic_path=orthomosaic_crop_path,
            M=ipf.M,
            output_path=output_path,
            scale_factor=1.2,
        )

        convert_to_cog(
            input_file=saved_geotiff, output_file=output_cog_path, overwrite=False
        )

        orthomosaic_crop_path.unlink()

        gc.collect()
