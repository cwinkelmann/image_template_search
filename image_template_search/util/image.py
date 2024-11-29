import numpy as np
from PIL import Image
from loguru import logger


def image_patcher(image_np: np.array, N_x: int, N_y: int):
    # Get the dimensions of the image
    height, width, channels = image_np.shape

    # Calculate tile sizes
    tile_height = height // N_y
    tile_width = width // N_x

    patches = []
    x_offsets = []
    y_offsets = []

    for i in range(N_y):
        for j in range(N_x):
            # Calculate coordinates for each tile
            logger.info(f"Patch tile {i}, {j}")
            y_start, y_end = i * tile_height, (i + 1) * tile_height
            x_start, x_end = j * tile_width, (j + 1) * tile_width

            patch = image_np[y_start:y_end, x_start:x_end]
            # Does the patch contain more 50% black pixels?
            # Get the total number of values in the array (H * W * 3 for RGB)
            total_pixels = patch.shape[0] * patch.shape[1]
            zero_pixels = (patch == [0, 0, 0]).all(axis=-1)
            non_zero_pixels = np.count_nonzero((patch != [0, 0, 0]).all(axis=-1))

            # Count the number of zero pixels
            zero_pixel_count = np.count_nonzero(zero_pixels)
            if zero_pixel_count > (0.9 * total_pixels):
                logger.warning(f"Patch {i}, {j} contains more than 50% black pixels")
                continue
            # Calculate and store offsets
            x_offsets.append(x_start)
            y_offsets.append(y_start)

            # TODO use a temporary directory / extract the feature directly
            patch_name = f"temp_patch_{i}_{j}.jpg"
            Image.fromarray(patch).save(patch_name)

            patches.append(patch_name)

    return patches, x_offsets, y_offsets
