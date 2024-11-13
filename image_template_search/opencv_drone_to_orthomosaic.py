import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**30) ## TODO this is a quickfix
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2211110758

from pathlib import Path

from image_template_search.image_similarity import find_patch_tiled

if __name__ == "__main__":





    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
    image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif" # DroneDeploy

    base_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    drone_image = base_path / "single_images/DJI_0066.JPG"
    # image_2 = base_path / "single_images/DJI_0067.JPG"
    image_2 = base_path / "mosaics/mosaic_100.jpg"

    # image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Pix4D.tiff" # pix4D
    # image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Agisoft.tif" # metashape
    interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

    tile_base_path = interm_path / "tiles"
    cache_path = interm_path / "cache"
    output_path = interm_path / "output"

    find_patch_tiled(template_path=drone_image,
                     large_image_path=image_2,
                         output_path=output_path,
                         tile_size_x=6000,
                         tile_size_y=4000,
                         tile_base_path=tile_base_path,
                         cache_path=cache_path,
                         MIN_MATCH_COUNT=50, visualise=True)