"""
Use open CV tile based image maching to find the drone image in the orthomosaic
"""


import os

from image_template_search.image_patch_finder import ImagePatchFinderCV
from image_template_search.util.util import visualise_image, visualise_polygons

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**30) ## TODO this is a quickfix
from PIL import Image
Image.MAX_IMAGE_PIXELS = 5223651122

from pathlib import Path


if __name__ == "__main__":

    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"

    # image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif" # demo
    # image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif" # DroneDeploy
    # image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Pix4D.tiff" # pix4D
    orthomosaic_path = base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Agisoft.tif" # metashape
    interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

    drone_image = base_path / "FCD01-02-03/template_images/Fer_FCD01-02-03_20122021_single_images/DJI_0366.JPG"
    orthomosaic_path = base_path / "FCD01-02-03/Metashape_FCD01-02-03-orthomosaic.tif" # metashape


    # base_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    # drone_image = base_path / "single_images/DJI_0066.JPG"
    # image_2 = base_path / "mosaics/mosaic_100.jpg"

    tile_base_path = interm_path / "tiles"
    cache_path = interm_path / "cache"
    output_path = interm_path / "output"

    ipf = ImagePatchFinderCV(template_path=drone_image,
                             large_image_path=orthomosaic_path)

    ipf.find_patch()
    ax_i = visualise_image(image_path=ipf.large_image_path, show=False, dpi=50)
    visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_i, show=True, color="red", linewidth=4)

    # TODO crop the mosaic with the projected polygon because warping is not possible

    # TODO crop the template extent itself matched_template_San_STJB01_10012023_DJI_0068_San_STJB01_10012023_orthomosaic_DDeploy.jpg

    # TODO The annotations could be projected within the crop


    # find_patch_tiled(template_path=drone_image,
    #                  large_image_path=image_2,
    #                      output_path=output_path,
    #                      tile_size_x=6000,
    #                      tile_size_y=4000,
    #                      tile_base_path=tile_base_path,
    #                      cache_path=cache_path,
    #                      MIN_MATCH_COUNT=50, visualise=True)