import tempfile
import unittest
from pathlib import Path
from time import sleep

import numpy as np

from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
from image_template_search.util.util import visualise_image


class ImageSimilarityTestCase(unittest.TestCase):



    def test_template_images_search(self):
        """
        Find similar images in a dataset
        :return:
        """

        template_path = Path(f"./data/crop_0_512.jpg")
        large_image_path = Path(f'./data/DJI_0019.JPG')

        output_path = Path("./output")
        crop = find_patch(template_path, large_image_path, output_path=output_path)

        ax_w = visualise_image(image=crop, show=True, title=f"Template Image",
                               dpi=75)
        ax_w = visualise_image(image=crop, show=True, title=f"Matched Image",
                               dpi=75)

        self.assertTrue(isinstance(crop, np.ndarray))  # add assertion here
        self.assertEqual((512, 512, 3), crop.shape, "The crop patch should be 512x512x3")


    def test_template_images_search_2(self):
        """
        Find a small patch on different other image
        :return:
        """

        template_path_640 = Path(f"./data/template_source_DJI_0049.640.jpg")
        template_path_1800 = Path(f"./data/template_source_DJI_0049.1280.jpg")

        # template_path = template_path_1280
        template_path = template_path_640
        large_image_path = Path(f'./data/DJI_0058.JPG')

        # template_path = Path(f"./data/crop_0_512.jpg")
        # large_image_path = Path(f'./data/DJI_0019.JPG')

        cache_dir = Path("./cache") # temporary hack to speed up testing
        output_path = Path("./output")

        # This does not work at all when the patch is 640x640
        crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

        # # TODO This method isn't working very well maybe because of OpenCV version
        # crop = find_patch_tiled(template_path_640,
        #                         large_image_path,
        #                         tile_size_x=700,
        #                         tile_size_y=700,
        #                         output_path=output_path,
        #                         cache_path=cache_dir, MIN_MATCH_COUNT=20)

        ax_w = visualise_image(image_path=template_path, show=True, title=f"Template Image",
                               dpi=75)

        ax_w = visualise_image(image=crop, show=True, title=f"Matched Image",
                               dpi=75)

        sleep(1)

        self.assertTrue(isinstance(crop, np.ndarray))  # add assertion here
        self.assertEqual((512, 512, 3), crop.shape, "The crop patch should be 512x512x3")

    def test_template_images_search_no_match(self):
        """
        Find the crop_0 in an image that does not contain it, therefore nothing should be found.
        TODO this functionality is not working yet
        :return:
        """

        template_path = Path(f"./data/crop_0_1280.jpg")
        large_image_path = Path(f'./data/DJI_0227.JPG')

        output_path = Path("./output")
        crop = find_patch(template_path, large_image_path, output_path=output_path)

        self.assertFalse(crop, "When the patch is not present nothing should be returned")



    def test_tiled_template_images_search(self):
        """
        Find similar images in a dataset
        :return:
        """
        template_path = Path(f"./data/crop_0_1280.jpg")
        large_image_path = Path(f'./data/DJI_0019.JPG')

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            # cache_dir = Path("./cache") # temporary hack to speed up testing


            output_path = Path("./output")
            crop = find_patch_tiled(template_path,
                                    large_image_path,
                                    tile_size_x=1500,
                                    tile_size_y=1500,
                                    output_path=output_path,
                                    cache_path=cache_dir, MIN_MATCH_COUNT=200)

        self.assertTrue(isinstance(crop, np.ndarray))  # add assertion here
        self.assertEqual((1280, 1280, 3), crop.shape, "The crop patch should be 512x512x3")

    def test_tiled_template_images_search_2(self):
        """
        Find similar images in a dataset
        :return:
        """
        template_path = Path(f"./data/crop_0_1280.jpg")
        large_image_path = Path(f'./data/DJI_0018.JPG')

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir = Path("./cache_nc") # temporary hack to speed up testing


            output_path = Path("./output")
            crop = find_patch_tiled(template_path,
                                    large_image_path,
                                    tile_size_x=1500,
                                    tile_size_y=1500,
                                    output_path=output_path,
                                    cache_path=cache_dir, MIN_MATCH_COUNT=200)

        self.assertTrue(isinstance(crop, np.ndarray))  # add assertion here
        self.assertEqual((1280, 1280, 3), crop.shape, "The crop patch should be 512x512x3")


    def test_tiled_template_images_search_no_match(self):
        """
        Find the patch crop_0 in an image that does not contain it, therefore nothing should be found
        :return:
        """
        template_path = Path(f"./data/crop_0_1280.jpg")
        large_image_path = Path(f'./data/DJI_0227.JPG') # NO MATCH IMAGE

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir = Path("./cache") # temporary hack to speed up testing

            output_path = Path("./output")
            crop = find_patch_tiled(template_path,
                                    large_image_path,
                                    tile_size_x=1500,
                                    tile_size_y=1500,
                                    output_path=output_path,
                                    cache_path=cache_dir, MIN_MATCH_COUNT=80)

        self.assertFalse(crop, "patch is not present")

    def test_stack_crops(self):
        template_path = Path(f"./data/crop_0_1280.jpg")
        large_image_path_1 = Path(f'./data/DJI_0018.JPG')
        large_image_path_2 = Path(f'./data/DJI_0019.JPG')
        large_image_path_3 = Path(f'./data/DJI_0227.JPG')  # NO MATCH IMAGE

        large_image_paths = [large_image_path_1, large_image_path_2, large_image_path_3]

        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir = Path("./cache") # temporary hack to speed up testing


            output_path = Path("./output")
            crops = find_patch_stacked(template_path,
                                       large_image_paths,
                                       output_path=output_path,
                                       tile_path=cache_dir,
                                       cache_path=cache_dir,
                                        tile_size_x=1500,
                                       tile_size_y=1500,
                                       MIN_MATCH_COUNT=100)

        self.assertEqual(2, len(crops), "It should find two matches")







if __name__ == '__main__':
    unittest.main()
