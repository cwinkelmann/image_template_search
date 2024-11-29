import tempfile
import unittest
from pathlib import Path
from time import sleep

import numpy as np

from conf.config_dataclass import CacheConfig
import pytest

@pytest.fixture
def test_config():
    cfg = CacheConfig()
    cfg.visualise_matching = True
    return cfg




from unittest.mock import patch

def test_template_images_search_small_small(test_config):
    """
    Find similar images in a dataset
    :return:
    """

    template_path = Path(f"./data/crop_0_512.jpg")
    large_image_path = Path(f"./data/crop_0_1280.jpg")

    with tempfile.TemporaryDirectory() as cache_dir:

        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

            from image_template_search.image_patch_finder import find_patch
            from image_template_search.util.util import visualise_image


            output_path = Path("./output")
            crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

            ax_w = visualise_image(image=crop, show=True, title=f"Template Image",
                                   dpi=75)
            ax_w = visualise_image(image=crop, show=True, title=f"Matched Image",
                                   dpi=75)

            assert isinstance(crop, np.ndarray)
            assert (512, 512, 3) == crop.shape, "The crop patch should be 512x512x3"

def test_template_images_search(test_config):
    """
    Find similar images in a dataset
    :return:
    """

    template_path = Path(f"./data/crop_0_512.jpg")
    large_image_path = Path(f'./data/DJI_0019.JPG')

    with tempfile.TemporaryDirectory() as cache_dir:

        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

            from image_template_search.image_patch_finder import find_patch
            from image_template_search.util.util import visualise_image


            output_path = Path("./output")
            crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

            ax_w = visualise_image(image=crop, show=True, title=f"Template Image",
                                   dpi=75)
            ax_w = visualise_image(image=crop, show=True, title=f"Matched Image",
                                   dpi=75)

            assert isinstance(crop, np.ndarray)
            assert (512, 512, 3) == crop.shape, "The crop patch should be 512x512x3"

def test_template_images_search_2(test_config):
    """
    Find a small patch on different other image
    :return:
    """

    with tempfile.TemporaryDirectory() as cache_dir:

        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

            cfg2 = mock_get_config()
            mock_get_config.assert_called_once()
            assert cfg2.cache_path == Path(cache_dir)

            from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
            from image_template_search.util.util import visualise_image

            template_path_640 = Path(f"./data/images/FMO04/templates/template_source_DJI_0049.640.jpg")
            template_path_1280 = Path(f"./data/images/FMO04/templates/template_source_DJI_0049.1280.jpg")

            # template_path = template_path_640
            template_path = template_path_1280
            large_image_path = Path(f'./data/DJI_0058.JPG')

            output_path = Path("./output")

            # This does not work at all when the patch is 640x640
            crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

            ax_w = visualise_image(image_path=template_path, show=True, title=f"Template Image",
                                   dpi=75)

            ax_w = visualise_image(image=crop, show=True, title=f"Matched Image",
                                   dpi=75)


            assert isinstance(crop, np.ndarray)  # add assertion here
            assert ((1280, 1280, 3) == crop.shape, "The crop patch should be 1280x1280x3")

def test_template_images_search_no_match(test_config):
    """
    Find a small patch on different other image
    :return:
    """
    template_path = Path(f"./data/crop_0_1280.jpg")
    large_image_path = Path(f'./data/DJI_0227.JPG')

    with tempfile.TemporaryDirectory() as cache_dir:
        output_path = cache_dir
        test_config.cache_path = Path(cache_dir)


        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:
            from image_template_search.image_patch_finder import find_patch

            crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

            assert False == crop, "When the patch is not present nothing should be returned"

class ImageSimilarityTestCase(unittest.TestCase):











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

            from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
            from image_template_search.util.util import visualise_image
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
        from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
        from image_template_search.util.util import visualise_image
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
        from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
        from image_template_search.util.util import visualise_image
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
        from image_template_search.image_patch_finder import find_patch, find_patch_stacked, find_patch_tiled
        from image_template_search.util.util import visualise_image
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
