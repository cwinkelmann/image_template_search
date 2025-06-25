import tempfile
import unittest
from pathlib import Path
from time import sleep

import numpy as np

from conf.config_dataclass import CacheConfig
import pytest
from unittest.mock import patch

from image_template_search.image_patch_finder import ImagePatchFinderLG
from image_template_search.types.exceptions import DetailedNoMatchError
from image_template_search.util.util import visualise_polygons
from image_template_search.util.util import visualise_image


@pytest.fixture
def test_config():
    cfg = CacheConfig()
    cfg.visualise = False
    cfg.visualise_matching = False
    cfg.visualise_info = False
    cfg.show_visualisation = False
    return cfg

# Removed unused test_visu_config fixture

@pytest.fixture
def template_path():
    return Path(__file__).parent / "data/crop_0_512.jpg"

@pytest.fixture
def large_image_path():
    return Path(__file__).parent / "data/crop_0_1280.jpg"

VISUALISE_FLAG = False


def test_template_images_search_small_small(test_config, template_path, large_image_path):
    """
    Find similar images in a dataset
    :return:
    """

    with tempfile.TemporaryDirectory() as cache_dir:

        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

            from image_template_search.image_patch_finder import find_patch
            from image_template_search.util.util import visualise_image


            output_path = Path("./output")
            ipf = ImagePatchFinderLG(template_path=template_path,
                                     large_image_path=large_image_path)
            result = ipf.find_patch(similarity_threshold=0.1)
            ipf.project_image(output_path=output_path)

            ax_w = visualise_image(image=ipf.warped_image_B, show=VISUALISE_FLAG, title=f"test_template_images_search_small_small_Template Image",
                                   dpi=75)
            ax_w = visualise_image(image=ipf.warped_image_B, show=VISUALISE_FLAG, title=f"test_template_images_search_small_small_Matched Image",
                                   dpi=75)

            assert isinstance(ipf.warped_image_B, np.ndarray)
            assert (512, 512, 3) == ipf.warped_image_B.shape, "The crop patch should be 512x512x3"

def test_template_images_ipf(test_config, template_path, large_image_path):
    """
    Find similar images in a dataset
    :return:
    """

    with tempfile.TemporaryDirectory() as cache_dir:
        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:
            from image_template_search.util.util import visualise_image

            output_path = Path(cache_dir) / "output"
            template_path = Path(__file__).parent / "data/crop_0_640.jpg"
            large_image_path = Path(__file__).parent / "data/DJI_0018.JPG"

            ipf = ImagePatchFinderLG(template_path=template_path,
                                     large_image_path=large_image_path)
            result = ipf.find_patch(similarity_threshold=0.1)
            ipf.project_image(output_path=output_path)

            ax_w = visualise_image(image=ipf.template_image, show=VISUALISE_FLAG,
                                   title=f"test_template_images_ipf__Template Image",
                                   dpi=75)
            ax_w = visualise_image(image=ipf.warped_image_B, show=VISUALISE_FLAG,
                                   title=f"test_template_images_ipf__Matched warped Image",
                                   dpi=75)

            assert isinstance(ipf.warped_image_B, np.ndarray)
            assert (640, 640, 3) == ipf.warped_image_B.shape, "The crop patch should be 640x640x3"

def test_template_images_ipf_2(test_config, template_path, large_image_path):
    """
    Find similar images in a dataset
    :return:
    """

    with tempfile.TemporaryDirectory() as cache_dir:
        test_config.cache_path = Path(cache_dir)

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:
            from image_template_search.util.util import visualise_image

            output_path = Path(cache_dir) / "output"

            ipf = ImagePatchFinderLG(template_path=template_path,
                                     large_image_path=large_image_path)
            result = ipf.find_patch(similarity_threshold=0.1)
            ipf.project_image(output_path=output_path)

            # crop, footprint = find_patch(template_path, large_image_path, output_path=output_path)

            ax_w = visualise_image(image=ipf.template_image, show=VISUALISE_FLAG,
                                   title=f"test_template_images_ipf__Template Image",
                                   dpi=75)
            ax_w = visualise_image(image=ipf.warped_image_B, show=VISUALISE_FLAG,
                                   title=f"test_template_images_ipf__Matched warped Image",
                                   dpi=75)

            assert isinstance(ipf.warped_image_B, np.ndarray)
            assert (512, 512, 3) == ipf.warped_image_B.shape, "The crop patch should be 512x512x3"



def test_find_footprint(test_config):
    """
    find and display the footprint of a template image
    :return:
    """

    with tempfile.TemporaryDirectory() as cache_dir:

        test_config.cache_path = Path(cache_dir)
        output_path = Path(cache_dir) / "output"

        with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

            cfg2 = mock_get_config()
            mock_get_config.assert_called_once()
            assert cfg2.cache_path == Path(cache_dir)


            template_path_1280 = Path(__file__).parent / "data/images/FMO04/templates/template_source_DJI_0049.1280.jpg"

            # template_path = template_path_640
            template_path = template_path_1280
            large_image_path = Path(__file__).parent / "data/DJI_0058.JPG"

            ipf = ImagePatchFinderLG(template_path=template_path,
                                     large_image_path=large_image_path)

            ipf.find_patch(output_path=output_path, similarity_threshold=0.1)
            ipf.project_image(output_path=output_path)

            ax_w = visualise_image(image_path=large_image_path, show=False, title=f"Large Image: {large_image_path.name} and template footprint",
                                   dpi=75)

            visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_w, color="red", linewidth=2.5,
                                      show=VISUALISE_FLAG,

                                      )

            assert isinstance(ipf.warped_image_B, np.ndarray)  # add assertion here
            assert ((1280, 1280, 3) == ipf.warped_image_B.shape, "The crop patch should be 1280x1280x3")





def test_stack_crops(test_config):
    """
    create series of images
    :return:
    """
    with patch('conf.config_dataclass.get_config', return_value=test_config) as mock_get_config:

        template_path = Path(__file__).parent / "data/crop_0_1280.jpg"
        large_image_path_1 = Path(__file__).parent / "data/DJI_0018.JPG"
        large_image_path_2 = Path(__file__).parent / "data/DJI_0019.JPG"
        large_image_path_3 = Path(__file__).parent / "data/DJI_0227.JPG"  # NO MATCH IMAGE

        large_image_paths = [large_image_path_1, large_image_path_2, large_image_path_3]
        from image_template_search.image_patch_finder import find_patch_stacked


        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            output_path = Path(cache_dir) / "output"

            crops = find_patch_stacked(template_path,
                                       large_image_paths,
                                       output_path=output_path)

        assert 2 == len(crops), "It should find two matches"

        for crop in crops:
            # Yes the white balance is off in these images
            assert isinstance(crop, np.ndarray)
            visualise_image(image=crop,
                            show=VISUALISE_FLAG,
                            title=f"test_stack_crops__Matched Image",)



if __name__ == '__main__':
    unittest.main()
