import pytest
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from unittest.mock import patch, MagicMock

from detection_deduplication import find_annotated_template_matches
from image_template_search.util.HastyAnnotationV2 import hA_from_file

@pytest.fixture
def hA():
    return hA_from_file(
        file_path=Path(__file__).parent / "data/annotations/annotations_FMO04_DJI_0049.JPG.json")

@pytest.fixture
def images_path():
    return Path(__file__ ).parent / "data/images/FMO04"

@pytest.fixture
def output_path():
    return Path(__file__ ).parent / "output/cutouts/"

def test_find_annotated_template_matches(hA, images_path, output_path):
    # hA = hA_from_file(
    #     file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_28.json"))
    # images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    # output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")
    #

    source_image = hA.images[0]  # take the first image as the template
    other_images = hA.images[1:]  # take the next two images as the other images we are looking for annotations in

    patch_size = 1280

    ## Re-Identify the objects form the source image in the other images
    covered_objects = find_annotated_template_matches(
        images_path,
        source_image,
        other_images,
        output_path,
        patch_size=patch_size)

    assert len(covered_objects) == 2
    assert covered_objects[0].keys() == {'template_id', 'template_image', 'source_image_name', 'source_image',
                                         'other_images', 'covered_objects', 'new_objects', 'template_extents'}

