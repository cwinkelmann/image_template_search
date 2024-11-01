import tempfile

import pytest
import numpy as np
from pathlib import Path

import shapely
from shapely.geometry import Polygon


from conf.config_dataclass import CacheConfig
from detection_deduplication import find_annotated_template_matches, cutout_detection_deduplication
from image_template_search.util.HastyAnnotationV2 import hA_from_file, ImageLabel, HastyAnnotationV2
from image_template_search.util.util import visualise_image


@pytest.fixture
def hA():
    return hA_from_file(
        file_path=Path(__file__).parent / "data/annotations/annotations_FMO04_DJI_0049.JPG.json")

@pytest.fixture
def template_image_path():
    return Path(__file__).parent / "data/images/FMO04/templates/template_source_DJI_0049.1280.jpg"

@pytest.fixture
def source_image_path():
    return Path(__file__).parent / "data/images/FMO04/single_images/DJI_0049.JPG"

@pytest.fixture
def images_path():
    return Path(__file__ ).parent / "data/images/FMO04"

@pytest.fixture
def output_path():
    return Path(__file__ ).parent / "output/cutouts/"

def test_find_annotated_template_matches(hA: HastyAnnotationV2,
                                         images_path: Path
                                         ):
    """

    :param hA:
    :param images_path:
    :return:
    """

    source_image = hA.images[0]  # take the first image as the template
    other_images = hA.images[1:]  # take the next two images as the other images we are looking for annotations in

    assert len(other_images) == 2, "There are two other images"

    patch_size = 1280

    with tempfile.TemporaryDirectory() as output_path:
        output_path = Path(output_path)

        ## Re-Identify the objects form the source image in the other images
        image_stacks = find_annotated_template_matches(
            images_path,
            source_image,
            other_images,
            output_path,
            patch_size=patch_size)

        assert len(image_stacks) == 2, "There are two distinct templates with one object each"


        # How many objects are covered in the two images?
        image_stacks

        assert len(image_stacks[0].covered_templates) == 3, "Template image + 2 other images"
        assert len(image_stacks[0].covered_templates[0].labels) == 1, "One object in the template 0 region"
        assert len(image_stacks[0].covered_templates[1].labels) == 1, "One object in the template 0 region"
        assert len(image_stacks[0].covered_templates[2].labels) == 1, "One object in the source image template 0 region"

        # TODO visualise the three images including the annotations
        # visualise_image(template_image_path, show=True,
        #                 title="Template from the Source image")
        # visualise_image(output_path / 'warped_source_template_source_DJI_0049.1280_match_DJI_0063.jpg', show=True,
        #                 title="Warped Template from the other image")


def test_find_annotated_template_matches_only(hA: HastyAnnotationV2,
                                         images_path: Path
                                         ):

    source_image = hA.images[0]  # take the first image as the template
    other_images = []  # take 0 other images, we should get the objects of the first iamges
    patch_size = 1280

    assert len(other_images) == 0, "There are No other images"


    with tempfile.TemporaryDirectory() as output_path:
        output_path = Path(output_path)

        ## Re-Identify the objects form the source image in the other images
        image_stacks = find_annotated_template_matches(
            images_path,
            source_image,
            other_images,
            output_path,
            patch_size=patch_size)

        assert len(image_stacks) == 2, "There are two distinct templates with one object each"
        assert len(image_stacks[0].covered_templates) == 1, "Template image only"
        assert len(image_stacks[0].covered_templates[0].labels) == 1, "One object in the template 0 region"
        assert len(image_stacks[1].covered_templates[0].labels) == 1, "One object in the template 0 region"



def test_cutout_detection_deduplication(hA, source_image_path, template_image_path,
                                        images_path, output_path):
    """

    :param hA:
    :param source_image_path:
    :param template_image_path:
    :param images_path:
    :param output_path:
    :return:
    """
    other_images = hA.images[1:]  # take the next two images as the other images we are looking for annotations in

    template_labels = [ImageLabel(id='4d31e5ce-8f68-4fa8-b05a-d3a1a5b22ed0', class_name='iguana',
                                 bbox=[606, 581, 673, 699], polygon=None,
                                 mask=None, z_index=0, attributes={'ID': '8', 'distance_to_nearest_edge': 847.0},
                                 keypoints=[])
                       ]
    points = [(3821.5, 207), (3821.5, 1487), (2541.5, 1487), (2541.5, 207), (3821.5, 207)]

    # Create the rectangular polygon
    template_extent = Polygon(points)

    with tempfile.TemporaryDirectory() as output_path:
        output_path = Path(output_path)

        images_list = cutout_detection_deduplication(
            source_image_path=source_image_path,
            template_image_path=template_image_path, # path to the template image which is part of the source image
            cutout_polygon=template_extent,  # ith template extent
            template_labels=template_labels,  # ith set of objects

            other_images=other_images,  # all the other images
            images_path=images_path,
            output_path=output_path)

        assert len(images_list) == 2 # source image + other images
        # If this code is correct the returned annotations are supposed to be the same as the annotations in the source image
        assert len(template_labels) == len(images_list[0].labels)

        generated_files = {f.name for f in output_path.glob("*")}
        assert len(generated_files) == 8

        assert generated_files == {'annotations_large_DJI_0052.JPG_template_source_DJI_0049.1280.jpg',
             'annotations_large_DJI_0063.JPG_template_source_DJI_0049.1280.jpg',
             'cropped_DJI_0052.JPG_template_source_DJI_0049.1280_1_objects.jpg',
             'cropped_DJI_0063.JPG_template_source_DJI_0049.1280_1_objects.jpg',
             'warped_source_DJI_0049_match_DJI_0052.jpg',
             'warped_source_DJI_0049_match_DJI_0063.jpg',
             'warped_source_template_source_DJI_0049.1280_match_DJI_0052.jpg',
             'warped_source_template_source_DJI_0049.1280_match_DJI_0063.jpg'
                                   }

        if CacheConfig.visualise_info:
            visualise_image(template_image_path, show=True,
                            title="Template from the Source image")
            visualise_image(output_path / 'warped_source_template_source_DJI_0049.1280_match_DJI_0063.jpg', show=True,
                            title="Warped Template from the other image")


