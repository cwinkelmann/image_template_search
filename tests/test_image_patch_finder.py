

import pytest
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from unittest.mock import patch, MagicMock

from image_template_search.image_patch_finder import ImagePatchFinderLG
from image_template_search.types.exceptions import NoMatchError


@pytest.fixture
def patch_finder():
    template_path = Path(__file__).parent / "data/crop_0_640.jpg"
    large_image_path = Path(__file__).parent / "data/DJI_0018.JPG"


    ipf = ImagePatchFinderLG(template_path=template_path,
                             large_image_path=large_image_path)


    return ipf


def test_init(patch_finder):
    assert patch_finder.template_path == Path(__file__).parent / "data/crop_0_640.jpg"
    assert patch_finder.large_image_path == Path(__file__).parent / "data/DJI_0018.JPG"

    assert isinstance(patch_finder.template_polygon, Polygon)
    assert patch_finder.M_ is None
    assert patch_finder.M is None
    assert patch_finder.mask is None

@patch('image_template_search.image_patch_finder.project_bounding_box')
@patch('image_template_search.image_patch_finder.find_rotation_gen_cv2')
#@patch('image_template_search.image_patch_finder.get_similarity')
def test_find_patch( mock_find_rotation, mock_project_bbox, patch_finder):
    # Set up mocks
    # mock_similarity.return_value = (0.9, np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]]))
    mock_find_rotation.return_value = (np.eye(3), np.ones((4, 1)), Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]))
    mock_project_bbox.return_value = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])

    result = patch_finder.find_patch(similarity_threshold=0.1)

    # Assertions
    # mock_similarity.assert_called_once()
    mock_find_rotation.assert_called_once()
    mock_project_bbox.assert_called_once()


    # Assert if the homography matrix was updated
    assert np.array_equal(patch_finder.M, np.eye(3))
    assert np.array_equal(patch_finder.mask, np.ones((4, 1)))
    assert isinstance(patch_finder.footprint, Polygon)
    assert patch_finder.theta == 0.0  # Since identity matrix, no rotation

# /Users/christian/PycharmProjects/hnee/image_template_search/image_template_search/image_patch_finder.py
@patch('image_template_search.image_patch_finder.get_similarity', return_value=(0.05, np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]])))
def test_find_patch_low_similarity(mock_similarity, patch_finder):
    with pytest.raises(NoMatchError):
        patch_finder.find_patch(similarity_threshold=0.1)


# TODO fix this later
# @patch('cv2.imwrite', return_value=True)
# @patch('cv2.warpPerspective', return_value=np.zeros((100, 100, 3)))
# @patch('cv2.imread', side_effect=[np.zeros((100, 100, 3)), np.zeros((500, 500, 3))])
# def test_project_image(mock_imread, mock_warp, mock_imwrite, patch_finder):
#     output_path = Path('./output')
#
#     # Call project_image and capture the returned path
#     result = patch_finder.project_image(output_path)
#
#     # Assert imread was called for both template and large image
#     assert mock_imread.call_count == 2
#     mock_warp.assert_called()
#     mock_imwrite.assert_called()
#
#     # Check if the returned path is correct
#     assert result == output_path / f"warped_source_{patch_finder.template_path.stem}_match_{patch_finder.large_image_path.stem}.jpg"
