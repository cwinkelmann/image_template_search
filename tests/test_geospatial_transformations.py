import pytest
import shapely

from image_template_search.geospatial_transformations import convert_point_crs


def test_convert_point_crs():
    location_long_lat = shapely.Point(90.0, 0)

    result = convert_point_crs(location_long_lat, target_crs="EPSG:32715", source_crs="EPSG:4326")

    assert result.x == pytest.approx(1010000, abs=1)