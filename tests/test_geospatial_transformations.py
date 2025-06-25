import pytest
import shapely

from image_template_search.geospatial_transformations import convert_point_crs


def test_convert_point_crs():
    location_long_lat = shapely.Point(-87.0, -10.0)  # 87°W, 10°S

    result = convert_point_crs(location_long_lat,
                               target_crs="EPSG:32715",
                               source_crs="EPSG:4326")

    assert result.x == pytest.approx(1158712, abs=1)


def test_convert_point_crs_2():
    location_long_lat = shapely.Point(90.0, 0)

    # Use the correct UTM zone for 90°E longitude
    result = convert_point_crs(location_long_lat,
                               target_crs="EPSG:32645",  # UTM 45N
                               source_crs="EPSG:4326")

    # 90°E is at the eastern edge of UTM Zone 45N
    assert result.x == pytest.approx(833978, abs=100)
    assert result.y == pytest.approx(0, abs=100)
