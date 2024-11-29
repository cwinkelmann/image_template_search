from dataclasses import dataclass
from pathlib import Path

@dataclass
class CacheConfig:
    cache_path: str = Path("/Users/christian/PycharmProjects/hnee/image_template_search/similarity_cache")
    visualise: bool = True
    visualise_info: bool = False
    visualise_matching: bool = False
    show_visualisation: bool = False
    device: str = "cpu"
    caching: bool = True
    max_num_keypoints = 6000 # good number for a 20MP image
    patch_size = 640
    patch_size_offset = 400
    ransac_reproj_threshold = 2


def get_config()->CacheConfig:
    """
    Get the default configuration for the cache
    :return:
    """
    return CacheConfig()