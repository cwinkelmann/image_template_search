from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheConfig:
    cache_path: str = Path("/Users/christian/PycharmProjects/hnee/image_template_search/similarity_cache")
    visualise: bool = True
    visualise_info: bool = True
    visualise_matching: bool = False
    show_visualisation: bool = True
    device: str = "cpu"
    caching: bool = False
    max_num_keypoints = 6000
    patch_size = 640
    patch_size_offset = 400
    ransac_reproj_threshold = 2