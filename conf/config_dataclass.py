from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheConfig:
    cache_path: str = Path("/Users/christian/PycharmProjects/hnee/image_template_search/similarity_cache")
    visualise: bool = True
    visualise_info: bool = False
    show_visualisation: bool = True
    device: str = "cpu"
    caching: bool = True
    max_num_keypoints = 8000
    patch_size = 640
    patch_size_offset = 400