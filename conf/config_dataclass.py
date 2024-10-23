from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheConfig:
    cache_path: str = Path("/Users/christian/PycharmProjects/hnee/image_template_search/similarity_cache")
