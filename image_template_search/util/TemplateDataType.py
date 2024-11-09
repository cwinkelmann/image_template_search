from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from shapely.geometry.polygon import Polygon

from image_template_search.util.HastyAnnotationV2 import ImageLabel


@dataclass
class TemplateData:
    template_image_path: Path
    template_extent: Polygon
    center_obj_template: ImageLabel
    template_image: np.ndarray