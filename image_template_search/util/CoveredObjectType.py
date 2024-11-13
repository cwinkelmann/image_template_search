from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any

import shapely

from image_template_search.util.HastyAnnotationV2 import AnnotatedImage, ImageLabel


@dataclass
class CoveredObject:
    """
    A dataclass to store the information about the covered object
    """
    template_id: str =  field(metadata={"description": "Unique identifier for the template"})
    template_image_path: Path = field(metadata={"description": "Path to the template image file"})
    source_image_name: str = field(metadata={"description": "Name of the source image"})
    source_image: AnnotatedImage = field(metadata={"description": "The source image object"})
    other_images: List[AnnotatedImage] = field(metadata={"description": "List of other images"})
    covered_templates: List[AnnotatedImage] = field(metadata={"description": "List of templates covered in the source image"})
    new_objects: List[ImageLabel] = field(metadata={"description": "List of new objects found in the source image"})
    template_extents: shapely.Polygon = field(metadata={"description": "The extent of the template in the source image"})


