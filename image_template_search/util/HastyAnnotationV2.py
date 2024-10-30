import typing

import uuid
from uuid import UUID
import random

import copy
import json
from pathlib import Path

import shapely
from loguru import logger
from pydantic import BaseModel, Field
from shapely import Polygon
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd



class LabelClass(BaseModel):
    class_id: str
    parent_class_id: Optional[str]
    class_name: str
    class_type: str
    color: str
    norder: float
    icon_url: Optional[str]
    attributes: List[str]



class Keypoint(BaseModel):
    x: int
    y: int
    id: UUID
    norder: int
    visible: bool
    created_by: UUID
    updated_by: Optional[UUID]
    create_date: datetime
    update_date: Optional[datetime]
    keypoint_class_id: UUID



class ImageLabel(BaseModel):
    id: typing.Union[str, int] = Field(default=str(uuid.uuid4()), alias='id')
    class_name: str = Field(alias='class_name')
    bbox: Optional[List[int]] = Field(None, alias='bbox')
    polygon: Optional[List[List[int]]] = Field(default=None) # A list of points that make up the polygon
    mask: Optional[List[int]] = Field(default=[])
    z_index: Optional[int] = 0
    attributes: dict = {}
    keypoints: Optional[List[Keypoint]] = None
    # incenter: Optional[List[int]] = None # The point which is either the centroid or the nearest point to the centroid that is withing the shape


    @property
    def x1y1x2y2(self):
        """
        deprecated, a bbox could be anything
        :return:
        """
        return self.bbox

    @property
    def bbox_polygon(self) -> Optional[shapely.Polygon]:
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            poly = shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            return poly
        else:
            return None

    @property
    def centroid(self) -> shapely.Point:
        """
        deprecated
        :return:
        """
        # TODO adobt this so it works with polygons too
        return self.bbox_polygon.centroid


    @property
    def polygon_s(self) -> shapely.Polygon:
        """ shapely represantation of the polygon """

        if self.polygon is not None:
            return Polygon(self.polygon)
        else:
            return None

    @bbox_polygon.setter
    def bbox_polygon(self, value):
        self._bbox_polygon = value
        self.bbox = [int(value.bounds[0]), int(value.bounds[1]), int(value.bounds[2]), int(value.bounds[3])]

    def __hash__(self):
        return self.id

class Image(BaseModel):
    image_id: typing.Union[str, int] = Field(default=str(uuid.uuid4()), alias='image_id')
    image_name: str = Field(alias='image_name', description="Name of the image file")
    dataset_name: Optional[str] = Field(default=None, alias='dataset_name')
    ds_image_name: Optional[str] = Field(default=None)
    width: int = Field()
    height: int = Field()
    image_status: Optional[str] = "Done"
    tags: Optional[List[str]] = []
    labels: List[ImageLabel]
    image_mode: Optional[str] = None



class HastyAnnotationV2(BaseModel):
    project_name: str = Field(alias='project_name')
    create_date: datetime = Field(default=datetime.now())
    export_format_version: str = Field(alias='export_format_version', default="1.1")
    export_date: datetime = Field(default=datetime.now())
    label_classes: List[LabelClass]
    images: List[Image]

    def save(self, file_path: Path):
        with open(file_path, 'w') as json_file:
            # Serialize the list of Pydantic objects to a list of dictionaries
            json_file.write(self.model_dump_json())


class HastyAnnotationV2_flat(BaseModel):
    project_name: str
    create_date: datetime
    export_format_version: str
    export_date: datetime
    label_classes: List[LabelClass]

    image_id: str | int = Field(default=uuid.uuid4(), alias='image_id')
    image_name: str
    dataset_name: str
    ds_image_name: Optional[str] = None
    width: int
    height: int
    image_status: Optional[str] = "New"
    tags: List[str]
    # labels: List[ImageLabel]
    image_mode: Optional[str] = None

    # ImageLabel
    label_id: str
    class_name: str
    bbox: Optional[List[int]] = Field(None)
    mask: Optional[List[int]] = []
    z_index: int
    ID: Optional[str] = None


    @property
    def x1y1x2y2(self):
        """
        deprecated, a bbox could be anything
        :return:
        """
        return self.bbox

    @property
    def bbox_polygon(self) -> shapely.Polygon:
        x1, y1, x2, y2 = self.bbox
        poly = shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return poly

    @property
    def centroid(self) -> shapely.Point:
        return self.bbox_polygon.centroid





def filter_by_class(hA: HastyAnnotationV2, class_names: Optional[str] = None) -> HastyAnnotationV2:
    """
    remove any labels that are not in the class_names list
    :param hA:
    :param class_names:
    :return:
    """
    assert type(class_names) is list or type(class_names) is tuple, "class_names must be a list or tuple"

    if len(class_names) > 0:
        # Create a deep copy of the project to avoid modifying the original object
        filtered_project = copy.deepcopy(hA)

        for image in filtered_project.images:
            # Filter labels by class_name
            filtered_labels = [label for label in image.labels if label.class_name in class_names]
            image.labels = filtered_labels
        return filtered_project
    else:
        return hA


def filter_by_image_tags(hA: HastyAnnotationV2, image_tags: Optional[List[str]] = None) -> HastyAnnotationV2:
    """
    remove any labels that are not in the image_tags list
    :param image_tags:
    :param hA:
    :param class_names:
    :return:
    """
    if image_tags is None or len(image_tags) == 0:
        return hA

    assert type(image_tags) is list or type(image_tags) is tuple, "image_tags must be a list or tuple"

    if len(image_tags) > 0:
        # Create a deep copy of the project to avoid modifying the original object
        filtered_project = copy.deepcopy(hA)

        for image in filtered_project.images:
            filtered_labels = [label for label in image.labels if len(list(set(image.tags) & set(image_tags))) > 0 ]
            image.labels = filtered_labels
        return filtered_project
    else:
        return hA


def convert_masks_to_bbox(hA: HastyAnnotationV2) -> HastyAnnotationV2:
    """
    convert every mask to a bounding box
    :param hA:
    :return:
    """

    # TODO it seems this already implemented in the HastyAnnotationV2 object

    return hA



def remove_images_with_no_labels(project: HastyAnnotationV2) -> HastyAnnotationV2:
    # Create a deep copy of the project to avoid modifying the original object
    filtered_project = copy.deepcopy(project)

    # Filter images to only include those with at least one label
    filtered_project.images = [image for image in filtered_project.images if len(image.labels) > 0]

    return filtered_project


def convert_HastyAnnotationV2_to_HastyAnnotationV2flat(project: HastyAnnotationV2) -> [HastyAnnotationV2_flat]:
    """

    :param project:
    :return:
    """
    flat_annotations = []

    for image in project.images:
        for label in image.labels:

            ID = label.attributes.get("ID", None)

            flat_annotations.append(
                HastyAnnotationV2_flat(
                    project_name=project.project_name,
                    create_date=project.create_date,
                    export_format_version=project.export_format_version,
                    export_date=project.export_date,
                    label_classes=project.label_classes,

                    image_id=image.image_id,
                    image_name=image.image_name,
                    dataset_name=image.dataset_name,
                    ds_image_name=image.ds_image_name,
                    width=image.width,
                    height=image.height,
                    image_status=image.image_status,
                    tags=image.tags,
                    image_mode=image.image_mode,

                    label_id=label.id,
                    class_name=label.class_name,
                    bbox=label.bbox,
                    mask=label.mask,
                    z_index=label.z_index,
                    ID=ID
                )
            )
    return flat_annotations

def get_flat_df(project: HastyAnnotationV2) -> pd.DataFrame:
    """
    Convert a HastyAnnotationV2 object to a flat DataFrame
    :param project:
    :return:
    """
    label_data = []

    for image in project.images:
        for label in image.labels:
            x1, y1, x2, y2 = label.x1y1x2y2
            poly = shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            s = label.attributes
            ID = label.attributes.get("ID", None)
            # [["image_name", "class_name", "ID", "centroid", "bbox", "bbox_polygon"]]
            label_data.append({
                "dataset_name": image.dataset_name,
                "image_name": image.image_name,
                "image_id": image.image_id,

                "class_name": label.class_name,
                "bbox_x1y1x2y2": label.x1y1x2y2,
                "bbox": label.x1y1x2y2,
                "bbox_polygon": poly,
                "centroid": poly.centroid,
                "ID": ID
            })

    labels_df = pd.DataFrame(label_data)

    return labels_df




def hA_from_file(file_path: Path) -> HastyAnnotationV2:
    """
    Load a HastyAnnotationV2 object from a file
    :param file_path:
    :return:
    """
    with open(file_path, mode="r") as f:
        data = json.load(f)
        hA = HastyAnnotationV2(**data)
    return hA



def label_dist_edge_threshold(patch_size, source_image):
    """
    remove labels which are too close to the border. Only in literal edge cases those are not covered anywhereelse
    :param patch_size:
    :param source_image:
    :return:
    """
    n_labels = len(source_image.labels)
    # bd_th = int((patch_size ** 2 // 2) ** 0.5)  # TODO THIS would be the right way to calculate the distance
    bd_th = int(patch_size // 2)
    source_image.labels = [l for l in source_image.labels if l.centroid.within(
        Polygon([(0 + bd_th, bd_th), (source_image.width - bd_th, bd_th),
                 (source_image.width - bd_th, source_image.height - bd_th), (bd_th, source_image.height - bd_th)]))]
    logger.info(
        f"After edge thresholding with distance {bd_th} in {len(source_image.labels)}, remove {n_labels - len(source_image.labels)} labels")

    return source_image.labels