"""
Upload annotations into FiftyOne for Evaluation OR modify annotations in CVAT

"""

import os
import typing
from pathlib import Path
from typing import List

import PIL.Image as Image
import fiftyone as fo
import pandas as pd
import shapely

from image_template_search.util.HastyAnnotationV2 import hA_from_file, AnnotatedImage


def _get_points_and_labels(img_path: str, df: pd.DataFrame) -> tuple[List[tuple[float, float]], List[str]]:
    w, h = Image.open(img_path).size
    img_name = os.path.basename(img_path)
    records = df[df['images'] == img_name].to_dict('records')

    points = [(r['x'] / w, r['y'] / h) for r in records]  # normalise the point coordinates
    labels = [r['labels'] for r in records]  # get the labels

    return points, labels


def _get_points_and_labels_hA(hA_image: AnnotatedImage) -> tuple[List[tuple[float, float]], List[str]]:
    w, h = hA_image.width, hA_image.height

    points = [(int(r.centroid.x) / w, int(r.centroid.y) / h) for r in
              hA_image.labels]  # normalise the point coordinates
    labels = [r.class_name for r in hA_image.labels]  # get the labels

    return points, labels


def _create_keypoints(points: list[tuple[int, int]], labels: list[str]) -> List[fo.Keypoint]:
    keypoints = []
    for pt, lab in zip(points, labels):
        kp = fo.Keypoint(id="basdas",
                         kind="str",
                         label=str(lab),
                         points=[pt]
                         )
        kp.id
        keypoints.append(kp)
    return keypoints


def _create_keypoints_s(hA_image: AnnotatedImage) -> List[fo.Keypoint]:
    """

    :param hA_image:
    :return:
    """
    keypoints = []
    w, h = hA_image.width, hA_image.height

    for r in hA_image.labels:


        pt = (int(r.incenter_centroid.x) / w, int(r.incenter_centroid.y) / h)
        lab = r.class_name

        kp = fo.Keypoint(
            # kind="str",
            hasty_id=r.id,
            label=str(lab),
            points=[pt],
            # attributes=r.attributes,
            # attributes={"custom_attribute": {"bla": "keks"}},
            # tags=["bla", "keks"]
        )

        keypoints.append(kp)
    return keypoints


def _create_boxes(boxes: typing.List[shapely.Polygon],
                  labels: typing.List[str]) -> typing.List[fo.Detection]:
    """

    :param boxes:
    :param labels:
    :return:
    """
    detections = []
    for b, lab in zip(boxes, labels):
        det = fo.Detection(label=str(lab), bounding_box=[b.bounds])
        detections.append(det)
    return detections

def _create_boxes_s(hA_image: AnnotatedImage) -> typing.List[fo.Detection]:
    """

    :param hA_image:
    :return:
    """
    boxes = []
    w, h = hA_image.width, hA_image.height

    for r in hA_image.labels:
        pt = (int(r.centroid.x) / w, int(r.centroid.y) / h)

        x1, y1, x2, y2 = r.x1y1x2y2[0], r.x1y1x2y2[1], r.x1y1x2y2[2], r.x1y1x2y2[3]
        box_w, box_h = x2 - x1, y2 - y1
        box_w /= w
        box_h /= h

        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
        lab = r.class_name


        kp = fo.Detection(label=str(lab), bounding_box=[x1, y1, box_w, box_h],
            # kind="str",
            hasty_id=r.id,
            # attributes=r.attributes,
            attributes={"custom_attribute": {"bla": "keks"}},
            tags=["bla", "keks"]
        )

        boxes.append(kp)
    return boxes


def debug_hasty_fiftyone(
        images_set = List[Path],
        # images_dir: Path,
        hA_gt: List[AnnotatedImage] = None,
        hA_dets: List[AnnotatedImage] = None,
        dataset_name="projection_comparison",
        type="points"):
    """
    Display these annotations in Fifty One
    :return:
    """
    # dataset = fo.Dataset.from_images_dir(images_dir=images_dir)
    # TODO ensure detections and gt are the same set

    # images_set: list[Path] = [images_dir / i.image_name for i in hA_gt]
    # Specify a name for the dataset


    # Create an empty dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    # fo.list_datasets()

    # dataset = fo.load_dataset(dataset_name) # loading a bad idea becase the single source of truth is the hasty annotations
    samples = []

    for image_path in images_set:


        hA_gt_sample = [i for i in hA_gt if i.image_name == image_path.name]
        assert len(hA_gt_sample) == 1, "There should be one single image left"

        hA_image = hA_gt_sample[0]
        keypoints = _create_keypoints_s(hA_image=hA_image)
        boxes = _create_boxes_s(hA_image=hA_image)

        sample = fo.Sample(filepath=image_path,
                           tags=hA_image.tags,
                           hasty_image_id=hA_image.image_id,
                           hasty_image_name=hA_image.image_name)


        if type == "points":
            sample['ground_truth_points'] = fo.Keypoints(keypoints=keypoints)
        elif type == "boxes":
            sample['ground_truth_boxes'] = fo.Detections(detections=boxes)
        else:
            raise ValueError("Unknown type, use 'boxes' or 'points'")

        # sample.save()
        samples.append(sample)

    dataset.add_samples(samples)

    return dataset


if __name__ == '__main__':
    ## The image for Andrea:
    base_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    images_dir = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068"
    drone_image = images_dir / "San_STJB01_10012023_DJI_0068.JPG"
    image_2 = base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif"

    annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels.json"
    # annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels_points.json"
    # load hasty annotations
    hA = hA_from_file(file_path=annotations_file_path)
    hA_images = [i for i in hA.images if i.image_name in [drone_image.name]]

    projected_annotation_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data/output/template_annotations_projected.json")
    projected_image = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data/output/matched_template_DJI_0366_Metashape_FCD01-02-03-orthomosaic_cropped.jpg")
    hA_proj = hA_from_file(file_path=projected_annotation_path)
    hA_proj_images = [i for i in hA_proj.images if i.image_name in [projected_image.name]]

    hA.images = hA_images + hA_proj_images

    assert len(hA.images) == 2, "There should be only a single image left"

    images_set = [projected_image, drone_image]

    dataset_name = "projection_comparison"
    try:
        fo.delete_dataset(dataset_name)
    except:
        pass

    # create dot annotations
    dataset = debug_hasty_fiftyone(hA_dets=hA.images,
                                   hA_gt=hA.images,
                                   images_set=images_set,
                                   dataset_name=dataset_name,
                                   type="points")

    ## Launch FiftyOne inspection app
    session = fo.launch_app(dataset, port=5151)
    session.wait()

    # sample_id = dataset.first().id
    # view = dataset.select(sample_id)

    # Step 3: Send samples to CVAT

    # A unique identifier for this run
    anno_key = "cvat_basic_recipe"

    dataset.annotate(
        anno_key,
        label_field="ground_truth_boxes",
        attributes=["iscrowd"],
        launch_editor=True,
    )
    print(dataset.get_annotation_info(anno_key))

    # Step 4: Perform annotation in CVAT and save the tasks
