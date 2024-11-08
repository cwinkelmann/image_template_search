__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

import argparse
import os
import typing
from pathlib import Path

import fiftyone as fo
import pandas as pd
import PIL.Image as Image

from typing import List

import shapely

from image_template_search.util.HastyAnnotationV2 import HastyAnnotationV2

parser = argparse.ArgumentParser(prog='view',
                                 description='View ground truth or detections (points) on images using FiftyOne app')

parser.add_argument('root', type=str,
                    help='path to the images directory (str)')
parser.add_argument('gt', type=str,
                    help='path to a csv file containing points ground truth (str)')
parser.add_argument('-dets', type=str,
                    help='path to a csv file containing a model\'s points detections (str). Defaults to None')

args = parser.parse_args()


def _create_dataset():
    return fo.Dataset.from_images_dir(args.root)


def _get_points_and_labels(img_path: str, df: pd.DataFrame) -> list:
    w, h = Image.open(img_path).size
    img_name = os.path.basename(img_path)
    records = df[df['images'] == img_name].to_dict('records')
    points = [(r['x'] / w, r['y'] / h) for r in records]
    labels = [r['labels'] for r in records]
    return points, labels


def _create_keypoints(points: list, labels: list) -> List[fo.Keypoint]:
    keypoints = []
    for pt, lab in zip(points, labels):
        kp = fo.Keypoint(label=str(lab), points=[pt])
        keypoints.append(kp)
    return keypoints

def _create_boxes(boxes: typing.List[shapely.Polygon], labels: typing.List[str]) -> typing.List[fo.Detection]:
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



def debug_hasty_fiftyone(hA: HastyAnnotationV2, image_path: Path):
    """
    Display these annotations in Fifty One
    :return:
    """
    gt = pd.read_csv(args.gt)
    dets = None
    if args.dets is not None:
        dets = pd.read_csv(args.dets)

    dataset = _create_dataset()
    for sample in dataset:

        gt_points, gt_labels = _get_points_and_labels(sample.filepath, df=gt)

        # ground truth
        sample['gt'] = fo.Keypoints(keypoints=_create_keypoints(gt_points, gt_labels))

        # detections
        if dets is not None:
            dets_points, dets_labels = _get_points_and_labels(sample.filepath, df=dets)
            sample['predictions'] = fo.Keypoints(keypoints=_create_keypoints(dets_points, dets_labels))

        sample.save()

    session = fo.launch_app(dataset, port=5151)
    session.wait()


if __name__ == '__main__':
    debug_hasty_fiftyone()