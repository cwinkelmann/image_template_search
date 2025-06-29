"""
step 3: find modified annotations from cvat, download them, delete them create a local annotation file


"""

import os
from pathlib import Path

import fiftyone as fo
import pandas as pd
import shapely
from loguru import logger

from image_template_search.types.workflow_config import (
    load_yaml_config,
    BatchWorkflowReportConfiguration,
    persist_file,
)
from image_template_search.util.HastyAnnotationV2 import (
    hA_from_file,
    ImageLabel,
    Keypoint,
    AnnotatedImage,
)
from image_template_search.util.util import visualise_image, visualise_polygons

if __name__ == "__main__":
    if (
        os.getenv("FIFTYONE_CVAT_PASSWORD") is None
        or os.getenv("FIFTYONE_CVAT_USERNAME") is None
    ):
        raise ValueError(
            "FIFTYONE_CVAT_PASSWORD and FIFTYONE_CVAT_USERNAME env variables must be set"
        )

    # config_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/FMO04/batch_workflow_report_config_FMO04.yaml")
    config_path = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/FMO04/batch_workflow_report_config_FMO04_short.yaml"
    )
    # config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/San_STJB01_10012023/batch_workflow_report_config_San_STJB01_10012023.yaml")
    # config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/Snt_STJB06_12012023/batch_workflow_report_config_Snt_STJB06_12012023.yaml")
    # config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/FCD01_02_03/batch_workflow_report_config_FCD01_02_03.yaml")

    batched_workflow_report = load_yaml_config(
        yaml_file_path=Path(config_path),
        cls=BatchWorkflowReportConfiguration,
    )
    assert isinstance(
        batched_workflow_report, BatchWorkflowReportConfiguration
    ), "The loaded config should be a BatchWorkflowReportConfiguration"

    base_path = batched_workflow_report.base_path

    # load hasty annotations
    hA = hA_from_file(file_path=batched_workflow_report.combined_annotations_path)

    hA_corrected = hA_from_file(
        file_path=batched_workflow_report.combined_annotations_path
    )
    hA_corrected.images = []
    corrected_annotations_file_path = (
        batched_workflow_report.base_path
        / f"corrected_annotations_{batched_workflow_report.dataset_name}.json"
    )

    dataset_name = batched_workflow_report.dataset_name
    anno_key = batched_workflow_report.anno_key

    cleanup = False

    # create dot annotations

    # Step 5: Merge annotations back into FiftyOne dataset

    dataset = fo.load_dataset(dataset_name)
    dataset.load_annotations(anno_key)

    # Load the view that was annotated in the App
    view = dataset.load_annotation_view(anno_key)

    # session = fo.launch_app(view=view)
    # session.wait()

    stats = []

    try:
        keypoint_class_id = (
            hA.keypoint_schemas[0].keypoint_classes[0].keypoint_class_id
        )  # a bit of a hack because there is only one keypoint schema here, but there could be more
    except:
        keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"

    # reconstruct an annotation file
    for sample in view:
        filepath = sample.filepath
        hasty_image_id = sample.hasty_image_id
        hasty_filename = sample.filename
        logger.info(f"Processing {hasty_filename}")
        image = [i for i in hA.images if i.image_id == sample.hasty_image_id][0]
        assert isinstance(image, AnnotatedImage)

        stats_row = {}
        updated_labels = []
        new_labels = []
        unchanged_labels = []

        if hasattr(sample, "ground_truth_boxes"):
            for kp in sample.ground_truth_boxes.detections:
                # print(kp)
                x1, y1, w, h = kp.bounding_box
                x2 = x1 + w
                y2 = y1 + h
                x1 *= image.width
                x2 *= image.width
                y1 *= image.height
                y2 *= image.height

                bbox = [int(x1), int(y1), int(x2), int(y2)]
                pt = shapely.box(*bbox).centroid

                if hasattr(kp, "hasty_id"):
                    # The object is a known object from before
                    # logger.info(f"Object {kp.hasty_id} was known before")

                    image_label = [l for l in image.labels if l.id == kp.hasty_id][0]

                    dist = pt.distance(image_label.centroid)
                    if dist > 2:
                        # The object was moved
                        # logger.info(f"Object {kp.hasty_id} was moved")

                        image_label.bbox = bbox
                        updated_labels.append(image_label)

                    else:
                        # The object was not moved
                        # logger.info(f"Object {kp.hasty_id} was not moved")

                        new_labels.append(image_label)

                else:

                    # The object is new
                    logger.info("New object")
                    il = ImageLabel(
                        class_name="iguana",
                        bbox=bbox,
                        polygon=None,
                        mask=None,
                        z_index=0,
                        keypoints=[],
                    )
                    new_labels.append(il)

        else:
            logger.info(f"Sample {sample.id} has no ground_truth_boxes")

        if hasattr(sample, "ground_truth_points"):
            for kp in sample.ground_truth_points.keypoints:
                # iterate over the keypoints
                pt = shapely.Point(
                    kp.points[0][0] * image.width, kp.points[0][1] * image.height
                )

                hkp = Keypoint(
                    x=int(pt.x),
                    y=int(pt.y),
                    norder=0,
                    keypoint_class_id=keypoint_class_id,
                )
                il = ImageLabel(
                    class_name=kp.label,
                    keypoints=[hkp],
                )

                if hasattr(kp, "hasty_id"):
                    # The object is a known object from before
                    logger.info(f"Object {kp.hasty_id} was known before")

                    image_label = [l for l in image.labels if l.id == kp.hasty_id][0]
                    assert isinstance(image_label, ImageLabel)
                    dist = pt.distance(image_label.incenter_centroid)

                    il.id = kp.hasty_id

                    if dist > 10:
                        # The object was moved
                        # logger.info(f"Object {kp.hasty_id} was moved")
                        il.attributes = (
                            il.attributes | image_label.attributes | {"cvat": "moved"}
                        )
                        updated_labels.append(il)
                    else:
                        # The object was not moved
                        # logger.info(f"Object {kp.hasty_id} was not moved")
                        il.attributes = (
                            il.attributes
                            | image_label.attributes
                            | {"cvat": "unchanged"}
                        )
                        unchanged_labels.append(il)

                else:
                    # The object is new
                    logger.info("New object")
                    il.attributes = {"cvat": "new"}
                    new_labels.append(il)
        else:
            logger.info(f"Sample {sample.id} has no ground_truth_points")
        stats_row["filename"] = hasty_filename

        updated_labels_ig = [il for il in updated_labels if il.class_name == "iguana"]
        new_labels_ig = [il for il in new_labels if il.class_name == "iguana"]
        unchanged_labels_ig = [
            il for il in unchanged_labels if il.class_name == "iguana"
        ]

        stats_row["updated_labels"] = len(updated_labels_ig)
        stats_row["new_labels"] = len(new_labels_ig)
        stats_row["unchanged_labels"] = len(unchanged_labels_ig)
        stats_row["after_correction"] = (
            len(updated_labels_ig) + len(new_labels_ig) + len(unchanged_labels_ig)
        )
        stats_row["before_correction"] = len(
            [il for il in image.labels if il.class_name == "iguana"]
        )

        stats.append(stats_row)
        image.labels = updated_labels + new_labels + unchanged_labels

        hA_corrected.images.append(image)

        ax_c = visualise_image(
            image_path=filepath, dpi=100, title=f"Corrected labels {hasty_filename}"
        )
        ax_c = visualise_polygons(
            points=[x.centroid for x in image.labels],
            ax=ax_c,
            filename=batched_workflow_report.base_path
            / f"{Path(hasty_filename).stem}_corrected.jpg",
            show=True,
            linewidth=3.5,
            markersize=5.5,
            color="red",
        )

    with open(corrected_annotations_file_path, "w") as json_file:
        json_file.write(hA_corrected.model_dump_json())
        batched_workflow_report.corrected_annotations_file_path = (
            corrected_annotations_file_path
        )

    stats_df = pd.DataFrame(stats)
    stats_path = batched_workflow_report.base_path / "stats.csv"
    stats_df.to_csv(stats_path, index=False)

    print(f"stats of the images set: {stats_df}")
    logger.info(f"Stats written to {stats_path}")
    batched_workflow_report.stats_path = stats_path

    persist_file(
        file_path=Path(f"{config_path.stem}_corrected.yaml"),
        config=batched_workflow_report,
    )

    logger.info(f"Wrote report to: {batched_workflow_report}")

    # if cleanup:
    #     # Step 6: Cleanup
    #
    #     ## Delete tasks from CVAT
    #     results = view.load_annotation_results(anno_key)
    #     results.cleanup()
    #
    #     ##  Delete run record (not the labels) from FiftyOne
    #     dataset.delete_annotation_run(anno_key)
    #     fo.delete_dataset(dataset_name)

    ## HOW to reconstruct the label cvat dataset
    # labels = [
    #     ImageLabel(
    #         category=label.label,
    #         bounding_box=label.bounding_box
    #     ) .detections  # Assuming `ground_truth` field
    # ]
    # session = fo.launch_app(dataset, port=5151)

    # logger.info("Deleting dataset")
    # fo.delete_dataset(dataset_name)
