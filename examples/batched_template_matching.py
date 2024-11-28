import typing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from examples.review_annotations import debug_hasty_fiftyone
from image_template_search.types.workflow_config import WorkflowConfiguration, persist_file, load_yaml_config, \
    BatchWorkflowConfiguration, BatchWorkflowReportConfiguration, data_to_cls
from image_template_search.util.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2, AnnotatedImage
from workflow_iguana_deduplication import workflow_project_single_image_drone_and_annotations


def init_image_set(bwc: BatchWorkflowConfiguration) -> tuple[typing.List[AnnotatedImage], typing.List[Path], HastyAnnotationV2]:
    assert isinstance(bwc, BatchWorkflowConfiguration), "bwc should be of type BatchWorkflowConfiguration"
    assert len(bwc.workflow_configurations) > 0, "There should be at least one workflow configuration"

    hA_projection_images: typing.List[AnnotatedImage] = []
    projection_images_paths: typing.List[Path] = []

    for c in bwc.workflow_configurations:
        c = WorkflowConfiguration(**c)

        # if the drone image is not part of the annotations, we need to add it for visualisation purposes later
        hA_drone_image = hA_from_file(file_path=c.annotations_file_path)
        hA_drone_images = [i for i in hA_drone_image.images if i.image_name in [Path(c.drone_image_path).name]]
        drone_image_path = Path(c.drone_image_path)
        if not drone_image_path in projection_images_paths:
            hA_projection_images.extend(hA_drone_images)
            projection_images_paths.append(drone_image_path)

        hA_template = hA_drone_image
        hA_template.images = []

    return hA_projection_images, projection_images_paths, hA_template

if __name__ == "__main__":


    # bwc_file_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/batched_workflow_config_Snt_STJB06_12012023.yaml")
    bwc_file_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/examples/workflow_configs/batched_workflow_config_San_STJB01_10012023.yaml")
    # bwc_file_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/examples/workflow_configs/batched_workflow_config_FCD01_02_03.yaml")
    bwc = load_yaml_config(yaml_file_path=bwc_file_path, cls=BatchWorkflowConfiguration)
    hA_projection_images, projection_images, hA_template = init_image_set(bwc)


    bwrc = load_yaml_config(yaml_file_path=bwc_file_path, cls=BatchWorkflowReportConfiguration)

    for c in bwc.workflow_configurations:
        hA_projection, images_set, report = workflow_project_single_image_drone_and_annotations(data_to_cls(c, WorkflowConfiguration))

        hA_projection_images.extend(hA_projection.images)
        projection_images.extend(images_set)

        bwrc.workflow_report_configurations.append(report)

    hA_template.images = hA_projection_images
    file_path = bwc.base_path / f"combined_annotations_{bwc.dataset_name}.json"
    HastyAnnotationV2.save(hA_template, file_path=file_path)
    bwrc.combined_annotations_path = file_path

    persist_file(config=bwrc, file_path=bwc.base_path / f"batch_workflow_report_config_{bwc.dataset_name}.yaml")

    try:
        # create dot annotations
        dataset = debug_hasty_fiftyone(
            annotated_images=hA_projection_images,
            images_set=projection_images,
            dataset_name=bwc.dataset_name,
            type="points",
        )

        dataset.annotate(
            anno_key=bwc.dataset_name,
            # Using a organization requires a team account for 33USD
            # project_name = "Orthomosaic_quality_control",
            # organization=organization
            label_field="ground_truth_points",
            attributes=["iscrowd"],
            launch_editor=True,
        )
        print(dataset.get_annotation_info(bwc.dataset_name))
    except ValueError as e:
        logger.error(e)
        logger.error(f"Could not create annotation dataset {bwc.dataset_name} probably because the name is already taken, delete it first.")