"""
Step 1 - Load the configuration

"""

from datetime import datetime
from pathlib import Path

from loguru import logger

from image_template_search.types.workflow_config import (
    WorkflowConfiguration,
    persist_file,
    BatchWorkflowConfiguration,
)
from image_template_search.util.HastyAnnotationV2 import (
    hA_from_file,
)


def get_config(scenario: str) -> BatchWorkflowConfiguration:
    if scenario == "FMO04_tracking":
        dataset_name = "FMO04"

        base_path = Path(
            f"/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/{dataset_name}"
        )

        bwc = BatchWorkflowConfiguration(base_path=base_path, dataset_name=dataset_name)

        image_url = "https://app.hasty.ai/projects/9d8deceb-77be-47aa-978f-abcbdc8e00f2/image/4db0c13f-692b-4ae5-b4bc-591d314d87e6?datasetId=a37b5089-ece1-42c2-bf4d-db92d2dd3e6c"
        drone_image_path = base_path / "images_2024_10_07/single_images/DJI_0066.JPG"

        orthomosaic_paths = [
            base_path
            / "DD_FMO04_Orthomosaic_export_MonFeb12205040089714.tif",  # DroneDeploy
            base_path / "Pix4D_FMO04-orthomosaic.tiff",  # pix4D
            base_path / "Metashape-FMO04-orthomosaic.tif",  # metashape
            base_path / "ODM_FMO04-05-06-03-02-2021-orthophoto.tif",  # ODM
        ]

        # Intermediate data path
        interm_path = Path(
            "/Users/christian/PycharmProjects/hnee/image_template_search/data"
        )

        annotations_file_path = base_path / "all_images_2024_11_10.json"

        output_path = interm_path / "output" / "FMO04"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 30

        hA = hA_from_file(file_path=annotations_file_path)
        hA.images = [i for i in hA.images if i.image_name in [drone_image_path.name]]
        assert len(hA.images) == 1, "There should be only a single image left"

    elif scenario == "San_STJB01_10012023_DJI_0068":
        dataset_name = "San_STJB01_10012023"
        base_path = Path(
            f"/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/{dataset_name}"
        )
        bwc = BatchWorkflowConfiguration(base_path=base_path, dataset_name=dataset_name)

        image_url = "https://app.hasty.ai/projects/7899e6d9-6668-45c1-902d-00be21cabf7d/image/0577fd68-f830-45eb-969d-249e8838a28a?datasetId=cb02bc9e-5df3-4894-968d-ed474cf51ae2"
        drone_image_path = (
            base_path
            / "template_images/Snt_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
        )
        annotations_file_path = base_path / "template_images/methods_paper_labels.json"

        orthomosaic_paths = [
            base_path
            / "Snt_STJB01to05_10012023_orthomosaic_DDeploy.tif",  # DroneDeploy
            base_path / "Snt_STJB01_10012023_orthomosaic_Pix4D.tiff",  # pix4D
            base_path / "Snt_STJB01_10012023_orthomosaic_Agisoft.tif",  # metashape
            base_path / "Snt_STJB01_10012023_orthomosaic_ODM.tif",  # ODM
        ]

        # Intermediate data path
        interm_path = Path(
            "/Users/christian/PycharmProjects/hnee/image_template_search/data"
        )

        tile_base_path = interm_path / "tiles"
        cache_path = interm_path / "cache"
        output_path = interm_path / "output" / "Snt_STJB01_10012023"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 60

    # FCD01_02_03 Scenario
    # This is one of the biggest orthomosaics we can find.
    elif scenario == "FCD01_02_03_DJI_0366":
        dataset_name = "FCD01_02_03"
        base_path = Path(
            f"/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/{dataset_name}"
        )

        bwc = BatchWorkflowConfiguration(base_path=base_path, dataset_name=dataset_name)

        image_url = "https://app.hasty.ai/projects/7899e6d9-6668-45c1-902d-00be21cabf7d/image/a012b2fd-6250-4af7-8d93-51fcf36930bf?datasetId=581cdd49-61ad-4e35-9dff-c6847a4f2db0"
        drone_image_path = (
            base_path
            / "template_images/Fer_FCD01-02-03_20122021_single_images/DJI_0366.JPG"
        )
        annotations_file_path = base_path / "template_images/2024_11_26_labels.json"

        ## Orthomosaics
        orthomosaic_paths = [
            base_path / "DroneDeploy_FCD010203_Orthomosaic.tif",  # Drone Deploy
            base_path / "Metashape_FCD01-02-03-orthomosaic.tif",  # Metashape
            base_path / "Pix4D_FCD01-02-03-orthomosaic.tiff",  # Pix4D
            base_path / "ODM_FCD01-02-03-orthophoto.tif",  # OpenDroneMap
        ]

        # Intermediate data path
        interm_path = Path(
            "/Users/christian/PycharmProjects/hnee/image_template_search/data"
        )

        output_path = base_path / f"output_{scenario}"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 60

    elif scenario == "San_STJB06_10012023_DJI_0145":
        dataset_name = "Snt_STJB06_12012023"
        base_path = Path(
            f"/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/{dataset_name}/"
        )

        bwc = BatchWorkflowConfiguration(base_path=base_path, dataset_name=dataset_name)

        image_url = "https://app.hasty.ai/projects/4a06f769-bcb8-4854-98f8-91e8de86021b/image/b25257aa-3b13-49dd-b433-df77805de6c6"
        drone_image_path = (
            base_path
            / "template_images/Snt_STJB06_12012023_DJI_0145/Snt_STJB06_12012023_DJI_0145.JPG"
        )
        annotations_file_path = base_path / "template_images/labels.json"

        ## Orthomosaics
        orthomosaic_paths = [
            base_path
            / "cog/all/Snt_STJB01to06_2012023_DDeploy_cog_pyramids.tif",  # Drone Deploy
            base_path / "Snt_STJB06_12012023_orthomosaic_Agisoft.tif",  # Agisoft
            base_path / "Snt_STJB06_10012023_orthomosaic_Agisoft_deghost.tif",
            # Agisoft using the deghosting
            base_path / "Snt_STJB06_12012023_orthomosaic_Pix4D.tiff",  # Pix4D
            base_path
            / "Snt_STJB06_12012023_orthomosaic_Pix4D_deghost.tiff",  # Pix4D with deghosting
            base_path / "Snt_STJB06_12012023_orthomosaic_ODM.tif",  # Open Dronemap
        ]

        # Intermediate data path
        interm_path = Path(
            "/Users/christian/PycharmProjects/hnee/image_template_search/data"
        )

        output_path = base_path / f"output_{scenario}"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 60

    else:
        raise ValueError("Wrong scenario given")

    for orthomosaic_path in orthomosaic_paths:
        c = WorkflowConfiguration(
            base_path=base_path,
            image_url=image_url,
            drone_image_path=drone_image_path,
            annotations_file_path=annotations_file_path,
            orthomosaic_path=orthomosaic_path,
            interm_path=interm_path,
            output_path=output_path / orthomosaic_path.stem,
            buffer_distance=buffer_distance,
        )

        bwc.workflow_configurations.append(c)

    return bwc


if __name__ == "__main__":

    # scenario = "San_STJB01_10012023_DJI_0068"
    # scenario = "San_STJB06_10012023_DJI_0145"
    # scenario = "FCD01_02_03_DJI_0366"
    scenario = "FMO04_tracking"

    # Get the current date and time
    now = datetime.now()

    # Format the date, hours, and minutes into a string
    formatted_string = now.strftime("%Y-%m-%d %H:%M")

    bwc = get_config(scenario=scenario)
    bwc_file_path = Path(
        f"./workflow_configs/batched_workflow_config_{bwc.dataset_name}.yaml"
    )
    persist_file(config=bwc, file_path=bwc_file_path)

    logger.info(f"Wrote config to {bwc_file_path.resolve()}")
