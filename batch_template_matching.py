import typing

from image_template_search.types.workflow_config import WorkflowConfiguration, persist_file, load_yaml_config
from workflow_iguana_deduplication import workflow_project_single_image_drone_and_annotations

scenario = "Snt_STJB01"
scenario = "Snt_STJB06"
# scenario = "FCD01-02-03"
# scenario = "FMO04_tracking"

from pathlib import Path
from image_template_search.util.HastyAnnotationV2 import hA_from_file




def get_config(scenario: str)-> typing.List[WorkflowConfiguration]:

    if scenario == "FMO04_tracking":
        base_path = Path(
            "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/FMO04")

        image_url = "https://app.hasty.ai/projects/9d8deceb-77be-47aa-978f-abcbdc8e00f2/image/4db0c13f-692b-4ae5-b4bc-591d314d87e6?datasetId=a37b5089-ece1-42c2-bf4d-db92d2dd3e6c"
        drone_image_path = base_path / "images_2024_10_07/single_images/DJI_0066.JPG"

        orthomosaic_path = base_path / "images_2024_10_07/mosaics/mosaic_100.jpg"  # Metashape
        # orthomosaic_path =  base_path / "images_2024_10_07/mosaics/FMO04subsetforChris_Orthomosaic_export_TueAug29134421061401.tif" # DroneDeploy

        # Intermediate data path
        interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

        annotations_file_path = base_path / "all_images_2024_11_10.json"

        tile_base_path = interm_path / "tiles"
        cache_path = interm_path / "cache"
        output_path = interm_path / "output" / "FMO04"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 30

        hA = hA_from_file(file_path=annotations_file_path)
        hA.images = [i for i in hA.images if i.image_name in [drone_image_path.name]]
        assert len(hA.images) == 1, "There should be only a single image left"
        drone_image_label = hA.images[0]

        drone_image_label.image_name

    if scenario == "Snt_STJB01":
        base_path = Path(
            "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")

        image_url = "https://app.hasty.ai/projects/7899e6d9-6668-45c1-902d-00be21cabf7d/image/0577fd68-f830-45eb-969d-249e8838a28a?datasetId=cb02bc9e-5df3-4894-968d-ed474cf51ae2"
        drone_image_path = base_path / "Snt_STJB01_10012023/template_images/Snt_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
        orthomosaic_path = base_path / "Snt_STJB01_10012023/Snt_STJB01to05_10012023_orthomosaic_DDeploy.tif"  # DroneDeploy
        # orthomosaic_path =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Pix4D.tiff" # pix4D
        # orthomosaic_path =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_Agisoft.tif" # metashape

        # Intermediate data path
        interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

        orthomosaic_path_crop = orthomosaic_path.parent.resolve() / f"{orthomosaic_path.stem}_crop.tif"

        annotations_file_path = base_path / "Snt_STJB01_10012023/template_images/methods_paper_labels.json"

        tile_base_path = interm_path / "tiles"
        cache_path = interm_path / "cache"
        output_path = interm_path / "output" / "Snt_STJB01_10012023"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 30

        hA = hA_from_file(file_path=annotations_file_path)
        hA.images = [i for i in hA.images if i.image_name in [drone_image_path.name]]
        assert len(hA.images) == 1, "There should be only a single image left"
        drone_image_label = hA.images[0]

        drone_image_label.image_name

    # FCD01-02-03 Scenario
    # This is one of the biggest orthomosaics we can find.
    #

    if scenario == "FCD01-02-03":
        base_path = Path(
            "/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/FCD01-02-03")

        image_url = "https://app.hasty.ai/projects/7899e6d9-6668-45c1-902d-00be21cabf7d/image/a012b2fd-6250-4af7-8d93-51fcf36930bf?datasetId=581cdd49-61ad-4e35-9dff-c6847a4f2db0"
        drone_image_path = base_path / "template_images/Fer_FCD01-02-03_20122021_single_images/DJI_0366.JPG"
        # orthomosaic_path =  base_path / "Metashape_FCD01-02-03-orthomosaic.tif" # Metashape
        orthomosaic_path = base_path / "DroneDeploy_FCD010203_Orthomosaic.tif"  # Drone Deploy

        # Intermediate data path
        interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

        annotations_file_path = base_path / "template_images/2024_11_13_labels_FCD01-02-03.json"

        tile_base_path = interm_path / "tiles"
        cache_path = interm_path / "cache"
        output_path = interm_path / "output" / "FCD01-02-03"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 30

        hA = hA_from_file(file_path=annotations_file_path)
        hA.images = [i for i in hA.images if i.image_name in [drone_image_path.name]]
        assert len(hA.images) == 1, "There should be only a single image left"
        drone_image_label = hA.images[0]

        drone_image_label.image_name

    if scenario == "Snt_STJB06":
        base_path = Path(
            "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/Snt_STJB06_12012023")

        image_url = "https://app.hasty.ai/projects/4a06f769-bcb8-4854-98f8-91e8de86021b/image/b25257aa-3b13-49dd-b433-df77805de6c6"
        drone_image_path = base_path / "template_images/Snt_STJB06_12012023_DJI_0145/Snt_STJB06_12012023_DJI_0145.JPG"
        annotations_file_path = base_path / "template_images/labels.json"

        ## Orthomosaics
        orthomosaic_paths = [
                             # base_path / "Snt_STJB01to06_12012023-1_orthomosaic_DDeploy.tif",  # Drone Deploy
                             #base_path / "Snt_STJB06_12012023_orthomosaic_Agisoft.tif",  # Agisoft
                             #base_path / "Snt_STJB06_10012023_orthomosaic_Agisoft_deghost.tif",
                             # Agisoft using the deghosting
                             #base_path / "Snt_STJB06_12012023_orthomosaic_Pix4D.tiff",  # Pix4D
                             #base_path / "Snt_STJB06_12012023_orthomosaic_Pix4D_deghost.tiff",  # Pix4D with deghosting
                             base_path / "Snt_STJB06_12012023_orthomosaic_ODM.tif",  # Open Dronemap
                             ]

        # Intermediate data path
        interm_path = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data")

        tile_base_path = interm_path / "tiles"
        cache_path = interm_path / "cache"
        output_path = base_path / "output"

        # buffer distance in meters around drone image to locate location in orthomosaic
        buffer_distance = 60

    for orthomosaic_path in orthomosaic_paths:
        c = WorkflowConfiguration(
            base_path=base_path,
            image_url=image_url,
            drone_image_path=drone_image_path,
            annotations_file_path=annotations_file_path,
            orthomosaic_path=orthomosaic_path,
            interm_path=interm_path,
            tile_base_path=tile_base_path,
            cache_path=cache_path,
            output_path=output_path / orthomosaic_path.stem,
            buffer_distance=buffer_distance,
        )
        yield c

### Yet another Scenario
# TODO STJB06 from andreas shared hasty folder
# TODO STJB01 from andreas shared hasty folder



if __name__ == "__main__":

    for c in get_config(scenario="Snt_STJB06"):
        file_path = c.base_path / f"workflow_config_{c.orthomosaic_path.stem}.yaml"


        persist_file(config=c, file_path=file_path )

        cl = load_yaml_config(yaml_file_path=file_path, cls=WorkflowConfiguration)

        workflow_project_single_image_drone_and_annotations(cl)