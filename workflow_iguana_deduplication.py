"""
Workflow to find a template image in a large orthomosaic

## add FIFTYONE_CVAT_USERNAME and FIFTYONE_CVAT_PASSWORD to your environment variables
os.environ["FIFTYONE_CVAT_USERNAME"] = "karisu"
os.environ["FIFTYONE_CVAT_PASSWORD"] = "Q6YRN7Z8Z4f5X4S"
"""

from PIL import Image

from conf.config_dataclass import CacheConfig

Image.MAX_IMAGE_PIXELS = 5223651122

from dataclasses import asdict
import shutil
from image_template_search.types.workflow_config import (
    WorkflowConfiguration,
    WorkflowReportConfiguration,
    persist_file,
)
from image_template_search.util.HastyAnnotationV2 import hA_from_file
from image_template_search.util.util import get_exif_metadata
from image_template_search.util.util import visualise_image, visualise_polygons
from image_template_search.image_patch_finder import ImagePatchFinderCV

from rasterio import CRS
from image_template_search.geospatial_transformations import (
    convert_point_crs,
    create_buffer_box,
    save_polygon_as_geojson,
)
from loguru import logger
import rasterio
from image_template_search.clip_by_location import clip_orthomoasic_by_location
import shapely
from image_template_search.image_similarity import ImagePatchFinderLG
from image_template_search.util.HastyAnnotationV2 import (
    HastyAnnotationV2,
    AnnotatedImage,
)
from PIL import Image as PILImage

from image_template_search.image_patch_finder import project_image

from examples.review_annotations import debug_hasty_fiftyone


#
# First the template needs to be found in the potentially arbitrary large image. Since the drone images are georeferenced and the orthomosaics the mosaic is cropped first by its rough location. Within the crop the drone is used as a template.
#
# This is done by tiling the large image and then searching for the template in each tile. The results are then projected back to the large image.


def workflow_project_single_image_drone_and_annotations(c: WorkflowConfiguration):
    """
    # ### Process
    # We start with three files.
    # - the annotations of the single image
    # - the single image itself
    # - the orthomosaic which contains the same cover area as the single image


    :param c:
    :return:
    """
    assert isinstance(
        c, WorkflowConfiguration
    ), "c should be a WorkflowConfiguration object"
    logger.info(f"process {c.drone_image_path} and {c.orthomosaic_path}")
    c.output_path.mkdir(exist_ok=True, parents=True)

    hA = hA_from_file(file_path=c.annotations_file_path)
    hA.images = [i for i in hA.images if i.image_name in [c.drone_image_path.name]]
    assert len(hA.images) == 1, "There should be only a single image left"
    drone_image_label = hA.images[0]

    wrconf = WorkflowReportConfiguration(**asdict(c))

    visualise_image(
        image_path=c.drone_image_path, show=CacheConfig.visualise_matching, dpi=75, title="Drone Image",
        output_file_name=c.output_path / f"{c.drone_image_path.stem}_drone_image.jpg"
    )

    orthomosaic_proj_path = c.output_path / f"{c.orthomosaic_path.stem}_proj.tif"
    wrconf.orthomosaic_proj_path = orthomosaic_proj_path

    # clipped orthomosaic
    orthomosaic_crop_path = c.output_path / f"{c.orthomosaic_path.stem}_cropped.tif"
    wrconf.orthomosaic_crop_path = orthomosaic_crop_path

    # project_orthomsaic(orthomosaic_path, orthomosaic_proj_path, target_crs="EPSG:4326")

    image_meta_data = get_exif_metadata(c.drone_image_path)
    location_long_lat = shapely.Point(
        image_meta_data.longitude, image_meta_data.latitude
    )

    with rasterio.open(str(c.orthomosaic_path)) as src:
        crs = src.crs
        logger.info(f"Orthomosaic CRS: {crs}")
        epsg = crs.to_epsg()
        logger.info(f"Orthomosaic EPSG: {epsg}")

        if epsg != 4326:
            projected_point = convert_point_crs(
                location_long_lat, target_crs=crs, source_crs="EPSG:4326"
            )
            buffer = create_buffer_box(
                projected_point, buffer_distance=c.buffer_distance
            )

        else:
            logger.warning(
                f"The EPSG Code is 4326 we need to assume EPSG:32715 for the Galapagos"
            )
            target_crs = CRS({"init": "EPSG:32715"})
            projected_point = convert_point_crs(
                location_long_lat,
                target_crs=target_crs.__str__(),
                source_crs="EPSG:4326",
            )
            buffer = create_buffer_box(
                projected_point, buffer_distance=c.buffer_distance
            )

            buffer = convert_point_crs(
                buffer, source_crs=target_crs.__str__(), target_crs="EPSG:4326"
            )

    buffer_geojson_path = (
        c.output_path / f"{c.drone_image_path.stem}_{c.buffer_distance}_buffer.geojson"
    )
    save_polygon_as_geojson(buffer, buffer_geojson_path, EPSG_code=epsg)
    wrconf.buffer_geojson_path = buffer_geojson_path

    # #### Step 1 - rough location
    # The process is to first find the rough location of the image by extracting the geolocation, then crop a buffer from this.
    clip_orthomoasic_by_location(
        bounding_box=buffer,
        orthomosaic_path=c.orthomosaic_path,
        cropped_orthomosaic_path=orthomosaic_crop_path,
    )

    logger.info(f"Clipped orthomosaic saved to {orthomosaic_crop_path}")

    visualise_image(
        image_path=orthomosaic_crop_path,
        show=CacheConfig.visualise_matching,
        dpi=75,
        title="Cropped and projected Mosaic image",
        output_file_name=c.output_path / f"{c.orthomosaic_path.stem}_cropped_by_location.jpg"
    )

    # #### Step 2 - cutout
    # Within the rough location cut the template image

    # Using OpenCV
    # This is memory efficient and not as scale dependent as the LightGlue based matching. But it is often inaccurate especialle with unsharp images.

    # ipf = ImagePatchFinderCV(
    #     template_path=c.drone_image_path, large_image_path=orthomosaic_crop_path
    # )
    #
    # ipf.find_patch()
    # ax_i = visualise_image(
    #     image_path=ipf.large_image_path,
    #     show=False,
    #     dpi=150,
    #     title="Project Orthomosaic",
    # )
    # visualise_polygons(
    #     polygons=[ipf.proj_template_polygon],
    #     ax=ax_i,
    #     show=CacheConfig.visualise_matching,
    #     color="red",
    #     linewidth=4,
    # )

    # Use the the LightGLue Based Matching instead

    ipf_lg = ImagePatchFinderLG(
        template_path=c.drone_image_path, large_image_path=orthomosaic_crop_path
    )

    ipf_lg.find_patch()
    try:
        ax_i = visualise_image(image_path=ipf_lg.large_image_path, show=False, dpi=150)

        visualise_polygons(
            polygons=[ipf_lg.proj_template_polygon],
            ax=ax_i,
            show=CacheConfig.visualise_matching,
            color="red",
            linewidth=4,
            filename=c.output_path / f"{c.orthomosaic_path.stem}_template.jpg"
        )
    except Exception as e:
        logger.error(f"Error visualising: {e}")
        if ipf_lg.proj_template_polygon is None:
            logger.error("template not found")

    # #### Step 3 - Warp the orthomosaic to the template extent

    projected_image_2_path = project_image(
        ipf_lg.M,
        template_path=c.drone_image_path,
        large_image_path=orthomosaic_crop_path,
        output_path=c.output_path,
        visualise=CacheConfig.visualise_matching,
    )

    logger.info(f"projected_image_2_path: {projected_image_2_path}")
    wrconf.projected_image_2_path = projected_image_2_path

    # #### Step 4 - Refined projection of the labes
    # Project labels from the template image to orthomosaic as an indicator where iguanas are.

    from find_template_and_project_labels import (
        single_stage_template_matching_projection,
    )

    projected_labels = single_stage_template_matching_projection(
        template_image_path=c.drone_image_path,
        large_image_path=projected_image_2_path,
        drone_image_labels=drone_image_label.labels,
    )

    ax_p = visualise_image(
        image_path=projected_image_2_path, show=False, title="projected labels", dpi=150
    )
    ax_p = visualise_polygons(
        polygons=[x.bbox_polygon for x in projected_labels],
        ax=ax_p,
        linewidth=4,
        color="red",
        show=CacheConfig.visualise_matching,
    )

    logger.info(f"projected_labels: {len(projected_labels)}")

    # #### Step 5 - create an Annotation for later use
    #

    image = PILImage.open(projected_image_2_path)
    # Ensure the bounding box fits within the image dimensions
    frame_width, frame_height = image.size

    annotated_projected_image = AnnotatedImage(
        image_name=projected_image_2_path.name,
        height=frame_height,
        width=frame_width,
        labels=projected_labels,
    )

    # create the new annotation, duplicate the metadata of the original Annotation, because we just moved the annotations around.
    hA_projected = HastyAnnotationV2(
        label_classes=hA.label_classes,
        project_name="cutouts",
        images=[annotated_projected_image],
        keypoint_schemas=hA.keypoint_schemas,
        tag_groups=hA.tag_groups,
        attributes=hA.attributes,
    )

    projected_annotation_path = c.output_path / f"template_annotations_projected.json"

    wrconf.projected_annotation_path = projected_annotation_path

    with open(projected_annotation_path, "w") as json_file:
        json_file.write(hA_projected.model_dump_json())
        logger.info(f"Wrote annotations to: {projected_annotation_path}")

    wrconf.projected_annotation_path = projected_annotation_path

    # If the Orthomosaic is perfect and incorporated the drone image completely the annotation should match the single image.
    # Due to distortions or moved animals either animals will be missing, duplicated or simply somewhere else.
    #
    # This can be corrected now.

    logger.info(f"projected_image_2_path: {projected_image_2_path}")

    # load hasty annotations
    hA = hA_from_file(file_path=c.annotations_file_path)
    # hA_images = [i for i in hA.images if i.image_name in [c.drone_image_path.name]]

    hA_projected = hA_from_file(file_path=projected_annotation_path)
    hA_projected_images = [
        i for i in hA_projected.images if i.image_name in [projected_image_2_path.name]
    ]

    logger.info(f"annotations_file_path: {c.annotations_file_path}")

    # hA.images = hA_images + hA_projected_images
    hA.images = hA_projected_images
    # images_set = [projected_image_2_path, c.drone_image_path]
    images_set = [projected_image_2_path]
    assert len(hA.images) == 1, "There should be one image in there"


    hA.save(projected_annotation_path)
    wrconf.projected_annotation_file_path = projected_annotation_path
    wrconf.projected_image_2_path = projected_image_2_path

    shutil.copy(c.drone_image_path, c.output_path)

    persist_file(
        file_path=wrconf.output_path
        / f"workflow_report_{c.orthomosaic_path.stem}.yaml",
        config=wrconf,
    )

    return hA, images_set, wrconf
