"""
Workflow to find a template image in a large orthomosaic

## add FIFTYONE_CVAT_USERNAME and FIFTYONE_CVAT_PASSWORD to your environment variables
os.environ["FIFTYONE_CVAT_USERNAME"] = "karisu"
os.environ["FIFTYONE_CVAT_PASSWORD"] = "Q6YRN7Z8Z4f5X4S"
"""
from PIL import Image
Image.MAX_IMAGE_PIXELS = 5223651122

from dataclasses import asdict

from image_template_search.types.workflow_config import WorkflowConfiguration, WorkflowReportConfiguration, persist_file
from image_template_search.util.HastyAnnotationV2 import hA_from_file
from image_template_search.util.util import get_exif_metadata
from image_template_search.util.util import visualise_image, visualise_polygons
from image_template_search.image_patch_finder import ImagePatchFinderCV

from rasterio import CRS
from image_template_search.geospatial_transformations import convert_point_crs, create_buffer_box, \
    save_polygon_as_geojson
from loguru import logger
import rasterio
from image_template_search.clip_by_location import clip_orthomoasic_by_location
import shapely
from image_template_search.image_similarity import ImagePatchFinderLG
from image_template_search.util.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage
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
    assert isinstance(c, WorkflowConfiguration), "c should be a WorkflowConfiguration object"

    c.output_path.mkdir(exist_ok=True, parents=True)

    hA = hA_from_file( file_path=c.annotations_file_path )
    hA.images = [i for i in hA.images if i.image_name in [c.drone_image_path.name]]
    assert len(hA.images) == 1, "There should be only a single image left"
    drone_image_label = hA.images[0]

    wrconf = WorkflowReportConfiguration(**asdict(c))



    visualise_image(image_path=c.drone_image_path, show=True, dpi=75, title="Drone Image")

    orthomosaic_proj_path = c.interm_path  / f"{c.orthomosaic_path.stem}_proj.tif"
    wrconf.orthomosaic_proj_path = orthomosaic_proj_path

    # clipped orthomosaic
    orthomosaic_crop_path = c.interm_path / f"{c.orthomosaic_path.stem}_cropped.tif"
    wrconf.orthomosaic_crop_path = orthomosaic_crop_path

    # project_orthomsaic(orthomosaic_path, orthomosaic_proj_path, target_crs="EPSG:4326")

    image_meta_data = get_exif_metadata(c.drone_image_path)
    location_long_lat = shapely.Point(image_meta_data.longitude, image_meta_data.latitude)

    with rasterio.open(str(c.orthomosaic_path)) as src:
        crs = src.crs
        logger.info(f"Orthomosaic CRS: {crs}")
        epsg = crs.to_epsg()
        logger.info(f"Orthomosaic EPSG: {epsg}")

        if epsg != 4326:
            projected_point = convert_point_crs(location_long_lat, target_crs=crs, source_crs="EPSG:4326")
            buffer = create_buffer_box(projected_point, buffer_distance=c.buffer_distance)

        else:
            logger.warning(f"The EPSG Code is 4326 we need to assume EPSG:32715 for the Galapagos")
            target_crs = CRS({'init': "EPSG:32715"})
            projected_point = convert_point_crs(location_long_lat, target_crs=target_crs.__str__(), source_crs="EPSG:4326")
            buffer = create_buffer_box(projected_point, buffer_distance=c.buffer_distance)

            buffer = convert_point_crs(buffer, source_crs=target_crs.__str__(), target_crs="EPSG:4326")



    # TODO create the buffer as EPSG:4326 if necessary
    buffer_geojson_path = c.interm_path / f"{c.drone_image_path.stem}_{c.buffer_distance}_buffer.geojson"
    save_polygon_as_geojson(buffer, buffer_geojson_path, EPSG_code=epsg)
    wrconf.buffer_geojson_path = buffer_geojson_path

    # #### Step 1 - rough location
    # The process is to first find the rough location of the image by extracting the geolocation, then crop a buffer from this.
    clip_orthomoasic_by_location(bounding_box=buffer,
                                 orthomosaic_path=c.orthomosaic_path,
                                 cropped_orthomosaic_path=orthomosaic_crop_path)

    logger.info(f"Clipped orthomosaic saved to {orthomosaic_crop_path}")



    visualise_image(image_path=orthomosaic_crop_path, show=True, dpi=75, title="Cropped and projected Mosaic image")

    # #### Step 2 - cutout
    # Within the rough location cut the template image

    # Using OpenCV
    # This is memory efficient and not as scale dependent as the LightGlue based matching. But it is often inaccurate especialle with unsharp images.



    ipf = ImagePatchFinderCV(template_path=c.drone_image_path,
                                 large_image_path=orthomosaic_crop_path)

    ipf.find_patch()
    ax_i = visualise_image(image_path=ipf.large_image_path, show=False, dpi=150, title="Project Orthomosaic")
    visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_i, show=True, color="red", linewidth=4)

    # Use the the LightGLue Based Matching instead


    ipf = ImagePatchFinderLG(template_path=c.drone_image_path,
                                 large_image_path=orthomosaic_crop_path)

    ipf.find_patch()
    ax_i = visualise_image(image_path=ipf.large_image_path, show=False, dpi=150)
    visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_i, show=True, color="red", linewidth=4)

    # #### Step 3 - Warp the orthomosaic to the template extent


    projected_image_2_path = project_image(ipf.M, template_path=c.drone_image_path, large_image_path=orthomosaic_crop_path,
                                           output_path=c.output_path, visualise=True)

    logger.info(f"projected_image_2_path: {projected_image_2_path}")
    wrconf.projected_image_2_path = projected_image_2_path

    # #### Step 4 - Refined projection of the labes
    # Project labels from the template image to orthomosaic as an indicator where iguanas are.

    from find_template_and_project_labels import single_stage_template_matching_projection

    projected_labels = single_stage_template_matching_projection(template_image_path=c.drone_image_path,
                                                               large_image_path=projected_image_2_path,
                                                               drone_image_labels=drone_image_label.labels)

    ax_p = visualise_image(image_path=projected_image_2_path, show=False, title="projected labels", dpi=150)
    ax_p = visualise_polygons(polygons=[x.bbox_polygon for x in projected_labels], ax=ax_p, linewidth=4, color="red", show=True)


    logger.info(f"projected_labels: {projected_labels}")

    # #### Step 5 - create an Annotation for later use
    #



    image = PILImage.open(projected_image_2_path)
    # Ensure the bounding box fits within the image dimensions
    frame_width, frame_height = image.size


    annotated_projected_image = AnnotatedImage(image_name=projected_image_2_path.name,
                                           height=frame_height, width=frame_width,
                                           labels=projected_labels)

    # create the new annotation, duplicate the metadata of the original Annotation, because we just moved the annotations around.
    hA_projected = HastyAnnotationV2(label_classes=hA.label_classes,
                           project_name="cutouts",
                           images=[annotated_projected_image],
                           keypoint_schemas=hA.keypoint_schemas,
                           tag_groups=hA.tag_groups,
                           attributes=hA.attributes)

    projected_annotation_path = c.output_path / f"template_annotations_projected.json"
    combined_annotations_file_path = c.output_path / f"combined_annotations.json"

    wrconf.projected_annotation_path = projected_annotation_path
    wrconf.combined_annotations_file_path = combined_annotations_file_path

    with open(projected_annotation_path, 'w') as json_file:
        json_file.write(hA_projected.model_dump_json())
        logger.info(f"Wrote annotations to: {projected_annotation_path}")


    # #### Step 6 - Refine the annotations using FiftyOne and CVAT
    # If the Orthomosaic is perfect and incorporated the drone image completely the annotation should match the single image.
    # Due to distortions or moved animals either animals will be missing, duplicated or simply somewhere else.
    #
    # This can be corrected now.

    logger.info(f"projected_image_2_path: {projected_image_2_path}")



    # annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels_points.json"
    # load hasty annotations
    hA = hA_from_file(file_path=c.annotations_file_path)
    hA_images = [i for i in hA.images if i.image_name in [c.drone_image_path.name]]

    hA_projected = hA_from_file(file_path=projected_annotation_path)
    hA_projected_images = [i for i in hA_projected.images if i.image_name in [projected_image_2_path.name]]


    logger.info(f"annotations_file_path: {c.annotations_file_path}")



    hA.images = hA_images + hA_projected_images
    images_set = [projected_image_2_path, c.drone_image_path]
    assert len(hA.images) == 2, "There should be two images in there"

    dataset_name = f"projection_comparison_{c.orthomosaic_path.stem}__{c.drone_image_path.stem}"
    hA.save(combined_annotations_file_path)
    wrconf.dataset_name = dataset_name


    import shutil
    shutil.copy(c.drone_image_path, c.output_path)

    # ## Delete the dataset if it exists

    ## ONLY DELETE THIS IF YOU WANT TO START FROM SCRATCH
    # fo.delete_dataset(dataset_name)


    # create dot annotations
    dataset = debug_hasty_fiftyone(hA_dets=hA.images, hA_gt=hA.images, images_set=images_set, dataset_name=dataset_name, type="points")

    ## Launch FiftyOne inspection app
    # session = fo.launch_app(dataset, port=5151)
    # session.wait()

    # sample_id = dataset.first().id
    # view = dataset.select(sample_id)

    for i, s in enumerate(dataset):
        logger.info(f"sample {i} of the dataset")

    # Step 3: Send samples to CVAT

    # A unique identifier for this run
    # anno_key = f"cvat_basic_recipe_{orthomosaic_path.stem}"
    anno_key = dataset_name

    dataset.annotate(
        anno_key,
        # Using a organization requires a team account for 33USD
        # project_name = "Orthomosaic_quality_control",
        #organization=organization
        label_field="ground_truth_points",
        attributes=["iscrowd"],
        launch_editor=True,
    )
    print(dataset.get_annotation_info(anno_key))

    persist_file(file_path=wrconf.output_path / f"workflow_report_{c.orthomosaic_path.stem}.yaml", config=wrconf)

    # #### Step 7 - change the annotations in CVAT
    # Review and update the labels on cvat.ai
    #
    #

    # #### Step 8 - download the annotations again
    # TODO
    # Look into the human_in_the_loop_delete script