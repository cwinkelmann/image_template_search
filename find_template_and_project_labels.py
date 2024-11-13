"""
workflow for andreas Method paper

Find a drone image in an orthomosaic

This entails matching the image to a potentially quite big geotiff/jpf

"""
import PIL

from PIL import Image
Image.MAX_IMAGE_PIXELS = 400000000

import copy
import gc
import typing
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon


from conf.config_dataclass import CacheConfig
from detection_deduplication import find_objects, find_objects_individual_all
from image_template_search.image_similarity import ImagePatchFinder, project_bounding_box
from image_template_search.util.CoveredObjectType import CoveredObject
from image_template_search.util.HastyAnnotationV2 import hA_from_file, ImageLabel, AnnotatedImage
from image_template_search.util.TemplateDataType import TemplateData
from image_template_search.util.util import visualise_image, visualise_polygons, calculate_nearest_border_distance, \
    crop_templates_from_image, crop_image_bounds, hash_objects, get_template_id
from tests.test_detection_deduplication import output_path


def single_stage_template_matching_projection(template_image_path: Path, large_image_path: Path,
                                            drone_image_labels: typing.List[ImageLabel] = None) -> typing.List[ImageLabel]:
    """
    project template annotations on the orthomosaic
    :param template_image_path:
    :param large_image_path:
    :param drone_image_labels:
    :return:
    """
    p_image = Image.open(
        template_image_path)  # Replace with your image file path

    source_image_width, source_image_height = p_image.size
    template_extent = Polygon(
        [(0, 0), (source_image_width, 0), (source_image_width, source_image_height), (0, source_image_height)])

    ## find the rough location of the template drone image in the orthomosaic
    ipf_t = ImagePatchFinder(template_path=template_image_path,
                             template_polygon=template_extent,
                             large_image_path=large_image_path)

    found_match = ipf_t.find_patch(similarity_threshold=0.0005)

    # Remove the shift by setting the translation components to zero
    # ipf_t.M[0, 2] = 0  # Set horizontal translation to zero
    # ipf_t.M[1, 2] = 0  # Set vertical translation to zero

    if found_match:
        logger.info(f"Found template {template_image_path.stem} object is in the image {large_image_path.stem}")

    ax_image = visualise_image(image_path=large_image_path, show=False, title="Orthomosaic", dpi=75)
    visualise_polygons([ipf_t.proj_template_polygon],
                       labels=["template extent"],
                       show=CacheConfig.visualise_info, ax=ax_image, color="red", linewidth=4.5)



    # project the labels from the large image to the template, therefore using their ids
    small_image_proj_labels = [project_bounding_box(l, ipf_t.M) for l in drone_image_labels]

    # Image(image_name=large_image_path.stem,
    #       height=source_image_height, width=source_image_width,
    #       labels=small_image_proj_labels)

    return small_image_proj_labels


def drone_template_orthomosaic_localization(template_image_path: Path, large_image_path: Path,
                                            drone_image_labels: typing.List[ImageLabel] = None) -> AnnotatedImage:
    # TODO extract this
    width = 5472
    height = 3648
    template_extent = Polygon([(0, 0), (width, 0), (width, height), (0, height)])

    # FIXME the objects should not be grouped together. Each object needs to extracted individually. It is not about the
    # counting but correct projection of the label-coordinates
    template_annotations, template_extents, cropped_annotations = find_objects_individual_all(drone_image_labels,
                                                                                         patch_size=CacheConfig.patch_size,
                                                                                              image_width=width,
                                                                                              image_height=height)

    p_image = Image.open(
        template_image_path)  # Replace with your image file path

    templates = crop_templates_from_image(image=p_image,
                                          bbox_polygons=template_extents)

    source_image_collection = []
    # get the original images objects
    for obj_in_template, template_image, template_cutout_extent in zip(cropped_annotations, templates, template_extents):

        combined_hash = hash_objects(objs=[obj_in_template])
        template_id = get_template_id(image_name=template_image_path.stem,
                                    combined_hash=combined_hash,
                                    patch_size=CacheConfig.patch_size)

        template_cutout_image_path = output_path / f"template_source_{template_id}.jpg"
        template_image.save(template_cutout_image_path)

        source_image_collection.append(TemplateData(
            template_image_path=template_cutout_image_path,
            template_extent=template_cutout_extent,
            center_obj_template=obj_in_template,
            template_image=template_image)
        )

    ## find the rough location of the template drone image in the orthomosaic
    ipf_t = ImagePatchFinder(template_path=template_image_path,
                             template_polygon=template_extent,
                             large_image_path=large_image_path)

    found_match = ipf_t.find_patch(similarity_threshold=0.0005)

    # This is just for visualisation
    # warped_path = ipf_t.project_image(output_path=large_image_path.parent / "warped")

    if found_match:
        logger.info(f"Found template {template_image_path.stem} object is in the image {large_image_path.stem}")

    ax_image = visualise_image(image_path=large_image_path, show=False, title="Orthomosaic")
    visualise_polygons([ipf_t.proj_template_polygon],
                       labels=["template extent"],
                       show=CacheConfig.visualise_info, ax=ax_image, color="red", linewidth=4.5)


    # project the labels from the large image to the template, therefore using their ids
    if drone_image_labels is not None:
        small_image_proj_labels = [project_bounding_box(l, ipf_t.M) for l in drone_image_labels]


        # ax_image_w = visualise_image(image_path=warped_path, show=False, title="Warped Orthomosaic")
        # # ax_image_w = visualise_polygons(polygons=[x.bbox_polygon for x in small_image_proj_labels], show=True, ax=ax_image_w)
        # ax_image_w = visualise_polygons(polygons=[x.bbox_polygon for x in drone_image_labels], show=True, ax=ax_image_w, linewidth=4, color="red")

        # projected labels
        ax_image_w = visualise_image(image_path=large_image_path, show=False, title="Original Orthomosaic", dpi=180)
        ax_image_w = visualise_polygons(polygons=[x.bbox_polygon for x in small_image_proj_labels], show=CacheConfig.visualise_info, ax=ax_image_w, linewidth=4, color="red")


        # cut the projected labels with a larger patch size
        template_annotations, template_extents, cropped_annotations = find_objects_individual_all(small_image_proj_labels,
                                                                                            patch_size=CacheConfig.patch_size + CacheConfig.patch_size_offset)

        p_image = Image.open(
            large_image_path)  # Replace with your image file path

        templates = crop_templates_from_image(image=p_image,
                                              bbox_polygons=template_extents)

        destination_image_collection = []


        for obj_in_template, template_image_mosaic, template_mosaic_extent in zip(cropped_annotations, templates, template_extents):

            ax_i = visualise_image(image = template_image_mosaic, show=False, title="Orthomosaic cutout")
            visualise_polygons(polygons=[obj_in_template.bbox_polygon],
                               labels=[obj_in_template.attributes.get("ID", "undefined")],
                               ax=ax_i, show=CacheConfig.visualise_info,
                               color="red", linewidth=2.5)

            combined_hash = hash_objects(objs=[obj_in_template])
            template_id = get_template_id(image_name=template_image_path.stem,
                                          combined_hash=combined_hash,
                                          patch_size=CacheConfig.patch_size + CacheConfig.patch_size_offset)

            template_image_mosaic_path = output_path / f"template_mosaic_{template_id}.jpg"
            template_image_mosaic.save(template_image_mosaic_path)

            destination_image_collection.append(TemplateData(
                template_image_path=template_image_mosaic_path,
                template_extent=template_mosaic_extent,
                center_obj_template=obj_in_template,
                template_image=template_image_mosaic)
            )

            # TODO find the patches of the single image in the drone image

    ## map these to collections
    refined_projected_annotations = []

    for td in destination_image_collection:
        center_obj_template = td.center_obj_template

        # TODO use something else to filter it later
        source_image_template = [td for td in source_image_collection if td.center_obj_template.attributes.get("ID") == center_obj_template.attributes.get("ID")][0]

        width = CacheConfig.patch_size
        height = CacheConfig.patch_size
        template_extent = Polygon([(0, 0), (width, 0), (width, height), (0, height)])

        ipf_refined = ImagePatchFinder(template_path=source_image_template.template_image_path,
                                 template_polygon=template_extent,
                                 large_image_path=td.template_image_path)

        found_match = ipf_refined.find_patch(similarity_threshold=0.0005)

        if found_match:
            logger.info(f"Found template {source_image_template.template_image_path.stem} object is in the image {td.template_image_path.stem}")

            ax_source = visualise_image(image_path=source_image_template.template_image_path, show=False, title="source annotations")
            ax_source = visualise_polygons(polygons=[source_image_template.center_obj_template.bbox_polygon],
                                          show=CacheConfig.show_visualisation, ax=ax_source,
                                          labels=[source_image_template.center_obj_template.attributes.get("ID", "undefined")],
                                          linewidth=4, color="red")

            ax_image = visualise_image(image_path=td.template_image_path, show=False, title="projected annotations")
            ax_image = visualise_polygons([ipf_refined.proj_template_polygon],
                               show=False, ax=ax_image, color="white", linewidth=4.5)

            # project the labels from the large image to the template, therefore using their ids
            center_obj_template_proj_labels = project_bounding_box(source_image_template.center_obj_template, ipf_refined.M)

            ax_image = visualise_polygons(polygons=[center_obj_template_proj_labels.bbox_polygon],
                                          show=CacheConfig.show_visualisation, ax=ax_image,
                                          labels=[center_obj_template_proj_labels.attributes.get("ID", "undefined")],
                                          linewidth=4, color="red")

            refined_projected_annotations.append(center_obj_template_proj_labels)


    ## TODO how can I project these now back to the original image
    refined_projected_annotations

# TODO this should become some sort Class method
def forward_template_matching_projection(
        source_image_path: Path,
        source_image_label: AnnotatedImage,
        dest_image_path: Path,
        output_path: Path,
        dest_image_labels: AnnotatedImage = None,
        patch_size=1280
) -> list[AnnotatedImage]:
    """
    match the source image to the destination image

    :param template_image_path:
    :param output_path:
    :param images_path:
    :param other_images:
    :param source_image_labels: # TODO add these optional labels to the code
    :param cutout_polygon:
    :param source_image_path:
    :param template_image:
    :return:
    """
    covered_objects = []
    source_image_labels = source_image_label.labels
    logger.info(f"Looking for these objects in {[l.attributes.get('ID', None) for l in source_image_labels]} images")
    logger.info(f"finding template patch {source_image_path.stem} in {source_image_path.stem}")

    with Image.open(source_image_path) as img:
        source_image_width, source_image_height = img.size
        source_image_extent = Polygon([(0, 0), (source_image_width, 0), (source_image_width, source_image_height), (0, source_image_height)])

    with Image.open(dest_image_path) as img:
        dest_frame_width, dest_image_height = img.size
        dest_image_extent = Polygon([(0, 0), (dest_frame_width, 0), (dest_frame_width, dest_image_height), (0, dest_image_height)])



    logger.warning(f"remove the source image extent parameter because it is not really necessary")
    ipf = ImagePatchFinder(template_path=source_image_path,
                           template_polygon=source_image_extent,
                           large_image_path=dest_image_path)

    found_match = ipf.find_patch(similarity_threshold=0.0001)

    if dest_image_extent.contains(ipf.proj_template_polygon):
    # Match and find homography for both full size images
        logger.info(f"The source image is completely within the target image")
    else:
        logger.info(f"The source image is not completely within the target image")

    if found_match:
        # Matching to full images
        logger.info(f"Found {source_image_path.stem} patch in {dest_image_path.stem}")
        cropped_destination_image = crop_image_bounds(image=dest_image_path, polygon=ipf.proj_template_polygon)
        cropped_destination_image_path = output_path / f"cropped_{dest_image_path.stem}_{source_image_path.stem}.jpg"
        cropped_destination_image.save(cropped_destination_image_path)

        if CacheConfig.visualise_info:
            ax_i = visualise_image(image_path=dest_image_path, show=False, title=f"{dest_image_path.stem}",
                                   dpi=100)
            ax_i = visualise_polygons([ipf.proj_template_polygon], color="white",
                                      ax=ax_i, linewidth=3.5, show=CacheConfig.show_visualisation)

            plt.close(ax_i.figure)

            # visualise_image(image=cropped_destination_image, show=CacheConfig.show_visualisation, title=f"{cropped_destination_image_path.stem}",
            #                        dpi=100)


        covered_objects: typing.List[CoveredObject] = []
        logger.info(
            f"Looking for objects in {source_image_label.image_name}, {[l.attributes.get('ID') for l in source_image_label.labels]}")

        distances = calculate_nearest_border_distance([l.centroid for l in source_image_label.labels], source_image_label.width,
                                                      source_image_label.height)

        # update the labels with distance
        for label, distance in zip(source_image_label.labels, distances):
            label.attributes["distance_to_nearest_edge"] = distance

        # sort the labels by distance to the nearest edge
        # logger.info(f"Looking for objects in {len(other_images)} images, with distances: {distances} to edge. ")

        objs_in_templates, template_extents, covered_labels, uncovered_labels = find_objects(source_image_label,
                                                                                            patch_size=patch_size)

        p_image = Image.open(
            images_path / source_image_label.dataset_name / source_image_label.image_name)  # Replace with your image file path

        templates = crop_templates_from_image(image=p_image,
                                              bbox_polygons=template_extents)


        for objs_in_template_i, template_image, template_extent in zip(objs_in_templates, templates, template_extents): # TODO this should be the list of template images

            combined_hash = hash_objects(objs=objs_in_template_i)
            template_id = get_template_id(image_name=source_image_label.image_name,
                                          combined_hash=combined_hash,
                                          patch_size=patch_size)

            template_image_path = output_path / f"template_{template_id}.jpg"
            template_image.save(template_image_path)

            visualise_image(image_path=template_image_path, title="template image")

            ipf_t = ImagePatchFinder(template_path=template_image_path,
                                     template_polygon=template_extent,
                                     large_image_path=cropped_destination_image_path)

            template_match = ipf_t.find_patch(similarity_threshold=0.000005) # FIXME this is where an error can come from. If both images
            if template_match:
                logger.info(f"FOUND: the template {template_image_path.stem} is in the image {cropped_destination_image_path.stem}.")
            else:
                logger.warning(f"NOT FOUND: the template {template_image_path.stem} is in the image {cropped_destination_image_path.stem}.")

            # filter out labels that are not within the template
            large_image_labels_containing = \
                [copy.deepcopy(l) for l in source_image_labels if template_extent.contains(l.centroid)]

            # project the labels from the template to the large image
            # template_labels_proj_labels = [project_bounding_box(l, ipf_t.M_) for l in copy.copy(large_image_labels_containing)]

            if CacheConfig.visualise_info:

                ax_c = visualise_image(image=cropped_destination_image, dpi=75)

                ax_c = visualise_polygons([ipf_t.proj_template_polygon],
                                          color="red", ax=ax_c,
                                          linewidth=4.5)

                # ax_c = visualise_polygons([c.bbox_polygon for c in template_labels_proj_labels],
                #                           color="red", show=CacheConfig.show_visualisation, ax=ax_c,
                #                           linewidth=4.5, title=f"Cropped {dest_image_path.stem} with {len(large_image_labels_containing)} objects",
                #                           )

                plt.close(ax_c.figure)



            # i = Image(image_name=warped_path.name,
            #           height=frame_height, width=frame_width,
            #           labels=large_image_labels_containing)
            #
            # covered_objects.append(i)
            gc.collect()


    ## these are supposed to be the same objects on multiple images

    return covered_objects


if __name__ == "__main__":
    # annotations_file_path = Path(
    #     "/Users/christian/data/2TB/ai-core/data/detection_deduplication/all_images_2024_11_10.json")

    base_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    drone_image = base_path / "single_images/DJI_0066.JPG"
    drone_image = base_path / "single_images/DJI_0066.JPG"

    # FMO04
    image_2 = base_path / "mosaics/mosaic_100.jpg"
    image_2 = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/FMO04/DD_FMO04_Orthomosaic_export_MonFeb12205040089714.tif")
    image_2 = Path("/Volumes/2TB/Download_2024_11_12/FMO04_Orthomosaic_MonFeb12205040089714/FMO04_Orthomosaic_export_MonFeb12205040089714.tif")


    ## The image for Andrea:
    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    drone_image = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068/San_STJB01_10012023_DJI_0068.JPG"
    image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif"

    annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels.json"
    # image_2 = base_path / "single_images/DJI_0067.JPG"

    images_path = base_path
    output_path = base_path / "output"

    hA = hA_from_file(file_path=annotations_file_path)
    hA.images = [i for i in hA.images if i.image_name in [drone_image.name]]
    assert len(hA.images) == 1, "There should be only a single image left"
    drone_image_label = hA.images[0]


    # ## Re-Identify the objects form the source image in the other images
    # covered_objects = forward_template_matching_projection(
    #     source_image_path=drone_image,
    #     source_image_label=drone_image_label,
    #     dest_image_path=orthomosaic,
    #     output_path=output_path,
    #     dest_image_labels=None,
    #     patch_size=1280)

    # # ========== Two stage projection ==========
    # projected_labels = drone_template_orthomosaic_localization(template_image_path=drone_image,
    #                                                            large_image_path=orthomosaic,
    #                                                            drone_image_labels=drone_image_label.labels)


    # ========== One stage projection ==========
    projected_labels = single_stage_template_matching_projection(template_image_path=drone_image,
                                                               large_image_path=image_2,
                                                               drone_image_labels=drone_image_label.labels)

    # hA_gt = hA_from_file(
    #     file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/all_images_2024_11_10.json"))
    # hA_gt.images = [i for i in hA_gt.images if i.image_name in [image_2.name]]
    # labels_gt = hA_gt.images[0].labels
    # assert len(hA_gt.images) == 1, "There should be only a single image left"
    # gt_image_label = hA_gt.images[0]

    # debug_hasty_fiftyone(hA, images_path)
    ax_p = visualise_image(image_path=image_2, show=False, title="projected labels", dpi=75)
    ax_p = visualise_polygons(polygons=[x.bbox_polygon for x in projected_labels], ax=ax_p, linewidth=4, color="red", show=True)
    # ax_p = visualise_polygons(polygons=[x.bbox_polygon for x in labels_gt], ax=ax_p, linewidth=4, color="blue",
    #                           show=True)