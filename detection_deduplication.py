"""

"""

import copy
import gc
import json
import random
import sys
import typing
from pathlib import Path

from PIL import Image as PILImage
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon


from conf.config_dataclass import CacheConfig
from image_template_search.image_similarity import ImagePatchFinder, project_bounding_box, project_annotations_to_crop
from image_template_search.util.CoveredObjectType import CoveredObject
from image_template_search.util.HastyAnnotationV2 import hA_from_file, Image, ImageLabel, label_dist_edge_threshold, \
    HastyAnnotationV2, LabelClass
from image_template_search.util.util import visualise_polygons, visualise_image, \
    crop_templates_from_image, create_box_around_point, calculate_nearest_border_distance, get_template_id, hash_objects

# TODO Wrap this somewhere, i.e. the Config.
# disable info logging messages
# Remove the default logger configuration
# logger.remove()
# # Add a new logger, but filter out INFO level
# logger.add(sink=lambda msg: None, level="INFO")  # This silences INFO level logs
# logger.add(sys.stderr, level="WARNING")


def persist_image_stacks(covered_objects: typing.List[CoveredObject], label_classes: typing.List[LabelClass],
                         output_path: Path) -> typing.List[HastyAnnotationV2]:
    """

    :param covered_objects:
    :param label_classes:
    :param output_path:
    :return:
    """
    stacked_annotations = []

    for i, stack in enumerate(covered_objects):
        template_id = stack.template_id
        with open(output_path / f"template_annotations__{template_id}.json", 'w') as json_file:
            hA = HastyAnnotationV2(label_classes=label_classes, project_name="cutouts", images=stack.covered_templates)
            json_file.write(hA.model_dump_json())

            stacked_annotations.append(hA)

    return stacked_annotations

def find_objects(image: Image, patch_size=1280) -> tuple[list[list[ImageLabel]], list[Polygon], list[ImageLabel], list[ImageLabel]]:
    """
    Find objects in the image and return them as a list of lists of ImageLabels and a list of polygons


    # TODO describe the function properly
    :param image:
    :param patch_size:
    :return:
    """
    covered_objects = []

    template_annotations = []
    template_extents = []

    image.labels.sort(key=lambda label: label.attributes.get("distance_to_nearest_edge", float('inf')), reverse=True)

    ## TODO polygons until all covered.

    for l in image.labels:
        if l not in covered_objects:
            if l.attributes.get("distance_to_nearest_edge", float("inf")) > patch_size / 2:
                # the current object is covered
                every_other_label = [il for il in image.labels if il not in covered_objects]
                pc = l.bbox_polygon.centroid

                buffer = create_box_around_point(pc, a=patch_size, b=patch_size)

                covered_objects.extend([l for l in image.labels if buffer.contains(l.centroid)])

                cropped_annotations, buffer = project_annotations_to_crop(buffer=buffer,
                                                                          imagelabels=every_other_label)
                template_annotations.append(cropped_annotations)
                template_extents.append(buffer)

            else:
                logger.warning(f"Label {l.attributes.get('ID')} is too close to the edge, skipping")

    covered_ids = {label.id for label in covered_objects}
    uncovered_labels = [label for label in image.labels if label.id not in covered_ids]
    covered_labels = [label for label in image.labels if label.id in covered_ids]

    return template_annotations, template_extents, covered_labels, uncovered_labels


def cutout_detection_deduplication(source_image_path: Path,
                                   template_image_path: Path,
                                   cutout_polygon: Polygon,
                                   template_labels: list[ImageLabel],
                                   other_images: list[Image],
                                   images_path: Path,
                                   output_path: Path,
                                   ) -> list[Image]:
    """
    Cutout the detection from the image

    :param template_image_path:
    :param output_path:
    :param images_path:
    :param other_images:
    :param template_labels: # TODO add these optional labels to the code
    :param cutout_polygon:
    :param source_image_path:
    :param template_image:
    :return:
    """
    covered_objects = []

    logger.info(f"Looking for these objects in {[l.attributes['ID'] for l in template_labels]} images")


    for dest_image in other_images:
        # FIXME this is a bit of a hack, because this image should not be changed, but somewhere it is
        large_image = copy.deepcopy(dest_image)
        logger.info(f"finding template patch in {large_image.image_name}")

        ipf = ImagePatchFinder(template_path=source_image_path,
                               template_polygon=cutout_polygon,
                               large_image_path=images_path / large_image.dataset_name / large_image.image_name)


        # Match and find homography for both full size images
        if ipf.find_patch(similarity_threshold=0.1):
            # Matching to full images
            logger.info(f"Found {source_image_path.stem} patch in {large_image.image_name}")

            ipf.project_image(output_path=output_path)

            frame_height, frame_width = ipf.warped_image_B.shape[:2]
            frame_polygon = Polygon([(0, 0), (0, frame_height), (frame_width, frame_height), (frame_width, 0)])

            # Check if the projected template polygon is fully within the frame polygon
            # if frame_polygon.within(ipf.proj_template_polygon):

            # if frame_polygon.contains(ipf.proj_template_polygon):
            if ipf.proj_template_polygon.within(frame_polygon) or True: # TODO this should not matter. Even if it is not within, it should be projected
            # if ipf.proj_template_polygon.within(frame_polygon): # TODO this should not matter. Even if it is not within, it should be projected
                logger.info(f"The small template {template_image_path.stem} frame is within the template polygon")


                # TODO visualise the template labels because they are not shown and the ground truth
                # project B annotations to template
                large_image_proj_labels = [project_bounding_box(l, ipf.M_) for l in copy.deepcopy(large_image.labels)]

                if CacheConfig.visualise_info:
                    ax_i = visualise_image(image=ipf.warped_image_B, show=False, title=f"Warped with annotations {large_image.image_name}",
                                           dpi=75)
                    ax_i = visualise_polygons([ipf.template_polygon], color="white",
                                              ax=ax_i, linewidth=2.5)
                    ax_i = visualise_polygons([l.bbox_polygon for l in large_image_proj_labels], color="red",
                                              show=CacheConfig.show_visualisation,
                                              ax=ax_i,
                                              linewidth=2.5,
                                              filename=output_path / f"annotations_large_{large_image.image_name}_{template_image_path.stem}.jpg")
                    plt.close(ax_i.figure)

                ipf_t = ImagePatchFinder(template_path=template_image_path,
                                         template_polygon=cutout_polygon,
                                         large_image_path=images_path / large_image.dataset_name / large_image.image_name)

                if len([l for l in large_image_proj_labels if cutout_polygon.contains(l.centroid)]) == 0:

                    logger.warning(f"template objects ( {[l.attributes.get('ID') for l in template_labels]} ) is not in the image {large_image.image_name}, which holds: {[l.attributes.get('ID') for l in large_image_proj_labels]}")

                elif ipf_t.find_patch(similarity_threshold=0.0005): # FIXME this is where an error can come from. If both images
                    # have the same size and 100% overlap they can have a 100% similarity, if one image is smaller than the other it can't...
                    logger.info(f"Found template {template_image_path.stem} object is in the image {large_image.image_name}")
                    warped_path = ipf_t.project_image(output_path=output_path)


                    # project the labels from the large image to the template, therefore using their ids
                    large_image_proj_labels = [project_bounding_box(l, ipf_t.M_) for l in copy.copy(large_image.labels)]

                    # TODO project the labels from the template to the large image.

                    xmin, ymin, xmax, ymax = ipf_t.template_polygon.bounds
                    frame_width = xmax - xmin
                    frame_height = ymax - ymin
                    frame_polygon = Polygon([(0, 0), (0, frame_height), (frame_width, frame_height), (frame_width, 0)])

                    # filter out labels that are not within the template
                    large_image_labels_containing = \
                        [l for l in large_image_proj_labels if frame_polygon.contains(l.centroid)]


                    if CacheConfig.visualise_info:

                        ax_c = visualise_image(image_path=warped_path,
                                               show=False,
                                               title=f"Cropped {large_image.image_name} with {len(large_image_labels_containing)} objects",
                                               dpi=75)

                        ax_c = visualise_polygons([c.bbox_polygon for c in large_image_labels_containing],
                                                  color="red", show=CacheConfig.show_visualisation, ax=ax_c,
                                                  linewidth=4.5,
                                                  filename=output_path / f"cropped_{large_image.image_name}_{template_image_path.stem}_{len(large_image_labels_containing)}_objects.jpg")

                        plt.close(ax_c.figure)

                    i = Image(image_name=warped_path.name,
                              height=frame_height, width=frame_width,
                              labels=large_image_labels_containing)

                    covered_objects.append(i)
                    gc.collect()

            else:
                logger.warning(f"The frame is not within the template polygon")

    ## these are supposed to be the same objects on multiple images

    return covered_objects


def find_annotated_template_matches(images_path: Path,
                                    source_image: Image,
                                    other_images: list[Image],
                                    output_path: Path,
                                    patch_size = 1280) -> list[CoveredObject]:
    """
    project every object on other images to the template image

    :param patch_size:
    :param images_path:
    :param source_image:
    :param other_images:
    :param output_path:
    :return:
    """
    # typing list of Image
    covered_objects: typing.List[CoveredObject] = []
    logger.info(f"Looking for objects in {source_image.image_name}, {[l.attributes['ID'] for l in source_image.labels]}")

    distances = calculate_nearest_border_distance([l.centroid for l in source_image.labels], source_image.width,
                                             source_image.height)

    # update the labels with distance
    for label, distance in zip(source_image.labels, distances):
        label.attributes["distance_to_nearest_edge"] = distance

    # sort the labels by distance to the nearest edge
    logger.info(f"Looking for objects in {len(other_images)} images, with distances: {distances} to edge. ")

    objs_in_template, template_extents, covered_labels, uncovered_labels = find_objects(source_image, patch_size=patch_size)

    p_image = PILImage.open(
        images_path / source_image.dataset_name / source_image.image_name)  # Replace with your image file path

    templates = crop_templates_from_image(image=p_image,
                                          bbox_polygons=template_extents)

    if CacheConfig.visualise:
        bd_th = int(patch_size // 2)

        center_polygon = Polygon([(0 + bd_th, bd_th), (source_image.width - bd_th, bd_th),
                                  (source_image.width - bd_th, source_image.height - bd_th),
                                  (bd_th, source_image.height - bd_th)])

        # visualise_polygons([t], color="green", show=True)
        ax = visualise_image(image=p_image, show=False, title=f"Template Extents and missing Objects, {source_image.image_name}")
        ax = visualise_polygons([center_polygon],
                                ax=ax, color="white")
        ax = visualise_polygons([x for x in template_extents],
                                ax=ax, color="white")
        # known_labels = [l for l in source_image.labels if l.id in known_labels]
        # ax = visualise_polygons([x.bbox_polygon for x in covered_labels],
        #                         labels=[x.attributes["ID"] for x in covered_labels],
        #                         ax=ax, color="yellow", linewidth=4.5, show=False)
        ax = visualise_polygons([x.bbox_polygon for x in covered_labels],
                                labels=[x.attributes["ID"] for x in covered_labels],
                                ax=ax, color="blue", linewidth=4.5)
        ax = visualise_polygons([x.bbox_polygon for x in uncovered_labels],
                                labels=[x.attributes["ID"] for x in uncovered_labels],
                                ax=ax, color="green", linewidth=4.5, show=CacheConfig.show_visualisation,
                                filename=output_path / f"template_extents_{source_image.image_name}.jpg")

        plt.close(ax.figure)

    for i, t in enumerate(templates):
        # Iterate through the templates and find the objects in the other images



        combined_hash = hash_objects(objs=objs_in_template[i])
        # template_id = f"{source_image.image_name}_{combined_hash}_{patch_size}"


        template_id = get_template_id(image_name=source_image.image_name, combined_hash=combined_hash, patch_size=patch_size)
        template_image_path = output_path / f"template_source_{template_id}.jpg"
        t.save(template_image_path)  # save the template

        template_image_ann = Image(image_name=template_image_path.name,
                                   height=t.height, width=t.width,
                                   labels=objs_in_template[i])
        if CacheConfig.visualise_info:
            ax_q = visualise_image(image=t, show=False,
                                   title=f"New Objects_{source_image.image_name} template:{i} with annotations:{len(objs_in_template[i])}")
            visualise_polygons([x.bbox_polygon for x in objs_in_template[i]], color="blue",
                               labels=[x.attributes["ID"] for x in objs_in_template[i]],
                               filename=output_path / f"template_ann_{source_image.image_name}_{i}.jpg",
                               show=CacheConfig.show_visualisation, ax=ax_q, linewidth=4.5)

            plt.close(ax_q.figure)

        # putting all matched cutouts in one list
        image_stack = cutout_detection_deduplication(
            source_image_path=images_path / source_image.dataset_name / source_image.image_name,
            template_image_path=template_image_path,
            cutout_polygon=template_extents[i], # ith template extent
            template_labels=objs_in_template[i], # ith set of objects
            other_images=other_images, # all the other images
            images_path=images_path,
            output_path=output_path)

        # the template itself
        image_stack.append(template_image_ann)


        covered_object = CoveredObject(
            template_id=template_id,
            template_image_path=template_image_path,
            source_image_name=source_image.image_name,
            source_image=source_image,
            other_images=other_images,
            covered_templates=image_stack,
            new_objects=objs_in_template[i],
            template_extents=template_extents[i]
        )

        # Append to covered_objects
        covered_objects.append(covered_object)


    return covered_objects


def demo_template():
    """
    take detections, look for those on other images
    :return:
    """
    hA = hA_from_file(
        file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_28.json"))
    images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")

    known_labels = []
    patch_size = 1280 # with less than 1280 the matching doesn't work
    total_object_count = 0
    total_objects = []

    hA.images = sorted(hA.images, key=lambda obj: obj.image_name, reverse=False)

    # True random order
    # random.shuffle(hA.images)

    hA.images = [i for i in hA.images if i.image_name in [
                            "DJI_0049.JPG",
       # "DJI_0050.JPG",
       #  "DJI_0051.JPG",
       #                  "DJI_0052.JPG",
        # "DJI_0053.JPG",
        # "DJI_0054.JPG",
        # "DJI_0055.JPG",
        # "DJI_0056.JPG",
        # "DJI_0057.JPG",
        # "DJI_0058.JPG",
        # "DJI_0059.JPG",
        # "DJI_0060.JPG",
        # "DJI_0061.JPG",
        # "DJI_0062.JPG",
        #                 "DJI_0063.JPG",  # First image with ID 7"
        # "DJI_0064.JPG",
        # "DJI_0065.JPG",
        # "DJI_0066.JPG",
        # "DJI_0067.JPG",
        # "DJI_0068.JPG",
        # "DJI_0069.JPG",
        # "DJI_0070.JPG",
        "DJI_0071.JPG",  # problematic image, not part of 49
        # "DJI_0072.JPG",
        # "DJI_0073.JPG",
        # "DJI_0074.JPG",
        "DJI_0075.JPG",   # 49-75 ID:9 hard to match
        # "DJI_0076.JPG",
        # "DJI_0077.JPG",
        # "DJI_0078.JPG",
        # "DJI_0079.JPG",
        # "DJI_0082.JPG",
        # "DJI_0085.JPG",
        # "DJI_0088.JPG",
        # "DJI_0091.JPG",  # with 71 a probelmatic image
        # "DJI_0094.JPG",
        # "DJI_0097.JPG",
        # "DJI_0100.JPG",
        # "DJI_0101.JPG",
    ]]


    ## remove images without ID 12
    # hA.images = [i for i in hA.images if any(l.attributes.get("ID") == "12" for l in i.labels)]
    # hA.images = [i for i in hA.images if any(l.attributes.get("ID") == "7" for l in i.labels)]



    # ## remove all IDs but X
    # for image in hA.images:
    #     image.labels = [l for l in image.labels if l.attributes.get("ID") == "12"]

    # TODO get only the nearest images to the template image

    # for i, source_image in enumerate(hA.images[0:1]):
    for i, source_image in enumerate(hA.images):
        """
        iterate through all images and find the objects in the template image
        """
        logger.warning(f"Looking for objects in {source_image.image_name}")

        #source_image = hA.images[0]  # take the first image as the template
        other_images = hA.images[i+1:]  # take the next two images as the other images we are looking for annotations in
        # TODO set select nearby images only

        # TODO save the other images and the source images hA annotations
        # hA.save(output_path / f"test_annotations_{source_image.image_name}.json")

        # remove already covered labels from source image
        logger.info(f"Removing known labels {known_labels} from source image {source_image.image_name} to avoid duplications.")
        source_image.labels = [l for l in source_image.labels if l.id not in known_labels]


        ## Re-Identify the objects form the source image in the other images
        covered_objects = find_annotated_template_matches(
            images_path,
            source_image,
            other_images,
            output_path,
            patch_size=patch_size)

        stack_annotations = persist_image_stacks(covered_objects, label_classes=hA.label_classes, output_path=output_path)

        for i, stack in enumerate(covered_objects):

            # since labels from other images are projected back to the template image, we can use their ids and remove them from the next images
            known_labels.extend([il.id for i in stack.covered_templates for il in i.labels])

            total_objects.extend([l.attributes["ID"] for l in stack.new_objects])
            logger.info(f"Total objects in all images: {len(total_objects)}")

            # TODO save the template extends and the covered objects

        logger.warning(f"Total objects in all images: {len(total_objects)}")
        logger.warning(f" objects in all images: {sorted(total_objects)}")

    # return stack_annotations, known_labels

if __name__ == '__main__':
    demo_template()
