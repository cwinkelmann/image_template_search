"""

"""

import copy
import json
import typing
from pathlib import Path

from PIL import Image as PILImage
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.affinity import affine_transform

import hydra
from omegaconf import DictConfig

from image_template_search.image_similarity import ImagePatchFinder, project_bounding_box, project_annotations_to_crop
from image_template_search.util.HastyAnnotationV2 import hA_from_file, Image, ImageLabel
from image_template_search.util.util import visualise_polygons, visualise_image, \
    crop_objects_from_image, create_box_around_point, calculate_nearest_border_distance


def find_objects(image: Image, patch_size=1280) -> tuple[list[list[ImageLabel]], list[Polygon]]:
    """
    Find objects in the image and return them as a list of lists of ImageLabels and a list of polygons

    :param image:
    :param patch_size:
    :return:
    """
    covered_objects = []

    template_annotations = []
    template_extents = []

    for l in image.labels:
        # FIXME: an object can be covered by multiple patches. It is returned multiple times. This has to be handled somewhere
        # if True or ( "ID" in l.attributes and l.attributes["ID"] == "2" ): # TODO remove this I just want to test it with ID 12
        if l not in covered_objects:
            # TODO calculate the distance to the center of the image

            # the current object is covered

            every_other_label = [il for il in image.labels if il not in covered_objects]
            pc = l.bbox_polygon.centroid

            # TODO
            buffer = create_box_around_point(pc, a=patch_size, b=patch_size)

            covered_objects.extend([l for l in image.labels if buffer.contains(l.centroid)])

            cropped_annotations, buffer = project_annotations_to_crop(buffer=buffer,
                                                                      imagelabels=every_other_label)
            template_annotations.append(cropped_annotations)
            template_extents.append(buffer)

    return template_annotations, template_extents


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
            logger.info(f"Found Full Image {source_image_path.stem} patch in {large_image.image_name}")

            ipf.project_image(output_path=output_path)

            frame_height, frame_width = ipf.warped_image_B.shape[:2]
            frame_polygon = Polygon([(0, 0), (0, frame_height), (frame_width, frame_height), (frame_width, 0)])

            # Check if the projected template polygon is fully within the frame polygon
            # if frame_polygon.within(ipf.proj_template_polygon):

            # if frame_polygon.contains(ipf.proj_template_polygon):
            if ipf.proj_template_polygon.within(frame_polygon) or True: # TODO this should not matter. Even if it is not within, it should be projected
                logger.info(f"The small template {template_image_path.stem} frame is within the template polygon")


                # TODO visualise the template labels because they are not shown and the ground truth
                # project B annotations to template
                large_image_proj_labels = [project_bounding_box(l, ipf.M_) for l in copy.deepcopy(large_image.labels)]
                ax_i = visualise_image(image=ipf.warped_image_B, show=False, title=f"Warped with annotations {large_image.image_name}",
                                       dpi=75)
                ax_i = visualise_polygons([ipf.template_polygon], color="white",
                                          ax=ax_i, linewidth=2.5)
                ax_i = visualise_polygons([l.bbox_polygon for l in large_image_proj_labels], color="red", show=True,
                                          ax=ax_i,
                                          linewidth=2.5,
                                          filename=output_path / f"annotations_large_{large_image.image_name}_{template_image_path.stem}.jpg")

                ipf_t = ImagePatchFinder(template_path=template_image_path,
                                         template_polygon=cutout_polygon,
                                         large_image_path=images_path / large_image.dataset_name / large_image.image_name)

                if len([l for l in large_image_proj_labels if cutout_polygon.contains(l.centroid)]) == 0:
                    logger.warning(f"template object is not in the image {large_image.image_name}")

                elif ipf_t.find_patch(similarity_threshold=0.005):
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

                    ax_c = visualise_image(image_path=warped_path,
                                           show=False,
                                           title=f"Cropped {large_image.image_name} with {len(large_image_labels_containing)} objects",
                                           dpi=75)

                    ax_c = visualise_polygons([c.bbox_polygon for c in large_image_labels_containing],
                                              color="red", show=True, ax=ax_c,
                                              linewidth=4.5,
                                              filename=output_path / f"cropped_{large_image.image_name}_{template_image_path.stem}_{len(large_image_labels_containing)}_objects.jpg")

                    i = Image(image_name=warped_path.name, height=frame_height, width=frame_width,
                              labels=large_image_labels_containing)

                    covered_objects.append(i)

            else:
                logger.warning(f"The frame is not within the template polygon")

    return covered_objects


def find_annotated_template_matches(images_path: Path,
                                    source_image: Image,
                                    other_images: list[Image],
                                    output_path: Path,
                                    patch_size = 1280) -> list[Image]:
    """
    project every object on other images to the template image

    :param images_path:
    :param source_image:
    :param other_images:
    :param output_path:
    :return:
    """
    # typing list of Image
    covered_objects: typing.List[Image] = []

    p_image = PILImage.open(
        images_path / source_image.dataset_name / source_image.image_name)  # Replace with your image file path

    objs_in_template, template_extents = find_objects(source_image, patch_size=patch_size)

    templates = crop_objects_from_image(image=p_image,
                                        bbox_polygons=template_extents)

    for i, t in enumerate(templates):

        # Save the templates
        template_image_path = output_path / f"template_source_{source_image.image_name}_{i}_{patch_size}.jpg"
        t.save(template_image_path)  # a bit

        template_image_ann = Image(image_name=template_image_path.name,
                                   height=t.height, width=t.width,
                                   labels=objs_in_template[i])

        ax_q = visualise_image(image=t, show=False,
                               title=f"New Objects_{source_image.image_name} template {i} with annotations")
        visualise_polygons([x.bbox_polygon for x in objs_in_template[i]], color="blue",
                           filename=output_path / f"template_ann_{source_image.image_name}_{i}.jpg",
                           show=True, ax=ax_q, linewidth=4.5)

        image_stack = cutout_detection_deduplication(
            source_image_path=images_path / source_image.dataset_name / source_image.image_name,
            template_image_path=template_image_path,
            cutout_polygon=template_extents[i], # ith template extent
            template_labels=objs_in_template[i], # ith set of objects
            other_images=other_images, # all the other images
            images_path=images_path,
            output_path=output_path)

        image_stack.append(template_image_ann)

        covered_objects.append(
            {
                "template_id": i,
                "template_image": template_image_path,
                "source_image": source_image.image_name,
                "covered_objects": image_stack
             }
        )

    return covered_objects


def demo_template():
    """
    take detections, look for those on other images
    :return:
    """
    hA = hA_from_file(
        file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_10.json"))
    images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")

    known_labels = []
    patch_size = 1280
    total_object_count = 0

    hA.images = sorted(hA.images, key=lambda obj: obj.image_name, reverse=False)
    # hA.images = [i for i in hA.images if i.image_name in ["DJI_0057.JPG", "DJI_0058.JPG", "DJI_0060.JPG", "DJI_0062.JPG"]]

    # hA.images = [i for i in hA.images if i.image_name in ["DJI_0049.JPG",
    #                                                       #"DJI_0050.JPG",
    #                                                       # "DJI_0052.JPG",
    #                                                       "DJI_0058.JPG", "DJI_0059.JPG",
    # #                                                      "DJI_0053.JPG", "DJI_0054.JPG", "DJI_0055.JPG",
    #                                                       # "DJI_0056.JPG",
    #                                                       # "DJI_0057.JPG", "DJI_0058.JPG", "DJI_0060.JPG",
    #                                                       "DJI_0062.JPG", "DJI_0063.JPG", "DJI_0064.JPG",
    #               ]]

    hA.images = [i for i in hA.images if i.image_name in ["DJI_0049.JPG",
                                                          "DJI_0050.JPG", "DJI_0051.JPG",
                                                          "DJI_0052.JPG",
                                                          "DJI_0053.JPG",
                                                          "DJI_0054.JPG", "DJI_0055.JPG",
                                                          "DJI_0056.JPG",
                                                          "DJI_0057.JPG", "DJI_0058.JPG", "DJI_0059.JPG",
                                                          "DJI_0060.JPG",
                                                          "DJI_0061.JPG",
                                                          "DJI_0062.JPG",
                                                          "DJI_0063.JPG", # First image with ID 7
                                                          "DJI_0064.JPG",
                                                          "DJI_0065.JPG",
                  ]]


    # hA.images = [i for i in hA.images if i.image_name in ["DJI_0049.JPG", "DJI_0060.JPG"]]
    # hA.images = hA.images[0:10]  # take only the first 8 images

    # TODO get only the nearest images to the template image

    for i, source_image in enumerate(hA.images):
        """
        iterate through all images and find the objects in the template image
        """
        logger.info(f"Looking for objects in {source_image.image_name}")

        #source_image = hA.images[0]  # take the first image as the template
        other_images = hA.images[i+1:]  # take the next two images as the other images we are looking for annotations in

        dist = calculate_nearest_border_distance([l.centroid for l in source_image.labels], source_image.width,
                                          source_image.height)
        logger.info(f"Looking for objects in {len(other_images)} images, with distances: {dist} to edge. ")

        # remove labels which are too close to the border. Only in literal edge cases those are not covered anywhereelse
        # For ID 9 in DJI_0054.JPG this is the case.
        # TODO use the patch size to calculate the buffer
        # TODO log if labels are removed
        # TODO this is bigger then the patch size on purpose, How big should it be?
        bd_th = int( (patch_size**2 // 2) ** 0.5 ) # TODO THIS would be the right way to calculate the distance

        bd_th = int(patch_size // 2)

        source_image.labels = [l for l in source_image.labels if l.centroid.within(
            Polygon([(0 + bd_th , bd_th), (source_image.width - bd_th, bd_th),
                     (source_image.width - bd_th, source_image.height - bd_th), (bd_th, source_image.height - bd_th)]))]

        logger.info(f"After edge thresholding in {len(source_image.labels)} potentially new labels remain on {len(other_images)} images")
        # remove already covered labels from source image
        logger.info(f"Removing known labels {known_labels} from source image {source_image.image_name} to avoid duplications.")
        source_image.labels = [l for l in source_image.labels if l.id not in known_labels]
        total_object_count += len(source_image.labels)
        logger.info(f"Total objects in all images: {total_object_count}")

        # other_images = [hA.images[14]]  # take a single image which covers perfectly
        # other_images = hA.images[14:17]  # take testing subset
        # other_images = [i for i in hA.images if i.image_name == "DJI_0058.JPG"] # take testing subset
        # other_images = hA.images  # take all
        # other_images = [i for i in hA.images if i.image_name != ann_template_image.image_name]

        ## FIXME: ensure when i.e. 49 to 50 was already matched, that 50 to 49 is not matched again ebcause it is reduandant
        ## TODO when there is a template matched in 49 to 50, then this object should not be matched again in 50 to 51
        # TODO feed only images in there which are nearby

        covered_objects = find_annotated_template_matches(
            images_path,
            source_image,
            other_images,
            output_path,
            patch_size=patch_size)

        for i, stack in enumerate(covered_objects):
            hA_template = copy.deepcopy(hA)
            with open(output_path / f"template_annotations_{source_image.image_name}_{i}.json", 'w') as json_file:
                # Serialize the list of Pydantic objects to a list of dictionaries

                hA_template.images = stack["covered_objects"]
                json_file.write(hA_template.model_dump_json())

                # since labels from other images are projected back to the template image, we can use their ids and remove them from the next images
                known_labels.extend( [il.id for i in stack["covered_objects"] for il in i.labels] )

                # find all covered label_ids

        # TODO remove already covered templates/annotations from images further down the road
        print(f"Known labels: {known_labels}, which should not be looked for again.")

if __name__ == '__main__':
    demo_template()
