import copy
import json
from pathlib import Path

from PIL import Image as PILImage
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.affinity import affine_transform

from detection_deduplication import find_objects
from image_template_search.image_similarity import ImagePatchFinder, project_bounding_box, project_annotations_to_crop
from image_template_search.util.HastyAnnotationV2 import hA_from_file, Image, ImageLabel
from image_template_search.util.util import visualise_polygons, visualise_image, \
    crop_objects_from_image


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

    :param template_image:
    :return:
    """
    covered_objects = []
    template_crops = []

    for dest_image in other_images:
        # FIXME this is a bit of a hack, because this image should not be changed, but somewhere it is
        large_image = copy.deepcopy(dest_image)
        logger.info(f"finding template patch in {large_image.image_name}")

        ipf = ImagePatchFinder(template_path=source_image_path,
                               template_polygon=cutout_polygon,
                               large_image_path=images_path / large_image.dataset_name / large_image.image_name)

        ipf_t = ImagePatchFinder(template_path=template_image_path,
                               template_polygon=cutout_polygon,
                               large_image_path=images_path / large_image.dataset_name / large_image.image_name)

        # This is hacky, but we need to make sure that the patch is found properly
        if ipf.find_patch(similarity_threshold=0.1):

            ipf.project_image(output_path=output_path)

            frame_height, frame_width = ipf.warped_image_B.shape[:2]
            frame_polygon = Polygon([(0, 0), (0, frame_height), (frame_width, frame_height), (frame_width, 0)])

            # Check if the polygon is fully within the frame polygon
            if frame_polygon.contains(ipf.proj_template_polygon):
                logger.info(f"The frame is within the template polygon")

                ax_w = visualise_image(image=ipf.warped_image_B, show=False, title=f"Warped {large_image.image_name}",
                                       dpi=75)
                ax_w = visualise_polygons([ipf.template_polygon], color="white", show=False, ax=ax_w, linewidth=2.5)
                # TODO visualise the template labels because they are not shown and the ground truth
                # project B annotations to template
                large_image_proj_labels = [project_bounding_box(l, ipf.M_) for l in copy.deepcopy(large_image.labels)]
                ax_i = visualise_polygons([l.bbox_polygon for l in large_image_proj_labels], color="red", show=False, ax=ax_w,
                                          linewidth=2.5,
                                          filename=output_path / f"annotations_{large_image.image_name}_{template_image_path.stem}.jpg")



                if len([l for l in large_image_proj_labels if cutout_polygon.contains(l.centroid)]) == 0:
                    logger.warning(f"template object is not in the image {large_image.image_name}")

                elif ipf_t.find_patch(similarity_threshold=0.05):
                    warped_path = ipf_t.project_image(output_path=output_path)

                    large_image_proj_labels = [project_bounding_box(l, ipf_t.M_) for l in copy.copy(large_image.labels)]

                    xmin, ymin, xmax, ymax =  ipf_t.template_polygon.bounds
                    frame_width = xmax - xmin
                    frame_height = ymax - ymin
                    frame_polygon = Polygon([ (0, 0), (0, frame_height), (frame_width, frame_height), (frame_width, 0) ])

                    # filter out labels that are not within the template
                    large_image_labels_containing = \
                        [l for l in large_image_proj_labels if frame_polygon.contains(l.centroid)]


                    ax_c = visualise_image(image_path=warped_path,
                                           show=False,
                                           title=f"Cropped {large_image.image_name} with {len(large_image_labels_containing)} objects",
                                           dpi=75)

                    ax_c = visualise_polygons([c.bbox_polygon for c in large_image_labels_containing],
                                              color="red", show=False, ax=ax_c,
                                              linewidth=4.5, filename=output_path / f"cropped_{large_image.image_name}_{template_image_path.stem}_{len(large_image_labels_containing)}_objects.jpg")

                    i = Image(image_name=warped_path.name, height=frame_height, width=frame_width, labels=large_image_labels_containing)

                    covered_objects.append(i)

            else:
                logger.warning(f"The frame is not within the template polygon")

    return covered_objects

def find_annotated_template_matches(images_path: Path,
                                    ann_template_image: Image,
                                    other_images: list[Image], output_path: Path):
    """
    project every object on other images to the template image

    :param images_path:
    :param ann_template_image:
    :param other_images:
    :param output_path:
    :return:
    """
    covered_objects = []

    p_image = PILImage.open(
        images_path / ann_template_image.dataset_name / ann_template_image.image_name)  # Replace with your image file path

    objs_in_template, template_extents = find_objects(ann_template_image, patch_size=1280)

    templates = crop_objects_from_image(image=p_image,
                                        bbox_polygons=template_extents)

    for i, t in enumerate(templates):
        template_image_path = output_path / f"{ann_template_image.image_name}_template_{i}.jpg"
        t.save(template_image_path)  # a bit

        ax_q = visualise_image(image=t, show=False,
                               title=f"{ann_template_image.image_name} template {i} with annotations")
        visualise_polygons([x.bbox_polygon for x in objs_in_template[i]], color="blue",
                           filename=output_path / f"{ann_template_image.image_name}_template_ann_{i}.jpg",
                           show=False, ax=ax_q, linewidth=4.5)

        covered_objects.extend(cutout_detection_deduplication(
            source_image_path=images_path / ann_template_image.dataset_name / ann_template_image.image_name,
            template_image_path=template_image_path,
            cutout_polygon=template_extents[i],
            template_labels=objs_in_template[i],
            other_images=other_images,
            images_path=images_path,
            output_path=output_path)
        )

        # we have N template crops and N covered objects
        logger.info(f"Found  {len(covered_objects)} covered objects")
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

    hA.images = sorted(hA.images, key=lambda obj: obj.image_name)
    # hA.images = hA.images[0:5]  # take only the first 8 images

    # TODO get only the nearest images to the template image

    for i, ann_template_image in enumerate(hA.images):
        # ann_template_image = hA.images[1]

        # other_images = [hA.images[14]]  # take a single image which covers perfectly
        # other_images = hA.images[14:17]  # take testing subset
        # other_images = [i for i in hA.images if i.image_name == "DJI_0058.JPG"] # take testing subset
        # other_images = hA.images  # take all
        # other_images = [i for i in hA.images if i.image_name != ann_template_image.image_name]
        other_images = hA.images[i+1:]
        ## FIXME: ensure when i.e. 49 to 50 was already matched, that 50 to 49 is not matched again ebcause it is reduandant

        # TODO feed only images in there which are nearby
        covered_objects = find_annotated_template_matches(images_path, ann_template_image, other_images, output_path)
        with open(output_path / f"image_data_{ann_template_image.image_name}.json", 'w') as json_file:
            # Serialize the list of Pydantic objects to a list of dictionaries
            image_data_dict_list = [json.loads(image_data.json()) for image_data in covered_objects]

            # Save the list of dictionaries to a JSON file
            json.dump(image_data_dict_list, json_file, indent=4)


if __name__ == '__main__':

    demo_template()
