import copy
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

    for large_image in other_images:
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


                # project B annotations to template
                large_image_proj_labels = [project_bounding_box(l, ipf.M_) for l in copy.deepcopy(large_image.labels)]

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
                                              color="red", show=True, ax=ax_c,
                                              linewidth=7.5, filename=output_path / f"cropped_{large_image.image_name}_{len(large_image_labels_containing)}_objects.jpg")

                    i = Image(image_name=warped_path.name, height=frame_height, width=frame_width, labels=large_image_labels_containing)

                    covered_objects.append(i)

            else:
                logger.warning(f"The frame is not within the template polygon")

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

    ann_template_image = hA.images[0]
    ann_template_image = hA.images[1]


    # other_images = [hA.images[14]]  # take a single image which covers perfectly
    # other_images = hA.images[14:17]  # take testing subset
    # other_images = [i for i in hA.images if i.image_name == "DJI_0058.JPG"] # take testing subset
    other_images = hA.images  # take all

    p_image = PILImage.open(
        images_path / ann_template_image.dataset_name / ann_template_image.image_name)  # Replace with your image file path

    # FIXME this should not return 14 extents if there 14 objects in the image
    objs_in_template, template_extents = find_objects(ann_template_image, patch_size=1280)

    # crop
    templates = crop_objects_from_image(image=p_image,
                                        bbox_polygons=template_extents)  # TODO refactor this

    for i, t in enumerate(templates):

        template_image_path = output_path / f"{ann_template_image.image_name}_template_{i}.jpg"
        t.save(template_image_path)  # a bit

        ax_q = visualise_image(image=t, show=False, title=f"template {i} with annotations", output_file_name=None)
        visualise_polygons([x.bbox_polygon for x in objs_in_template[i]], color="blue",
                           filename=output_path / f"{ann_template_image.image_name}_template_ann_{i}.jpg",
                           show=True, ax=ax_q, linewidth=4.5)



        covered_objects = cutout_detection_deduplication(
            source_image_path=images_path / ann_template_image.dataset_name / ann_template_image.image_name,
            template_image_path=template_image_path,
            cutout_polygon=template_extents[i],
            template_labels=objs_in_template[i],
            other_images=other_images,
            images_path=images_path,
            output_path=output_path)

        # we have N template crops and N covered objects
        logger.info(f"Found  {len(covered_objects)} covered objects")




if __name__ == '__main__':

    demo_template()
