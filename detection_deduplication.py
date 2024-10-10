import copy
from pathlib import Path

from PIL import Image as PILImage
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.affinity import affine_transform

from image_template_search.image_similarity import ImagePatchFinder
from image_template_search.util.HastyAnnotationV2 import hA_from_file, Image, ImageLabel
from image_template_search.util.util import visualise_polygons, visualise_image, create_box_around_point, \
    crop_objects_from_image


def find_objects(image: Image, patch_size=1280) -> tuple[list[list[ImageLabel]], list[Polygon]]:
    covered_objects = []

    template_annotations = []
    template_extent = []

    for l in image.labels:

        # if True or ( "ID" in l.attributes and l.attributes["ID"] == "2" ): # TODO remove this I just want to test it with ID 12
        if l not in covered_objects:
            # TODO calculate the distance to the center of the image

            covered_objects.append(l)  # the current object is covered
            every_other_label = [il for il in image.labels if il not in covered_objects]

            # create a buffer around the centroid of the polygon
            pc = l.bbox_polygon.centroid
            buffer = create_box_around_point(pc, a=patch_size // 2, b=patch_size // 2)
            minx, miny, maxx, maxy = buffer.bounds

            obj_in_crop = [copy.copy(il) for il in image.labels if
                           il.centroid.within(buffer)]  # all objects withing the buffer
            cropped_annotations = [l for l in obj_in_crop if buffer.contains(l.centroid)]

            a, b, d, e = 1.0, 0.0, 0.0, 1.0  # Scale and rotate
            xoff, yoff = -minx, -miny  # Translation offsets

            # Apply the affine transformation to the polygon to reproject into image coordinates
            transformation_matrix = [a, b, d, e, xoff, yoff]

            for ca in cropped_annotations:
                ca.bbox_polygon = affine_transform(ca.bbox_polygon, transformation_matrix)


            template_annotations.append(cropped_annotations)
            template_extent.append(buffer)

    return template_annotations, template_extent




def cutout_detection(image: Image,
                     images_path: Path,
                     output_path: Path,
                     patch_size=1280,
                     ):
    """
    Cutout the detection from the image
    :param image:
    :return:
    """
    covered_objects = []
    if True or image.image_name in ("DJI_0077.JPG", "DJI_0078.JPG"):


        for l in image.labels:

            # if True or ( "ID" in l.attributes and l.attributes["ID"] == "2" ): # TODO remove this I just want to test it with ID 12
            if l not in covered_objects:

                # TODO calculate the distance to the center of the image

                covered_objects.append(l)  # the current object is covered
                every_other_label = [il for il in image.labels if il not in covered_objects]

                # create a buffer around the centroid of the polygon
                pc = l.bbox_polygon.centroid
                buffer = pc.buffer(patch_size // 2)
                minx, miny, maxx, maxy = buffer.bounds

                obj_in_crop = [copy.copy(il) for il in image.labels if
                               il.centroid.within(buffer)]  # all objects withing the buffer
                cropped_annotations = [l for l in obj_in_crop if buffer.contains(l.centroid)]


                a, b, d, e = 1.0, 0.0, 0.0, 1.0  # Scale and rotate
                xoff, yoff = -minx, -miny  # Translation offsets

                # Apply the affine transformation to the polygon to reproject into image coordinates
                transformation_matrix = [a, b, d, e, xoff, yoff]

                for ca in cropped_annotations:
                    ca.bbox_polygon = affine_transform(ca.bbox_polygon, transformation_matrix)

                pil_image = PILImage.open(images_path / image.dataset_name / image.image_name)
                cropped_image = pil_image.crop(buffer.bounds)

                ## TODO template search this crop!

                cropped_image_path = output_path / f"{l.attributes['ID']}_{Path(image.image_name).stem}_n{len(cropped_annotations)}.jpg"
                cropped_image.save(cropped_image_path)

                cropped_vis_image_path = output_path / f"{l.attributes['ID']}_{Path(image.image_name).stem}_n{len(cropped_annotations)}.png"
                ax = visualise_image(image_path=cropped_image_path, show=False, title="cutout", dpi=75)
                ax = visualise_polygons([c.bbox_polygon for c in cropped_annotations],
                                   max_x=patch_size, max_y=patch_size, color="white", show=False, ax=ax,
                                   filename=cropped_vis_image_path)
                plt.close(ax.figure)
                # plt.show()
                # sleep(1)

                # are other objects in the cutout?
                for il in every_other_label:
                    if il.centroid.within(buffer):
                        covered_objects.append(il)
        print(f"On {image.image_name} are {len(covered_objects)}")


    return covered_objects


def cutout_detection_deduplication(source_image_path: Path,
                                   cutout_polygon: Polygon,
                                   template_labels: list[ImageLabel],
                                   other_images: list[Image],
                                   images_path: Path,
                                   output_path: Path,

                                   ):
    """
    Cutout the detection from the image
    :param template_image:
    :return:
    """

    for large_image in other_images:
        logger.info(f"finding template patch in {large_image.image_name}")

        ipf = ImagePatchFinder(source_image_path,
                               template_polygon=cutout_polygon,
                               large_image_path=images_path / large_image.dataset_name / large_image.image_name)
        if ipf.find_patch():
            # ipf.project_image(output_path=output_path)

            # ipf.footprint


            ax = visualise_image(image_path=images_path / large_image.dataset_name / large_image.image_name, show=False, title=f"{large_image.image_name}", dpi=75)
            ax = visualise_polygons([ipf.footprint], color="white", show=False, ax=ax, linewidth=2.5)
            ax = visualise_polygons([ipf.proj_template_polygon], color="red", show=True, ax=ax, linewidth=2.5, filename=output_path / f"template_proj_to_{large_image.image_name}.jpg")






def detection_deduplication():
    """
    With a couple of detection on multiple images,
    :return:
    """
    hA = hA_from_file(
        file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_07.json"))
    images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")

    image_list = list(images_path.glob("*.JPG"))

    for i in hA.images:
        covered_objects = cutout_detection(image=i, images_path=images_path, output_path=output_path, patch_size=640)

        # TODO have a look at the individual iguanas in annotations, There should N iguanas in there
        # TODO how many are found in the cutouts?

    # take detections from multiple images from BDII dataset

    # cutout the detections with border

    # stack the cutouts

    # analyse the detections


def demo_template():
    """
    take detections, look for those on other images
    :return:
    """
    hA = hA_from_file(
        file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_07.json"))
    images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")
    template_image = hA.images[0]
    other_images = [hA.images[14]] # take a single image which covers perfectly
    other_images = hA.images[14:20]  # take testing subset
    other_images = hA.images  # take al

    p_image = PILImage.open(
        images_path / template_image.dataset_name / template_image.image_name)  # Replace with your image file path

    objs_in_template, template_extents = find_objects(template_image, patch_size=1280)
    templates = crop_objects_from_image(image=p_image, bbox_polygons=template_extents)

    # checking if the cut are right
    #visualise_polygons(polygons=[o.bbox_polygon for o in objs_in_template[0]], show=False)
    #visualise_polygons(polygons=[o for o in template_extents], show=False)
    for i, t in enumerate(templates):
        visualise_image(image=t, show=True, title=f"template {i}", output_file_name=None)

        template_image_path = output_path / f"{template_image.image_name}_template.jpg"
        t.save(template_image_path) # a bit

        cutout_detection_deduplication(
                                       source_image_path=images_path / template_image.dataset_name / template_image.image_name,
                                       cutout_polygon=template_extents[i],
                                       template_labels=objs_in_template[i],
                                       other_images=other_images,
                                       images_path=images_path,
                                       output_path=output_path)


if __name__ == '__main__':
    ## demo code usefull for the method paper
    ## detection_deduplication()

    demo_template()



