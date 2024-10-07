import copy
from pathlib import Path
from time import sleep

from matplotlib import pyplot as plt
from shapely import Point
from shapely.affinity import affine_transform

from image_template_search.util.HastyAnnotationV2 import hA_from_file, Image
from PIL import Image as PILImage

from image_template_search.util.util import visualise_polygons, visualise_image


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

    if image.image_name == "DJI_0077.JPG":
        covered_objects = []

        for l in image.labels:

            # if True or ( "ID" in l.attributes and l.attributes["ID"] == "2" ): # TODO remove this I just want to test it with ID 12
            if l not in covered_objects:
                covered_objects.append(l) # the current object is covered
                every_other_label = [il for il in image.labels if il not in covered_objects]

                # create a buffer around the centroid of the polygon
                pc = l.bbox_polygon.centroid
                buffer = pc.buffer(patch_size // 2)
                minx, miny, maxx, maxy = buffer.bounds


                obj_in_crop = [copy.copy(il) for il in image.labels if il.centroid.within(buffer)]  # all objects withing the buffer

                a, b, d, e = 1.0, 0.0, 0.0, 1.0  # Scale and rotate
                xoff, yoff = -minx, -miny  # Translation offsets

                # Apply the affine transformation to the polygon to reproject into image coordinates
                transformation_matrix = [a, b, d, e, xoff, yoff]

                cropped_annotations = [l for l in obj_in_crop if buffer.contains(l.centroid)]
                # cropped_annotations = [(l.centroid.x - minx, l.centroid.y - miny) for l in cropped_annotations]

                for ca in cropped_annotations:
                    ca.bbox_polygon = affine_transform(ca.bbox_polygon, transformation_matrix)

                pil_image = PILImage.open(images_path / image.dataset_name / image.image_name)
                cropped_image = pil_image.crop(buffer.bounds)
                cropped_image_path = output_path / f"{l.attributes['ID']}_{Path(image.image_name).stem}_n{len(cropped_annotations)}.jpg"
                cropped_image.save(cropped_image_path)


                cropped_vis_image_path = output_path / f"{l.attributes['ID']}_{Path(image.image_name).stem}_n{len(cropped_annotations)}.png"
                ax = visualise_image(image_path=cropped_image_path, show=False, title="cutout", dpi=75)
                visualise_polygons([c.bbox_polygon for c in cropped_annotations],
                                   max_x=patch_size, max_y=patch_size, color="white", show=False, ax=ax, filename=cropped_vis_image_path)

                # plt.show()
                # sleep(1)

                # are other objects in the cutout?
                for il in every_other_label:
                    if il.centroid.within(buffer):
                        covered_objects.append(il)

        return covered_objects


def detection_deduplication():
    """
    With a couple of detection on multiple images,
    :return:
    """
    hA = hA_from_file(file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_07.json"))
    images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
    output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")

    image_list = list(images_path.glob("*.JPG"))

    for i in hA.images:
        cutout_detection(image=i, images_path=images_path, output_path=output_path, patch_size=640)


    # take detections from multiple images from BDII dataset

    # cutout the detections with border

    # stack the cutouts

    # analyse the detections


if __name__ == '__main__':
    detection_deduplication()