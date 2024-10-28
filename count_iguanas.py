"""
Count igunas based on annotations which are in a stack of images
"""
from pathlib import Path
from statistics import mean

import attr
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Polygon
from PIL import Image as PILImage

from image_template_search.util.HastyAnnotationV2 import hA_from_file, get_flat_df
from image_template_search.util.util import visualise_image, visualise_polygons

annotation_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")
annotation_0049_t0 = annotation_path / "template_annotations_DJI_0049.JPG_0.json"
annotation_0049_t1 = annotation_path / "template_annotations_DJI_0049.JPG_1.json"

# anno = [annotation_0049_t0, annotation_0049_t1]
anno = annotation_path.glob("template_annotations_DJI*.json")


patch_size = 1280
output_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")

images_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07")
# How many igunanas exist in the ground truth
hA = hA_from_file(
        file_path=Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/labels_2024_10_10.json"))

# hA.images = [i for i in hA.images if i.image_name in ["DJI_0049.JPG",
#                                                       "DJI_0050.JPG", "DJI_0051.JPG",
#                                                       "DJI_0052.JPG",
#                                                       "DJI_0053.JPG",
#                                                       "DJI_0054.JPG", "DJI_0055.JPG",
#                                                       "DJI_0056.JPG",
#                                                       "DJI_0057.JPG", "DJI_0058.JPG", "DJI_0059.JPG",
#                                                       "DJI_0060.JPG",
#                                                       "DJI_0061.JPG",
#                                                       "DJI_0062.JPG",
#                                                       "DJI_0063.JPG",  # First image with ID 7
#                                                       "DJI_0064.JPG",
#                                                       "DJI_0065.JPG",
#                                                       ]]



bd_th = int(patch_size // 2)
all_outer_labels = []
all_inner_labels = []

for i, source_image in enumerate(hA.images):
    center_polygon = Polygon([(0 + bd_th, bd_th), (source_image.width - bd_th, bd_th),
                              (source_image.width - bd_th, source_image.height - bd_th),
                              (bd_th, source_image.height - bd_th)])


    inner_labels = [l for l in source_image.labels if l.centroid.within(center_polygon)]
    inner_labels_IDs = [l.attributes["ID"] for l in inner_labels]
    outer_labels = [l for l in source_image.labels if not l.centroid.within(center_polygon)]
    outer_labels_IDs = [l.attributes["ID"] for l in outer_labels]

    all_inner_labels.extend(inner_labels)

    logger.info(f"Inner labels: {inner_labels_IDs}")
    all_outer_labels.extend(outer_labels)

    p_image = PILImage.open(
        images_path / source_image.dataset_name / source_image.image_name)
    ax_c = visualise_image(image=p_image,
                           show=False, title=source_image.image_name,
                           dpi=75)

    ax_c = visualise_polygons([center_polygon],
                              color="white", show=False, ax=ax_c,
                              linewidth=4.5)
    ax_c = visualise_polygons([c.bbox_polygon for c in outer_labels], labels=outer_labels_IDs,
                              color="green", show=False, ax=ax_c,
                              linewidth=4.5)
    ax_c = visualise_polygons([c.bbox_polygon for c in inner_labels], labels=inner_labels_IDs,
                              color="blue", show=True, ax=ax_c,
                              linewidth=4.5,
                              filename=output_path / f"gt_inner_outer_{source_image.image_name}")
    plt.close(ax_c.figure)

    unique_inner = {a.attributes["ID"] for a in all_inner_labels}

# TODO plot the images with the annotations

# df_hA = get_flat_df(hA)
# logger.info(f"There are {df_hA['ID'].nunique()} objects with the IDs {sorted(df_hA['ID'].unique())} in the ground truth")
logger.info(f"There are {len(unique_inner)} objects with the IDs {sorted(unique_inner)} in the ground truth")


# for a in anno:
#         """ how would I count iguanas in a stack of images with multiple template polygons? """
#         hA = hA_from_file(
#                 file_path=a)
#         df_hA_single = get_flat_df(hA)
#         numbers = ([len(image.labels) for image in hA.images])
#
#         mean_igunas = mean(numbers)
#         # print(f"Mean iguanas: {mean_igunas}")
#         print(f"{hA.images[0].image_name} has {len(hA.images[0].labels)} iguanas")
#         print(f"IDs iguanas: {sorted(df_hA_single['ID'].unique())}")



"""

"""