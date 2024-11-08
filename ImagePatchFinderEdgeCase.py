from pathlib import Path

from shapely.geometry.polygon import Polygon

from image_template_search.image_similarity import ImagePatchFinder
from PIL import Image as PILImage

from image_template_search.util.util import visualise_image

# template_image_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/output/template_DJI_0066.JPG__c4a9f1f00e5e65ec2f407c1a704398252555c2f337d24bed38040fcd35c9cd4d__1280.jpg")
# cropped_destination_image_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/output/cropped_mosaic_100_DJI_0066.jpg")


template_image_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0066.JPG")
cropped_destination_image_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/mosaics/mosaic_100.jpg")

base_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/")
output_path = base_path / "output"

with PILImage.open(template_image_path) as img:
    source_image_width, source_image_height = img.size
    template_extent = Polygon(
        [(0, 0), (source_image_width, 0), (source_image_width, source_image_height), (0, source_image_height)])


ipf_t = ImagePatchFinder(template_path=template_image_path,
                         template_polygon=template_extent,
                         large_image_path=cropped_destination_image_path)

template_match = ipf_t.find_patch(similarity_threshold=0.000005)

ipf_t.project_image(output_path)

