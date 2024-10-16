"""
Count igunas based on annotations which are in a stack of images
"""
from pathlib import Path
from statistics import mean

from image_template_search.util.HastyAnnotationV2 import hA_from_file

annotation_path = Path("/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/")
annotation_0049_t0 = annotation_path / "template_annotations_DJI_0049.JPG_0.json"
annotation_0049_t1 = annotation_path / "template_annotations_DJI_0049.JPG_1.json"

anno = [annotation_0049_t0, annotation_0049_t1]

for a in anno:
        """ how would I count iguanas in a stack of images with multiple template polygons? """
        hA = hA_from_file(
                file_path=a)

        numbers = ([len(image.labels) for image in hA.images])

        mean_igunas = mean(numbers)
        print(f"Mean iguanas: {mean_igunas}")



"""

['b149971c-6541-4b79-b179-0d6757e9ec44', '3ba79a3b-fae6-4a69-9287-f285737974a9', '51c1c889-e0d3-4964-81c9-08c4b26521e3', 'cbc50407-f0a0-4713-8be5-de176d38e920', '8ececbea-e297-40cc-82b3-b51b34aac395', '4d31e5ce-8f68-4fa8-b05a-d3a1a5b22ed0', 'f9081575-857e-4437-91d6-3ae2cae35619', 'cf0deb9a-b4ab-4120-8004-4a551eb6a0eb', 'aed7fc6e-33d1-4470-8681-1d50b442bee1', '208f4a90-37cf-4264-8608-49a882b22dcc', 'd9673802-1490-49a3-b5d0-e4eda74db14e']


"""