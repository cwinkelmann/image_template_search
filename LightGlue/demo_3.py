from pathlib import Path

from matplotlib import pyplot as plt

from image_template_search.image_similarity import TiledExtractor
# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, Extractor
from lightglue import viz2d
import torch
from lightglue import SIFT
from PIL import Image
import numpy as np







if __name__ == '__main__':
    torch.set_grad_enabled(False)
    images = Path("assets")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    N_x, N_y = 2, 2
    small_image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/cutouts/template_source_DJI_0049.JPG__a02b70954a14ab0a16eb97ff187e44ecc9c0e8f0437a1b33ce56e5e0cce1f413__1280.jpg"
    image_path = "/Users/christian/data/2TB/ai-core/data/detection_deduplication/images_2024_10_07/single_images/DJI_0075.JPG"

    e = SIFT(max_num_keypoints=6096).eval().to(device)  # load the extractor
    extractor = TiledExtractor(extractor=e)
    # TODO bundle this into patched extractor

    feats1 = extractor.extract(image_path = image_path, N_x=N_x, N_y=N_y)

    # extracting the features the LightGlue Default way
    e = SIFT(max_num_keypoints=6096).eval().to(device)  # load the extractor
    image0 = load_image(small_image_path)
    feats0 = e.extract(image0.to(device))



    matcher = LightGlue(features="sift").eval().to(device)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    ## Display the matches
    image1 = load_image(image_path)
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)


    plt.show()