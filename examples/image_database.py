"""
read a folder of images and parse the results into a datatype
"""
import json
import typing
from pathlib import Path

import pandas as pd

from image_template_search.util.georeferenced_image import ExtendImageMetaData
from image_template_search.util.util import get_exif_metadata

metadata_folder = Path("./")

folder = Path("/Users/christian/data/2TB/ai-core/data/02.02.21/FMO03")
folder_2 = Path("/Users/christian/data/2TB/ai-core/data/02.02.21/FMO02")
folder_3 = Path("/Users/christian/data/2TB/ai-core/data/03.02.21/FMO04")

folders = [folder, folder_2, folder_3]

def get_mission_metadata(folder: Path) -> typing.List[ExtendImageMetaData]:
    metadata = []

    for i in  folder.glob("*.JPG"):
        image_meta_data = get_exif_metadata(i)
        metadata.append(image_meta_data)
        # image_meta_data.model_dump()
    return metadata


for f in folders:
    md = get_mission_metadata(f)

    metadata_file = f / "metadata.jsonl"

    with open(metadata_file, 'w') as f:
        for m in md:
            f.write(m.model_dump_json() + "\n")

    with open(metadata_file, 'r') as f:
        metadata_entries = [json.loads(line) for line in f]
        metadata_entries = [ExtendImageMetaData(**m) for m in metadata_entries]


df_metadata = pd.DataFrame([m.model_dump() for m in metadata_entries])

df_metadata


