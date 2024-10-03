from pathlib import Path
import hashlib

import numpy as np


def get_image_id(filename: Path = None, image: np.ndarray = None):
    """
    @ has moved to flight_image_capturing_sim/helper/image.py
    generate an id from the image itself which can be used to find images which are exactly the same
    @param filename:
    @return:
    """

    if filename is not None:
        with open(filename, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash
    if image is not None:
        image_bytes = image.tobytes()
        readable_hash = hashlib.sha256(image_bytes).hexdigest()
        return readable_hash
