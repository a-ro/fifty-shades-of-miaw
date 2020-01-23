from glob import glob
from os.path import join
from typing import Iterator

import tensorflow as tf

from fifty_shades.image_processing import load_image_tensor


def generate_image_tensors(directory_path: str) -> Iterator[tf.Tensor]:
    file_path_regex = join(directory_path, "*")
    for file_path in sorted(glob(file_path_regex)):
        image_tensor = load_image_tensor(file_path)
        yield image_tensor
