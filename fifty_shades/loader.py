from glob import glob
from os.path import realpath, dirname, join
from typing import Iterator

import tensorflow as tf

from fifty_shades.image_processing import load_image_tensor


class ImageType:
    HERO = "hero"
    DRAWING = "drawing"
    CAT = "cat"
    CANVAS_CAT = "canvas_cat"
    CANVAS_HERO = "canvas_hero"


def generate_project_images(directory_name: str) -> Iterator[tf.Tensor]:
    return generate_image_tensors(join(get_data_path(), directory_name))


def generate_image_tensors(directory_path: str) -> Iterator[tf.Tensor]:
    file_path_regex = join(directory_path, "*")
    for file_path in sorted(glob(file_path_regex)):
        image_tensor = load_image_tensor(file_path)
        yield image_tensor


def get_result_path() -> str:
    return join(get_data_path(), "result")


def get_data_path() -> str:
    return join(realpath(dirname(__file__)), "..", "data")


def get_hero_one_file_path() -> str:
    return join(get_data_path(), ImageType.HERO, "1.jpg")


def get_drawing_one_file_path() -> str:
    return join(get_data_path(), ImageType.DRAWING, "1.jpg")


def get_cat_bromance_file_path() -> str:
    return join(get_data_path(), "cat", "bromance.jpg")
