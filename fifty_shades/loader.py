from glob import glob
from os.path import realpath, dirname, join
from typing import Iterator

import tensorflow as tf

from fifty_shades.image_processing import load_image_tensor


class Style:
    HERO = "hero"
    DRAWING = "drawing"


def generate_cat_pictures() -> Iterator[tf.Tensor]:
    return generate_pictures("cat")


def generate_style_pictures(style_name: str) -> Iterator[tf.Tensor]:
    return generate_pictures(style_name)


def generate_pictures(directory_name: str) -> Iterator[tf.Tensor]:
    for file_path in glob(join(get_data_path(), directory_name, "*")):
        image = load_image_tensor(file_path)
        yield image


def get_result_path() -> str:
    return join(get_data_path(), "result")


def get_data_path() -> str:
    return join(realpath(dirname(__file__)), "..", "data")


def get_hero_one_file_path() -> str:
    return join(get_data_path(), Style.HERO, "1.jpg")


def get_drawing_one_file_path() -> str:
    return join(get_data_path(), Style.DRAWING, "1.jpg")


def get_cat_bromance_file_path() -> str:
    return join(get_data_path(), "cat", "bromance.jpg")
