from glob import glob
from os.path import realpath, dirname, join
from typing import Iterator

from fifty_shades.image_processing import load_image_tensor


def generate_cat_pictures() -> Iterator:
    return generate_pictures("cat")


def generate_style_pictures(style_name: str) -> Iterator:
    return generate_pictures(style_name)


def generate_pictures(directory_name: str) -> Iterator:
    for file_path in glob(join(get_data_path(), directory_name, "*")):
        image = load_image_tensor(file_path)
        yield image


def get_result_path() -> str:
    return join(get_data_path(), "result")


def get_data_path() -> str:
    return join(realpath(dirname(__file__)), "..", "data")
