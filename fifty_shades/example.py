from os.path import join

from fifty_shades.image_processing import load_image_tensor
from fifty_shades.loader import (
    Style,
    generate_style_pictures,
    generate_cat_pictures,
    get_result_path,
    get_hero_one_file_path,
    get_drawing_one_file_path,
    get_cat_bromance_file_path,
)
from fifty_shades.model import NeuralStyleTransfer


def show_stylized_cat_example():
    cat_tensor = load_image_tensor(get_cat_bromance_file_path())
    style_tensor = load_image_tensor(get_drawing_one_file_path())
    model = NeuralStyleTransfer()
    predicted_image = model.predict(cat_tensor, style_tensor)
    predicted_image.show()


def transform_all_cat_pictures(style_name: str, save_directory: str) -> None:
    cat_generator = generate_cat_pictures()
    style_generator = generate_style_pictures(style_name)
    model = NeuralStyleTransfer()
    model.predict_and_save_all(cat_generator, style_generator, save_directory)


def transform_all_cat_pictures_with_single_style(style_image_path: str, save_directory: str) -> None:
    cat_generator = generate_cat_pictures()
    style_tensor = load_image_tensor(style_image_path)
    model = NeuralStyleTransfer()
    model.predict_single_style_and_save_all(cat_generator, style_tensor, save_directory)


def transform_single_cat_picture_with_all_styles(cat_image_path: str, style_name: str, save_directory: str) -> None:
    cat_tensor = load_image_tensor(cat_image_path)
    style_generator = generate_style_pictures(style_name)
    model = NeuralStyleTransfer()
    model.predict_single_content_and_save_all(cat_tensor, style_generator, save_directory)


if __name__ == "__main__":
    show_stylized_cat_example()

    transform_all_cat_pictures(Style.HERO, save_directory=join(get_result_path(), "all_heroes"))
    transform_all_cat_pictures(Style.DRAWING, save_directory=join(get_result_path(), "all_drawings"))

    transform_all_cat_pictures_with_single_style(
        get_hero_one_file_path(), save_directory=join(get_result_path(), "single_hero")
    )
    transform_all_cat_pictures_with_single_style(
        get_drawing_one_file_path(), save_directory=join(get_result_path(), "single_drawing")
    )

    transform_single_cat_picture_with_all_styles(
        get_cat_bromance_file_path(), Style.HERO, save_directory=join(get_result_path(), "bromance_heroes")
    )
    transform_single_cat_picture_with_all_styles(
        get_cat_bromance_file_path(), Style.DRAWING, save_directory=join(get_result_path(), "bromance_drawings")
    )
