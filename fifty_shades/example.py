from os.path import join

from fifty_shades.image_processing import load_image_tensor
from fifty_shades.loader import (
    ImageType,
    generate_project_images,
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


def transform_all_cat_images(style_name: str, save_directory: str) -> None:
    cat_generator = generate_project_images(ImageType.CAT)
    style_generator = generate_project_images(style_name)
    model = NeuralStyleTransfer()
    model.predict_and_save_all(cat_generator, style_generator, save_directory)


def transform_all_cat_images_with_single_style(style_image_path: str, save_directory: str) -> None:
    cat_generator = generate_project_images(ImageType.CAT)
    style_tensor = load_image_tensor(style_image_path)
    model = NeuralStyleTransfer()
    model.predict_single_style_and_save_all(cat_generator, style_tensor, save_directory)


def transform_single_cat_image_with_all_styles(cat_image_path: str, style_name: str, save_directory: str) -> None:
    cat_tensor = load_image_tensor(cat_image_path)
    style_generator = generate_project_images(style_name)
    model = NeuralStyleTransfer()
    model.predict_single_content_and_save_all(cat_tensor, style_generator, save_directory)


def transform_canvas_images(save_directory: str) -> None:
    cat_generator = generate_project_images(ImageType.CANVAS_CAT)
    style_generator = generate_project_images(ImageType.CANVAS_HERO)
    model = NeuralStyleTransfer()
    model.predict_and_save_all(cat_generator, style_generator, save_directory)


if __name__ == "__main__":
    show_stylized_cat_example()

    transform_all_cat_images(ImageType.HERO, save_directory=join(get_result_path(), "all_heroes"))
    transform_all_cat_images(ImageType.DRAWING, save_directory=join(get_result_path(), "all_drawings"))

    transform_all_cat_images_with_single_style(
        get_hero_one_file_path(), save_directory=join(get_result_path(), "single_hero")
    )
    transform_all_cat_images_with_single_style(
        get_drawing_one_file_path(), save_directory=join(get_result_path(), "single_drawing")
    )

    transform_single_cat_image_with_all_styles(
        get_cat_bromance_file_path(), ImageType.HERO, save_directory=join(get_result_path(), "bromance_heroes")
    )
    transform_single_cat_image_with_all_styles(
        get_cat_bromance_file_path(), ImageType.DRAWING, save_directory=join(get_result_path(), "bromance_drawings")
    )

    transform_canvas_images(save_directory=join(get_result_path(), "canvas"))
