from os.path import join

import tensorflow as tf
from keras_preprocessing.image import array_to_img


def load_image_tensor(image_path: str, max_dim: int = 512) -> tf.Tensor:
    image_tensor = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_image(
        image_tensor, channels=3, dtype=tf.float32
    )
    image_tensor = tf.image.resize(
        image_tensor, (max_dim, max_dim), preserve_aspect_ratio=True
    )
    image_tensor = image_tensor[tf.newaxis, :]
    return image_tensor


def save_image_from_tensor(
    save_directory: str, file_name: str, tensor_image: tf.Tensor
) -> None:
    save_path = join(save_directory, file_name)
    image = array_to_img(tensor_image[0])
    image.save(save_path)
