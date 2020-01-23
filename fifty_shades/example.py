from os.path import join
from os.path import realpath, dirname

from fifty_shades.loader import generate_image_tensors
from fifty_shades.model import NeuralStyleTransfer


data_path = join(realpath(dirname(__file__)), "..", "data")
cat_directory_path = join(data_path, "canvas_cat")
style_directory_path = join(data_path, "canvas_hero")
save_directory = join(data_path, "result", "canvas")

cat_generator = generate_image_tensors(cat_directory_path)
style_generator = generate_image_tensors(style_directory_path)
model = NeuralStyleTransfer()
model.predict_and_save_all(cat_generator, style_generator, save_directory)
