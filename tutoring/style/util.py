from PIL import Image
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def open_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.float32)
    reds = np.copy(image_array[:, :, 0])
    blues = image_array[:, :, 2]
    image_array[:, :, 0] = blues
    image_array[:, :, 2] = reds
    image_array[:, :, 0] -= _B_MEAN
    image_array[:, :, 1] -= _G_MEAN
    image_array[:, :, 2] -= _R_MEAN
    return image_array

def save_image(image_data, image_path):
    image_data = np.copy(image_data)
    image_data[:, :, 0] += _B_MEAN
    image_data[:, :, 1] += _G_MEAN
    image_data[:, :, 2] += _R_MEAN
    reds = np.copy(image_data[:, :, 0])
    blues = image_data[:, :, 2]
    image_data[:, :, 0] = blues
    image_data[:, :, 2] = reds
    image = Image.fromarray(np.uint8(np.clip(image_data, 0, 255)))
    image.save(image_path)

image_array = open_image('./images/Tuebingen.jpg')
save_image(image_array, './images/Tuebingen_saved.jpg')
