from PIL import Image
import numpy as np
def open_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    reds = np.copy(image_array[:, :, 0])
    blues = image_array[:, :, 2]
    image_array[:, :, 0] = blues
    image_array[:, :, 2] = reds
    return image_array

def save_image(image_data, image_path):
    image_data = np.copy(image_data)
    reds = np.copy(image_data[:, :, 0])
    blues = image_data[:, :, 2]
    image_data[:, :, 0] = blues
    image_data[:, :, 2] = reds
    image = Image.fromarray(np.uint8(image_data))
    image.save(image_path)

image_array = open_image('./images/Tuebingen.jpg')
save_image(image_array, './images/Tuebingen_saved.jpg')
