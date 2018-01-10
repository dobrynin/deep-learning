from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from util import save_image, open_image
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape
import keras.backend as K
from keras.optimizers import SGD

def build_style_tensor(content_tensor):
    height = content_tensor.shape[1].value
    width = content_tensor.shape[2].value
    num_channels = content_tensor.shape[3].value
    permuted_content = K.permute_dimensions(content_tensor, (0, 3, 1, 2))
    flattened_channels = K.reshape(permuted_content, (-1, num_channels, height*width))
    flattened_channels_t = K.permute_dimensions(flattened_channels, (0, 2, 1))
    return K.batch_dot(flattened_channels, flattened_channels_t) / (height * width * num_channels)

content_image_array = open_image('./images/Tuebingen.jpg')
content_image_array = np.expand_dims(content_image_array, axis=0)
style_image_array = open_image('./images/starry_night.jpg')
style_image_array = np.expand_dims(style_image_array, axis=0)


vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(768, 1024, 3), pooling='avg')


block1_conv2 = vgg_model.get_layer('block1_conv2')
block2_conv2 = vgg_model.get_layer('block2_conv2')
block3_conv2 = vgg_model.get_layer('block3_conv2')
block4_conv2 = vgg_model.get_layer('block4_conv2')
block5_conv2 = vgg_model.get_layer('block5_conv2')

content_output_tensor = block5_conv2.output

style_layer = Lambda(build_style_tensor)
block1_conv2_style_tensor = style_layer(block1_conv2.output)
block2_conv2_style_tensor = style_layer(block2_conv2.output)
block3_conv2_style_tensor = style_layer(block3_conv2.output)
block4_conv2_style_tensor = style_layer(block4_conv2.output)
block5_conv2_style_tensor = style_layer(block5_conv2.output)


model = Model(inputs=vgg_model.input, output=[
    content_output_tensor,
    block1_conv2_style_tensor,
    block2_conv2_style_tensor,
    block3_conv2_style_tensor,
    block4_conv2_style_tensor,
    block5_conv2_style_tensor])

content_matrix, *_ = model.predict(content_image_array)
_, *style_matrices = model.predict(style_image_array)

for layer in vgg_model.layers:
    layer.trainable = False

input_tensor = Input(shape=(1,))
dense_layer = Dense(1024*768*3, activation='linear', use_bias=False)
dense_tensor = dense_layer(input_tensor)
reshape_layer = Reshape((768, 1024, 3))
reshaped_tensor = reshape_layer(dense_tensor)
final_content, *final_styles = model(reshaped_tensor)

styled_image_model = Model(inputs=input_tensor, output=[
    final_content,
    *final_styles
])

sgd = SGD(lr=0.01)
styled_image_model.compile(optimizer=sgd, loss='mse')
styled_image_model.summary()
styled_image_model.fit(numpy.ones((1,1)), (content_matrix, *style_matrices))
