# Need the package: tf_explain which from: https://github.com/sicara/tf-explain

import tensorflow as tf
from keras.models import load_model
from tf_explain.core import GradCAM
import numpy as np

IMAGE_PATH = './test/surprise/PrivateTest_85402036.jpg'
model = load_model('./saved_model/model_optimal.h5')

# Load a sample image (or multiple ones)
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(48, 48), color_mode='grayscale')
img = tf.keras.preprocessing.image.img_to_array(img)
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
data = ([img], label_dict[2])

# Start explainer
explainer = GradCAM()
#chose the layer of the model and visualize the image, generating heatmap
grid = explainer.explain(data, model, layer_name='max_pooling2d_3', class_index=3)  # 281 is the tabby cat index in ImageNet

explainer.save(grid, ".", "surprise4.png")
