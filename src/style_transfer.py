import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# Load Image Function
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    return np.expand_dims(img, axis=0) / 255.0

# Build NST Model using VGG19
def build_nst_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
    outputs = [vgg.get_layer(layer).output for layer in layers]
    return Model(inputs=vgg.input, outputs=outputs)

# Apply NST to an image
def apply_style(content_img, style_img):
    model = build_nst_model()
    content_features = model.predict(content_img)
    style_features = model.predict(style_img)
    
    stylized_img = (content_features[0] + style_features[0]) / 2
    return (stylized_img * 255).astype(np.uint8)
