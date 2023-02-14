import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from nst import neural_style_transfer
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Program to stylize an image using Neural Style Transfer')

parser.add_argument('--content_image',dest='content_image',required=True,help='Path of the image to be stylized')
parser.add_argument('--style_image',dest='style_image',required=True,help='Path of the style image')
parser.add_argument('--dest_folder',dest='dest_folder',required=True,help='Path to store stylized image')

def load_img(path_to_img):
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image

model = neural_style_transfer()

args = parser.parse_args()
content_image = load_img(args.content_image)
style_image = load_img(args.style_image)
dest_path = args.dest_folder

style_weight =  0.1
content_weight = 1e-32 

print("Applying Neural Style transfer to {} using Mobilenet_v2 model using Adam optimizer and looping for 50 epochs".format(args.content_image))

adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=40.0, decay_steps=100, decay_rate=0.80
    )
)

stylized_image, display_images = model.fit_style_transfer(style_image=style_image, content_image=content_image, 
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    optimizer=adam, epochs=50, steps_per_epoch=1)

cv2.imwrite(dest_path,np.array(stylized_image[0]))

print("Stylized images stored at {}".format(dest_path))