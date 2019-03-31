from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sess = tf.Session()
w = tf.Variable('w', dtype='float32')
w = w.reshape(-1,1)
#w1 = tf.Variable('w1', dtype='float32')

init = tf.initialize_all_variables()
sess.run(init)

X = tf.placeholder(dtype='float32', tf.shape(-1,1))
y = tf.placeholder(dtype='float32')

linear_model = tf.layers.Dense(units=1)
y = linear_model(X)