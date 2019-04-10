import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('datashet.csv').values
N,d = data.shape
x = data[:,d-1].reshape(-1,2)
y = data[:,d].reshape(-1,1)

X = tf.placeholder('float32')
Y = tf.placeholder('float32')

w = tf.Variable('float32', np.array([0.,0.5,1.]))

