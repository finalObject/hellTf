import tensorflow as tf
from mnist import read_data_sets

input_data = read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()

