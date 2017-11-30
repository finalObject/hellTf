import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def showNum(image):
	newImage=image.reshape(28,28)
	plt.imshow(newImage,cmap='Greys_r')
	plt.show()
	return
def testImage(sess,data,W,b):
	x=tf.placeholder(tf.float32,[None,784])
	op=tf.matmul(x,W)+b;
	data=data.reshape(1,784)
	print(sess.run(op,feed_dict={x:data}))
	return
