
#Uses the Box Muller transform to generate a normally distributed sample from a uniformly distributed sample
#https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

sampleSize = tf.TensorShape(100000000)
u1 = tf.random.uniform(sampleSize, dtype=tf.float32)
u2 = tf.random.uniform(sampleSize, dtype=tf.float32)
z0 = tf.sqrt(-2*tf.math.log(u1))*tf.cos(2*math.pi*u2)

plt.hist(z0.numpy(), bins=1000)
plt.yscale('log')
plt.savefig(f"results/{os.path.basename(__file__)}.png")