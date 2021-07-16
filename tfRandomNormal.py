#Control
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

sampleSize = tf.TensorShape(100000)
sample = tf.random.normal(sampleSize)

plt.hist(sample.numpy(), bins=1000)
plt.savefig(f"results/{os.path.basename(__file__)}.png")