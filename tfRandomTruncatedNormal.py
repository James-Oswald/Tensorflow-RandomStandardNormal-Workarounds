
#Investigating truncated normal

import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

sampleSize = tf.TensorShape(1000000)
sample = tf.random.truncated_normal(sampleSize)

plt.hist(sample.numpy(), bins=1000)
#plt.yscale('log')
plt.savefig(f"results/{os.path.basename(__file__)}.png")