
#Uses the sum of uniform distributions to resonably model a normal, see 
#https://stats.stackexchange.com/a/16411/312105

import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

sampleSize = tf.TensorShape(1000000)
numUniforms = 12  #The number of uniforms to sum
uniformSum = tf.zeros(sampleSize)
for _ in range(numUniforms):
    uniformSum += tf.random.uniform(sampleSize, dtype=tf.float32)
uniformSum = uniformSum - (numUniforms/2)

plt.hist(uniformSum.numpy(), bins=1000)
#plt.yscale('log')
plt.savefig(f"results/{os.path.basename(__file__)}.png")