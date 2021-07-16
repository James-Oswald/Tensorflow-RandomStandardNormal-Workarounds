
# Tensorflow RandomStandardNormal Workarounds

As of 7/16/21, [tfjs](https://github.com/tensorflow/tfjs) does not support the RandomStandardNormal Op (see [https://github.com/tensorflow/tfjs/issues/4156](https://github.com/tensorflow/tfjs/issues/4156)) and hence one cant use the tf.random.normal in models meant to be run in browser. 

As I am far too lazy to attempt to implement the op myself in tfjs, I experiment here with creating methods for python models that reasonably can reproduce tf.random.normal using only tf.random.uniform (represented as the RandomUniform OP in tfjs and thus is leagal for tfjs models)