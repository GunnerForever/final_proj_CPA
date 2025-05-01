import tensorflow as tf
from grad_reverse import GradientReversalLayer

x = tf.constant([[1.0]])
grl = GradientReversalLayer(lambda_=1.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = grl(x)
grad = tape.gradient(y, x)
print(grad.numpy())  # should be -1.0