import tensorflow as tf

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, inputs):
        @tf.custom_gradient

        def reverse_gradient(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad

        return reverse_gradient(inputs)