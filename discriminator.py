import tensorflow as tf
from keras import layers

class CPADiscriminator(layers.Layer):
    def __init__(self, input_dim, num_classes, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.discrimator = tf.keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),
            layers.Dense(hidden_size, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, z_basal):
        return self.discrimator(z_basal)