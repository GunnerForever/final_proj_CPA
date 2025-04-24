import tensorflow as tf
from keras import Model, layers

class CPADiscriminator(Model):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int=128, **kwargs):
        super().__init__(**kwargs)
        self.discrimator = tf.keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),
            layers.Dense(hidden_size, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        ])

    def call(self, z_basal):
        return self.discrimator(z_basal)