import tensorflow as tf
from keras import layers

class CPAEncoder(layers.Layer):
    def __init__(self, input_dim, latent_size=256, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_size, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(hidden_size, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(hidden_size, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(latent_size, activation=None)
        ])
        
    def call(self, x):
        return self.encoder(x)

