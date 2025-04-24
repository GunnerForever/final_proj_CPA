import tensorflow as tf
from keras import layers, regularizers, Model

class CPADecoder(Model):
    def __init__(self, input_dim, output_dim, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),
            # layers.Dense(hidden_size, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-6)),
            # layers.BatchNormalization(),
            # layers.ReLU(),
            layers.Dense(hidden_size, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-6)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(hidden_size, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-6)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(output_dim)
        ])
        
    def call(self, x):
        return self.decoder(x)