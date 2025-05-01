import tensorflow as tf
from keras import layers, regularizers, Model

class CPAEncoder(Model):
    def __init__(self, input_dim: int, latent_size: int=256, hidden_size: int=512, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),
            layers.Dense(hidden_size, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5)),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.Dense(hidden_size // 2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5)),
            layers.LayerNormalization(),
            layers.ReLU(),
            # layers.Dense(hidden_size//2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-5)),
            # layers.LayerNormalization(),
            # layers.ReLU(),
            layers.Dense(latent_size)
        ])
        
    def call(self, x):
        return self.encoder(x)

