import tensorflow as tf
import keras

class CpaDecoder(keras.layers.Layer):

    def __init__(self, out_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.decoder = tf.keras.layers.Dense(self.hidden_size, activation=None)
        

        
    def call(self, composite):
        #encoded_images = self.mlp1(encoded_images)
        decoded_cell = self.mlp2(composite)
        return decoded_cell