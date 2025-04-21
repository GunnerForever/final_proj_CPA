# Theoretically, I think we should split the model into half, each half being embeddings and decoding respectively, to make the "swap out perturbations" testing simpler
    #Embedding.py does the encoder half of the model, decoder is the decoding
# Alternatively we just make one CPA model file

import tensorflow as tf
import keras

class CpaEmbeddings(keras.layers.Layer):

    def __init__(self, vars_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.vars_size = vars_size
        self.hidden_size = hidden_size
        self.cell_encoder = tf.keras.layers.Dense(self.hidden_size, activation=None)
        self.perts_encoder = tf.keras.layers.Dense(self.hidden_size, activation=None)
        self.dosage_encoder = tf.keras.layers.Dense(self.hidden_size, activation=None)
        self.covs_encoder = tf.keras.layers.Dense(self.hidden_size, activation=None)
        

        
    def call(self, cell_state, perts, covs):
        encoded_cell = self.cell_encoder(cell_state)
        encoded_perts = self.perts_encoder(self.dosage_encoder(perts))
        encoded_covs = self.covs_encoder(covs)

        return encoded_cell, encoded_perts, encoded_covs 

