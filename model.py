import numpy as np
import tensorflow as tf
from tensorflow import keras
#@tf.keras.saving.register_keras_serializable(package="ImCapModel")
class CpaModel(tf.keras.Model):
    def __init__(self, decoder, embeddings, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self.decoder = decoder
        
    def call(self, cell_state, perts, covs):
        #Get Z_basal, apply V_perturbations and V_covariates
        basal_state, pert_embed, cov_embed = self.embeddings.call(cell_state, perts, covs)

        #Put them through the "composition" part of the autoencoder
        composite = self.composition(basal_state, pert_embed, cov_embed)

        #Decode
        return self.decoder(composite)  
    
    def call(self, cell_state, perts, covs, new_perts): #Function overloading for test case
        #Get Z_basal, apply V_perturbations and V_covariates
        basal_state, pert_embed, cov_embed = self.embeddings.call(cell_state, perts, covs)

        #Put them through the "composition" part of the autoencoder
        composite = self.composition(basal_state, pert_embed, new_perts)

        #Decode
        return self.decoder(composite)  
    
    def composition(self, basal_state, pert_embed, cov_embed):
        #I thought this was going to be more complex than it actually ended up being
        composite = basal_state + pert_embed + cov_embed
        return composite

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, data, batch_size=10):
        for index, end in enumerate(range(batch_size, len(data)+1, batch_size)):
            ## Get batches of data, convert into (cell state, perturbations, covariates) triplets
            cell_state, perts, covs = (-1, -1, -1)

            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            with tf.GradientTape() as tape:
                decoded_z = self(cell_state, perts, covs)
                loss = self.loss_function(decoded_z)

            ## Perform a backwards pass
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def test(self, data, new_pertubations, batch_size=30):
        #num_batches = int(len(test_captions) / batch_size)

        total_loss = 0
        for index, end in enumerate(range(batch_size, len(data)+1, batch_size)):
            ## Get batches of data, convert into (cell state, perturbations, covariates) triplets and also get the extra pertubations data
            cell_state, perts, covs = (-1, -1, -1)

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            decoded_z = self(cell_state, perts, covs, new_pertubations)
            #num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            #loss = self.loss_function(probs, decoder_labels, mask)
            #accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            avg_acc = -1
        return avg_acc


    def loss_function(self, decoded_z):
        loss = -1
        return loss