import numpy as np
import tensorflow as tf
from tensorflow import keras
#@tf.keras.saving.register_keras_serializable(package="ImCapModel")
class CpaModel(tf.keras.Model):
    def __init__(self, decoder, embeddings, dosage_encoder, **kwargs):
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


            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            with tf.GradientTape() as tape:
                #probs = self(cell)
                loss = self.loss_function(probs, decoder_labels, mask)

            ## Perform a backwards pass
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def test(self,  batch_size=30):
        """
        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc


    def loss_function(prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """
        masked_labs = tf.boolean_mask(labels, mask)
        masked_prbs = tf.boolean_mask(prbs, mask)
        scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
        loss = tf.reduce_sum(scce)
        return loss