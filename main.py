from train import train, test
import tensorflow as tf
from model import CPAModel
import numpy as np
from keras import optimizers
import os

def main():
    LOCAL_TRAIN_FILE = 'data/train'
    LOCAL_TEST_FILE = 'data/test'

    # TODO: preprocess data and load data

    input_dim = 5000                        # Number of genes
    latent_size = 256                       # Embedding size
    output_dim = 5000                       # Number of genes  
    enc_doc_hidden_size = 512               # Hidden size of encoder and decoder
    dosage_enc_hidden_size = 64             # Hidden size of dosage encoder
    discriminator_hidden_size = 128         # Hidden size of discriminators
    num_perts = 10                          # Number of unique perturbations
    use_covariates = True                   # Whether there are covariates
    num_covs = 5                            # Number of covariates

    batch_size = 128             # Batch size
    learning_rate = 3e-3         # Learning rate (subject to change)
    epochs = 20                  # Number of epochs

    model = CPAModel(input_dim, latent_size, output_dim, enc_doc_hidden_size,
                     dosage_enc_hidden_size, discriminator_hidden_size, num_perts, 
                     use_covariates, num_covs)
    
    optimizer = optimizers.Adam(learning_rate)

    train(model, ..., ..., optimizer, epochs=epochs)


if __name__ == '__main__':
    main()