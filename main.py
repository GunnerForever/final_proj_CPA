from train import train, test
import tensorflow as tf
from model import CPAModel
import numpy as np
from keras import optimizers
import os

from preprocess_kang import load_kang, prepare_kang


def main(dataset_name="kang"):
    assert dataset_name in ["kang", "norman"], "Dataset not supported"

    input_dim = 5000                        # Number of genes
    latent_size = 128                       # Embedding size
    output_dim = 5000                       # Number of genes  
    enc_dec_hidden_size = 128               # Hidden size of encoder and decoder
    dosage_enc_hidden_size = 64             # Hidden size of dosage encoder
    discriminator_hidden_size = 128         # Hidden size of discriminators
    num_perts = 2                           # Number of unique perturbations
    use_covariates = True                   # Whether there are covariates
    num_covs = 8                            # Number of covariates

    batch_size = 128             # Batch size
    epochs = 300                 # Number of epochs

    if dataset_name == "kang":
        kang = load_kang("../1470_script/final_project_data/kang_normalized_hvg.h5ad")
        train_kang, val_kang, ood_kang = prepare_kang(kang, batch_size=batch_size, split_col="split_CD14 Mono")

    model = CPAModel(input_dim, latent_size, output_dim, enc_dec_hidden_size,
                     dosage_enc_hidden_size, discriminator_hidden_size, num_perts, 
                     use_covariates, num_covs)
    
    ae_optimizer = optimizers.Adam(1e-3)
    dose_optimizer = optimizers.Adam(4e-3)
    adv_optimizer = optimizers.Adam(3e-4)

    train(model, 
          train_kang, val_kang, 
          ae_optimizer, dose_optimizer, adv_optimizer, 
          epochs=epochs, batch_size=batch_size,
          n_discriminator_steps=3)
    # ood_avg_loss, ood_avg_r2 = test(model, ood_kang)

    # print(f"Out-of-distribution average loss: {ood_avg_loss}, R2 score: {ood_avg_r2}")


if __name__ == '__main__':
    # datasets available: "kang", "norman", 

    main("kang")
    # main("norman")