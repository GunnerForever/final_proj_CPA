import numpy as np
import tensorflow as tf
from keras import layers
from encoder import CPAEncoder
from decoder import CPADecoder
from grad_reverse import GradientReversalLayer
from discriminator import CPADiscriminator

class CPAModel(tf.keras.Model):
    def __init__(self, input_dim, latent_size, output_dim, enc_doc_hidden_size, dosage_enc_hidden_size,
                 discriminator_hidden_size, num_perts, use_covariates=False, num_covs=0, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = CPAEncoder(input_dim, latent_size, enc_doc_hidden_size)
        self.decoder = CPADecoder(latent_size, output_dim, enc_doc_hidden_size)

        self.pert_embeddings = self.add_weight(name='pert_embeddings', 
                                               shape=(num_perts, latent_size), 
                                               initializer="glorot_uniform",
                                               trainable=True)
        
        if use_covariates:
            self.cov_embeddings = self.add_weight(name='cov_embeddings', 
                                                  shape=(num_covs, latent_size), 
                                                  initializer="glorot_uniform",
                                                  trainable=True)
            
        self.use_covariates = use_covariates

        self.dose_encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(1,)),
            layers.Dense(dosage_enc_hidden_size, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(1, activation=None)
        ])

        self.pert_discriminator = CPADiscriminator(latent_size, num_perts, discriminator_hidden_size)
        if use_covariates:
            self.cov_discriminator = CPADiscriminator(latent_size, num_covs, discriminator_hidden_size)
        else:
            self.cov_discriminator = None

        self.GRL = GradientReversalLayer(lambda_=1.0)
        
        
    def call(self, x, pert_idx, dose, cov_idx=None, training=False):
        z_basal = self.encoder(x)

        outputs = {}

        if training:
            reversed_z_basal = self.GRL(z_basal)
            outputs["pert_pred"] = self.pert_discriminator(reversed_z_basal)
            if self.use_covariates and cov_idx is not None:
                outputs["cov_pred"] = self.cov_discriminator(reversed_z_basal)
        
        pert_vec = tf.nn.embedding_lookup(self.pert_embeddings, pert_idx)
        scaled_dose = self.dose_encoder(tf.expand_dims(dose, axis=-1))
        scaled_pert = tf.squeeze(scaled_dose, axis=-1)[:, tf.newaxis] * pert_vec

        z = z_basal + scaled_pert
        if self.use_covariates and cov_idx is not None:
            cov_vec = tf.nn.embedding_lookup(self.cov_embeddings, cov_idx)
            z = z + cov_vec
        
        x_hat = self.decoder(z)
        outputs["x_hat"] = x_hat

        return outputs