import numpy as np
import tensorflow as tf
from keras import layers
from encoder import CPAEncoder
from decoder import CPADecoder
from grad_reverse import GradientReversalLayer
from discriminator import CPADiscriminator

class CPAModel(tf.keras.Model):
    def __init__(self, input_dim, latent_size, output_dim, enc_dec_hidden_size, dose_enc_hidden_size,
                 discriminator_hidden_size, num_perts, use_covariates=False, num_covs=0, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = CPAEncoder(input_dim, latent_size, enc_dec_hidden_size)
        self.decoder = CPADecoder(latent_size, output_dim, enc_dec_hidden_size)

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

        # self.dose_encoder = tf.keras.Sequential([
        #     layers.Dense(1, activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-7)),
        # ])

        self.pert_discriminator = CPADiscriminator(latent_size, num_perts, discriminator_hidden_size)
        if use_covariates:
            self.cov_discriminator = CPADiscriminator(latent_size, num_covs, discriminator_hidden_size)
        else:
            self.cov_discriminator = None

        self.GRL = GradientReversalLayer(lambda_=1.0)
        
        
    def call(self, x, pert_idx, dose, cov_idx=None, use_grl=False):
        z_basal = self.encoder(x)
        outputs = {}

        if use_grl:
            z_for_disc = self.GRL(z_basal + tf.random.normal(tf.shape(z_basal), stddev=0.05))
        else:
            z_for_disc = z_basal
    
        outputs["pert_pred"] = self.pert_discriminator(z_for_disc)
        if self.use_covariates and cov_idx is not None:
            outputs["cov_pred"] = self.cov_discriminator(z_for_disc)

        pert_vec = tf.nn.embedding_lookup(self.pert_embeddings, pert_idx)
        # scaled_dose = self.dose_encoder(tf.expand_dims(dose, -1))
        # scaled_pert = tf.squeeze(scaled_dose, -1)[:, tf.newaxis] * pert_vec
        scaled_pert = dose[:, tf.newaxis] * pert_vec

        z = z_basal + scaled_pert
        if self.use_covariates and cov_idx is not None:
            cov_vec = tf.nn.embedding_lookup(self.cov_embeddings, cov_idx)
            z = z + cov_vec

        x_hat = self.decoder(z)
        outputs["x_hat"] = x_hat

        return outputs