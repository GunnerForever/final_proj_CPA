from matplotlib import pyplot as plt
import tensorflow as tf
from keras import losses, metrics
import numpy as np
from tqdm import tqdm

reconstruction_loss_fn = losses.MeanSquaredError()
discriminator_loss_fn = losses.SparseCategoricalCrossentropy()
r2_score = metrics.R2Score()


def train_step(model, batch, ae_optimizer, dose_optimizer, adv_optimizer,
               lambda_adv=5.0, lambda_gp=3.0, n_discriminator_steps=3):
    x, pert_idx, dose, cov_idx = batch

    with tf.GradientTape(persistent=True) as tape:
        outputs = model(x, pert_idx, dose, cov_idx, training=True)
        recon_loss = reconstruction_loss_fn(x, outputs["x_hat"])
        pert_loss = discriminator_loss_fn(pert_idx, outputs["pert_pred"])
        cov_loss = discriminator_loss_fn(cov_idx, outputs["cov_pred"]) if model.use_covariates else 0.0
        adv_loss = pert_loss + cov_loss

        # z_basal = model.encoder(x)
        # reversed_z = model.GRL(z_basal)
        # tape.watch(z_basal)

        # pert_logits = model.pert_discriminator(reversed_z)
        # gp_pert = tf.reduce_mean(tf.reduce_sum(tf.square(tape.gradient(pert_logits, z_basal)), axis=1))
        # 
        # gp_cov = 0.0
        # if model.use_covariates:
        #     cov_logits = model.cov_discriminator(reversed_z)
        #    gp_cov = tf.reduce_mean(tf.reduce_sum(tf.square(tape.gradient(cov_logits, z_basal)), axis=1))

        # gradient_penalty = lambda_gp * (gp_pert + gp_cov)

        total_ae_loss = recon_loss - lambda_adv * adv_loss
        # total_adv_loss = adv_loss + gradient_penalty
        total_adv_loss = adv_loss

    ae_vars = model.encoder.trainable_variables + model.decoder.trainable_variables + [model.pert_embeddings]
    if model.use_covariates:
        ae_vars += [model.cov_embeddings]
    ae_grads = tape.gradient(total_ae_loss, ae_vars)
    ae_optimizer.apply_gradients(zip(ae_grads, ae_vars))

    dose_grads = tape.gradient(total_ae_loss, model.dose_encoder.trainable_variables)
    dose_optimizer.apply_gradients(zip(dose_grads, model.dose_encoder.trainable_variables))

    adv_vars = model.pert_discriminator.trainable_variables
    if model.use_covariates:
        adv_vars += model.cov_discriminator.trainable_variables
    for _ in range(n_discriminator_steps):
        adv_grads = tape.gradient(total_adv_loss, adv_vars)
        adv_optimizer.apply_gradients(zip(adv_grads, adv_vars))

    return {
        "total_ae_loss": total_ae_loss.numpy(),
        "reconstruction_loss": recon_loss.numpy(),
        "perturbation_loss": pert_loss.numpy(),
        "covariates_loss": cov_loss.numpy() if model.use_covariates else 0.0,
        "r2_score": r2_score(x, outputs["x_hat"]).numpy(),
    }


def test_step(model, batch):
    x, pert_idx, dose, cov_idx = batch
    outputs = model(x, pert_idx, dose, cov_idx, training=False)

    return reconstruction_loss_fn(x, outputs["x_hat"]).numpy(), r2_score(x, outputs["x_hat"]).numpy()


def train(model, 
          train_dataset, val_dataset, 
          ae_optimizer, dose_optimizer, adv_optimizer, 
          epochs=200, batch_size=128,
          n_discriminator_steps=3):
    
    x_train, pert_train, dose_train, cov_train = train_dataset
    x_val, pert_val, dose_val, cov_val = val_dataset

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, pert_train, dose_train, cov_train)
        ).shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for step, batch in enumerate(train_ds):
            metrics = train_step(model, batch, ae_optimizer, dose_optimizer, adv_optimizer, 
                                 lambda_adv=5.0, lambda_gp=3.0, n_discriminator_steps=3)
            if step % 64 == 0:
                print(f"Step {step}: {metrics}")
        
        val_ds = tf.data.Dataset.from_tensor_slices(
            (x_val, pert_val, dose_val, cov_val)
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_losses, val_r2 = [], []
        for batch in val_ds:
            loss, r2 = test_step(model, batch)
            val_losses.append(loss)
            val_r2.append(r2)
        
        print(f"Validation Loss: {np.mean(val_losses)}, R2 Score: {np.mean(val_r2)}")


def test(model, test_dataset):
    test_losses, test_r2 = [], []
    for batch in test_dataset:
        loss, r2 = test_step(model, batch)
        test_losses.append(loss.numpy())
        test_r2.append(r2.numpy())
    
    print(f"OOD Loss: {np.mean(test_losses)}, R2 Score: {np.mean(test_r2)}")
    return np.mean(test_losses), np.mean(test_r2)