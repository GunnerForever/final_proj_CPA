from matplotlib import pyplot as plt
import tensorflow as tf
from keras import losses, metrics
import numpy as np
from tqdm import tqdm

reconstruction_loss_fn = losses.MeanSquaredError()
discriminator_loss_fn = losses.SparseCategoricalCrossentropy()
r2_score = metrics.R2Score()


def train_step(model, batch, ae_optimizer, dose_optimizer, adv_optimizer,
               lambda_adv=1.0, n_discriminator_steps=1):
    x, pert_idx, dose, cov_idx = batch

    with tf.GradientTape(persistent=True) as tape:
        outputs = model(x, pert_idx, dose, cov_idx, use_grl=True)

        recon_loss = reconstruction_loss_fn(x, outputs["x_hat"])
        pert_loss = discriminator_loss_fn(pert_idx, outputs["pert_pred"])
        cov_loss = discriminator_loss_fn(cov_idx, outputs["cov_pred"]) if model.use_covariates else 0.0
        adv_loss = pert_loss + cov_loss

        total_ae_loss = recon_loss + lambda_adv * adv_loss

    ae_vars = model.encoder.trainable_variables + model.decoder.trainable_variables + [model.pert_embeddings]
    if model.use_covariates:
        ae_vars += [model.cov_embeddings]
    ae_grads = tape.gradient(total_ae_loss, ae_vars)
    ae_optimizer.apply_gradients(zip(ae_grads, ae_vars))

    # dose_grads = tape.gradient(total_ae_loss, model.dose_encoder.trainable_variables)
    # dose_optimizer.apply_gradients(zip(dose_grads, model.dose_encoder.trainable_variables))

    for _ in range(n_discriminator_steps):
        with tf.GradientTape(persistent=True) as disc_tape:
            outputs_disc = model(x, pert_idx, dose, cov_idx, use_grl=False)

            pert_logits = outputs_disc["pert_pred"]
            pert_loss_d = discriminator_loss_fn(pert_idx, pert_logits)

            cov_loss_d = 0.0
            if model.use_covariates:
                cov_logits = outputs_disc["cov_pred"]
                cov_loss_d = discriminator_loss_fn(cov_idx, cov_logits)

            total_disc_loss = pert_loss_d + cov_loss_d

        disc_vars = model.pert_discriminator.trainable_variables
        if model.use_covariates:
            disc_vars += model.cov_discriminator.trainable_variables
        disc_grads = disc_tape.gradient(total_disc_loss, disc_vars)
        adv_optimizer.apply_gradients(zip(disc_grads, disc_vars))

    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    acc.update_state(pert_idx, outputs_disc["pert_pred"])
    print(f"Discriminator accuracy: {acc.result().numpy():.4f}")

    return {
        "reconstruction_loss": recon_loss.numpy(),
        "perturbation_loss": pert_loss.numpy(),
        "covariates_loss": cov_loss.numpy() if model.use_covariates else 0.0,
        "adv_loss": adv_loss.numpy(),
        "total_ae_loss": total_ae_loss.numpy(),
        "r2_score": r2_score(x, outputs["x_hat"]).numpy()
    }


def test_step(model, batch):
    x, pert_idx, dose, cov_idx = batch
    outputs = model(x, pert_idx, dose, cov_idx, use_grl=False)

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
                                 lambda_adv=5, n_discriminator_steps=n_discriminator_steps)
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