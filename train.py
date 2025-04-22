from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch

import os
import tensorflow as tf
from keras import losses, metrics
import numpy as np
import random
import math

reconstruction_loss_fn = losses.MeanSquaredError()
discriminator_loss_fn = losses.SparseCategoricalCrossentropy()
r2_score = metrics.R2Score()


def train_step(model, batch, optimizer, lambda_adv=1.0):
    x, pert_idx, dose, cov_idx = batch

    with tf.GradientTape() as tape:
        outputs = model(x, pert_idx, dose, cov_idx, training=True)

        reconstruction_loss = reconstruction_loss_fn(x, outputs["x_hat"])

        pert_loss = discriminator_loss_fn(pert_idx, outputs["pert_pred"])
        if model.use_covariates:
            cov_loss = discriminator_loss_fn(cov_idx, outputs["cov_pred"])
        else:
            cov_loss = 0.0

        total_loss = reconstruction_loss + lambda_adv * (pert_loss + cov_loss)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "perturbation_loss": pert_loss,
        "covariates_loss": cov_loss,
        "r2_score": r2_score(x, outputs["x_hat"]),
    }


def test_step(model, batch):
    x, pert_idx, dose, cov_idx = batch

    outputs = model(x, pert_idx, dose, cov_idx, training=False)

    return reconstruction_loss_fn(x, outputs["x_hat"]), r2_score(x, outputs["x_hat"])


def train(model, train_dataset, val_dataset, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for step, batch in enumerate(train_dataset):
            metrics = train_step(model, batch, optimizer)
            if step % 100 == 0:
                print(f"Step {step}: {metrics}")
        
        val_losses, val_r2 = [], []
        for batch in val_dataset:
            loss, r2 = test_step(model, batch)
            val_losses.append(loss.numpy())
            val_r2.append(r2.numpy())
        
        print(f"Validation Loss: {np.mean(val_losses)}, R2 Score: {np.mean(val_r2)}")

def visualize_loss(losses):
    pass

def visualize_results():
    pass
