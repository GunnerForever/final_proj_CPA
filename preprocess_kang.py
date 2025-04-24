import scanpy as sc
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_kang(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    return adata

def prepare_kang(adata, batch_size=128, split_col="split_CD14 Mono"):
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    pert = (adata.obs["stim"].astype(str).str.upper() == "STIM").astype(int).values
    dose = pert.astype(np.float32)

    le = LabelEncoder()
    cov = le.fit_transform(adata.obs["cell_type"].astype(str))

    split = adata.obs[split_col]

    train_mask = split == "train"
    val_mask = split == "valid"
    ood_mask = split == "ood"

    train_data = (
        X[train_mask],
        pert[train_mask],
        dose[train_mask],
        cov[train_mask]
    )

    val_data = (
        X[val_mask],
        pert[val_mask],
        dose[val_mask],
        cov[val_mask]
    )

    ood_data = (
        X[ood_mask],
        pert[ood_mask],
        dose[ood_mask],
        cov[ood_mask]
    )

    return train_data, val_data, ood_data