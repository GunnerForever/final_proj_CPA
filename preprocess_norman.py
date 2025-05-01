import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_norman(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    return adata

def prepare_norman(adata, batch_size=1000, split_col="split_2"):
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    pert_encoder = LabelEncoder()
    pert_idx = pert_encoder.fit_transform(adata.obs["condition_ID"].astype(str))

    def parse_dose(dose_str):
        parts = dose_str.split('+')
        return sum([float(p) for p in parts])

    dose = adata.obs["dose_value"].astype(str).apply(parse_dose).astype(np.float32).values

    cov_idx = np.zeros(len(adata), dtype=np.int32)          # No covariates in this dataset

    split = adata.obs[split_col]

    train_mask = split == "train"
    val_mask = split == "valid"
    ood_mask = split == "ood"

    train_data = (
        X[train_mask],
        pert_idx[train_mask],
        dose[train_mask],
        cov_idx[train_mask]
    )

    val_data = (
        X[val_mask],
        pert_idx[val_mask],
        dose[val_mask],
        cov_idx[val_mask]
    )

    ood_data = (
        X[ood_mask],
        pert_idx[ood_mask],
        dose[ood_mask],
        cov_idx[ood_mask]
    )

    return train_data, val_data, ood_data