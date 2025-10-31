
import numpy as np

def train_val_test_split(n, seed=0, val_frac=0.2, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(val_frac*n)
    n_test = int(test_frac*n)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx = idx[n_val+n_test:]
    return train_idx, val_idx, test_idx

def kfold_indices(n, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds
