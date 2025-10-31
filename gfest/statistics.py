
import numpy as np
from dataclasses import dataclass

@dataclass
class FitStats:
    mse: float
    aicc: float
    r2: float
    r2_adj: float
    k: int  # number of active terms + bias
    n: int

def mse(y, yhat):
    err = y - yhat
    return float(np.mean(err*err))

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-18
    return 1.0 - ss_res/ss_tot

def aicc_from_mse(mse_val: float, n: int, k: int) -> float:
    # AICc = n*ln(MSE) + 2k + (2k(k+1))/(n-k-1)  (assuming Gaussian noise)
    if n <= k + 1:
        return np.inf
    return n*np.log(max(mse_val, 1e-18)) + 2*k + (2*k*(k+1))/(n - k - 1)

def adjusted_r2(y, yhat, k):
    n = len(y)
    r2 = r2_score(y, yhat)
    if n <= k + 1:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def compute_stats(y, yhat, k) -> FitStats:
    n = len(y)
    m = mse(y, yhat)
    aicc = aicc_from_mse(m, n, k)
    r2 = r2_score(y, yhat)
    r2_adj = adjusted_r2(y, yhat, k)
    return FitStats(m, aicc, r2, r2_adj, k, n)
