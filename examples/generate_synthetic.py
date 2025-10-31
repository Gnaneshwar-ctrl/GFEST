import numpy as np, pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

def michaelis_menten(n=300, Vmax=2.0, Km=0.5, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 3.0, size=n)
    y_clean = Vmax * x/(Km + x)
    y = y_clean + rng.normal(0.0, noise*np.std(y_clean), size=n)
    df = pd.DataFrame({"x": x, "y": y})
    return df

def second_order(n=300, k=0.12, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 3.0, size=n)
    B = rng.uniform(0.0, 3.0, size=n)
    y_clean = k * A * B
    y = y_clean + rng.normal(0.0, noise*np.std(y_clean), size=n)
    df = pd.DataFrame({"A": A, "B": B, "y": y})
    return df

if __name__ == "__main__":
    (OUT/"michaelis_menten.csv").write_text(michaelis_menten().to_csv(index=False))
    (OUT/"second_order.csv").write_text(second_order().to_csv(index=False))
    print("Wrote:", OUT/"michaelis_menten.csv")
    print("Wrote:", OUT/"second_order.csv")
