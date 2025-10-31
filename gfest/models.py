import numpy as np
from dataclasses import dataclass
from typing import List
from .features import apply_primitive, primitive_name

@dataclass
class Chromosome:
    """Fixed-length chromosome encoding up to K terms.
    Each term t has fields:
      - family: 0 (unary) or 1 (binary product)
      - var1: index into input columns
      - var2: index into input columns (unused for unary, but stored)
      - prim1: index into primitive list
      - prim2: index into primitive list (unused for unary)
      - active: 0/1 toggle
    Shape: (K, 6)
    """
    genes: np.ndarray  # int dtype, shape (K, 6)

    def copy(self):
        return Chromosome(self.genes.copy())

    def K(self):
        return self.genes.shape[0]

    def to_expression(self, xnames: List[str], coefs: np.ndarray, bias: float) -> str:
        terms = []
        active_counter = 0
        for t in range(self.K()):
            fam, v1, v2, p1, p2, active = self.genes[t]
            if not active: 
                continue
            c = coefs[active_counter] if active_counter < len(coefs) else 0.0
            active_counter += 1
            if np.abs(c) < 1e-12: 
                continue
            if fam == 0:
                s = f"{primitive_name(p1)}({xnames[v1]})"
            else:
                s = f"{primitive_name(p1)}({xnames[v1]})*{primitive_name(p2)}({xnames[v2]})"
            terms.append(f"{c:+.6g}*{s}")
        if len(terms)==0:
            return f"y ≈ {bias:.6g}"
        return f"y ≈ {bias:.6g} " + " ".join(terms)

    def to_latex(self, xnames: List[str], coefs: np.ndarray, bias: float) -> str:
        terms = []
        active_counter = 0
        for t in range(self.K()):
            fam, v1, v2, p1, p2, active = self.genes[t]
            if not active: continue
            c = coefs[active_counter] if active_counter < len(coefs) else 0.0
            active_counter += 1
            if np.abs(c) < 1e-12: 
                continue
            name1 = primitive_name(p1).replace("_", "\\,")
            s1 = f"\\mathrm{{{name1}}}({xnames[v1]})"
            if fam == 0:
                s = s1
            else:
                name2 = primitive_name(p2).replace("_", "\\,")
                s2 = f"\\mathrm{{{name2}}}({xnames[v2]})"
                s = s1 + "\\,\\cdot\\," + s2
            terms.append(f"{c:+.6g}\\,{s}")
        if len(terms)==0:
            return f"$y \\approx {bias:.6g}$"
        return "$y \\approx " + f"{bias:.6g} " + " ".join(terms) + "$"

def build_feature_matrix(ch: Chromosome, X: np.ndarray):
    n, d = X.shape
    K = ch.K()
    cols = []
    act_idx = []
    for t in range(K):
        fam, v1, v2, p1, p2, active = ch.genes[t]
        if not active:
            continue
        v1 = int(v1) % d
        v2 = int(v2) % d
        col1 = apply_primitive(int(p1), X[:, v1])
        if fam == 0:
            col = col1
        else:
            col2 = apply_primitive(int(p2), X[:, v2])
            col = col1 * col2
        cols.append(col.reshape(-1,1))
        act_idx.append(t)
    if len(cols)==0:
        Phi = np.zeros((n, 0))
    else:
        Phi = np.hstack(cols)
    return Phi, np.array(act_idx, dtype=int)

def eval_expression(ch: Chromosome, X: np.ndarray, coefs: np.ndarray, bias: float) -> np.ndarray:
    Phi, act_idx = build_feature_matrix(ch, X)
    if Phi.shape[1] == 0:
        return np.full(X.shape[0], bias)
    yhat = bias + Phi @ coefs
    return yhat
