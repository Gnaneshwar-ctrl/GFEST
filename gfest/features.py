
import numpy as np

_EPS = 1e-8

def _id(x): return x
def _square(x): return x*x
def _cube(x): return x*x*x
def _sqrt(x): return np.sqrt(np.abs(x) + _EPS)
def _log1p_abs(x): return np.log1p(np.abs(x))
def _exp(x): return np.exp(np.clip(x, -20.0, 20.0))  # clip to avoid overflow
def _sin(x): return np.sin(x)
def _cos(x): return np.cos(x)
def _inv(x): return 1.0/(np.abs(x) + _EPS)

PRIMITIVES = [
    ("id", _id),
    ("square", _square),
    ("sqrt", _sqrt),
    ("log1p_abs", _log1p_abs),
    ("exp", _exp),
    ("sin", _sin),
    ("cos", _cos),
    ("inv", _inv),
    ("cube", _cube),
]

PRIM_NAME_TO_IDX = {name:i for i,(name,_) in enumerate(PRIMITIVES)}

def apply_primitive(idx, x):
    return PRIMITIVES[idx][1](x)

def primitive_name(idx):
    return PRIMITIVES[idx][0]
