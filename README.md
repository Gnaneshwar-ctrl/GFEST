# GFEST — Genetic Feature Extraction Statistical Testing

This repository contains reproduced implementation to **discover mechanistic, first‑principles‑inspired models** from data It couples a **genetic search** over interpretable primitives with **statistical regression** (OLS / LASSO) and **model selection** (AICc, CV‑MSE).

### Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Quickstart (synthetic replications)

1) Generate synthetic datasets (Michaelis–Menten, 2nd‑order kinetics):
```bash
python examples/generate_synthetic.py
```

2) Run GFEST on Michaelis–Menten (noisy) with LASSO selection:
```bash
python gfest/run_gfest.py --csv examples/michaelis_menten.csv --y y   --x x --reg lasso --pop 40 --gens 25 --terms 12 --seed 42   --sg_window 11 --sg_poly 3
```

3) Run GFEST on 2nd‑order kinetics (A + B → …) with OLS selection:
```bash
python gfest/run_gfest.py --csv examples/second_order.csv --y y   --x A B --reg ols --pop 40 --gens 25 --terms 2 --seed 0
```

Outputs (model, metrics, plots) land in `results/`.

### Cite
This implementation is a *replication* based on the following work https://doi.org/10.1016/j.compchemeng.2020.106900. 
