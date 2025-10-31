
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from .models import Chromosome, build_feature_matrix, eval_expression
from .statistics import compute_stats
from .features import PRIMITIVES

def init_population(pop: int, K: int, dvars: int, n_prims: int, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(pop):
        genes = np.zeros((K,6), dtype=int)
        for t in range(K):
            fam = rng.integers(0,2)            # 0 unary, 1 product
            v1 = rng.integers(0, dvars)
            v2 = rng.integers(0, dvars)
            p1 = rng.integers(0, n_prims)
            p2 = rng.integers(0, n_prims)
            active = 1 if rng.random() < 0.7 else 0
            genes[t] = [fam, v1, v2, p1, p2, active]
        out.append(Chromosome(genes))
    return out

def mutate(ch: Chromosome, dvars: int, n_prims: int, pmut=0.1, seed=None) -> Chromosome:
    rng = np.random.default_rng(seed)
    g = ch.genes.copy()
    K = g.shape[0]
    for t in range(K):
        if rng.random() < pmut:
            field = rng.integers(0,6)
            if field == 0:  # family
                g[t,0] = 1 - g[t,0]
            elif field in (1,2):  # var idx
                g[t,field] = rng.integers(0, dvars)
            elif field in (3,4):  # primitive
                g[t,field] = rng.integers(0, n_prims)
            else:  # active toggle
                g[t,5] = 1 - g[t,5]
    return Chromosome(g)

def crossover(a: Chromosome, b: Chromosome, seed=None):
    rng = np.random.default_rng(seed)
    K = a.genes.shape[0]
    cut = rng.integers(1, K)  # single-point
    c1 = np.vstack([a.genes[:cut], b.genes[cut:]])
    c2 = np.vstack([b.genes[:cut], a.genes[cut:]])
    return Chromosome(c1), Chromosome(c2)

def build_and_fit(ch: Chromosome, X: np.ndarray, y: np.ndarray, reg: str, seed=0):
    Phi, act_idx = build_feature_matrix(ch, X)
    n = X.shape[0]
    if Phi.shape[1] == 0:
        bias = float(np.mean(y))
        coef = np.zeros((0,))
        yhat = np.full(n, bias)
        stats = compute_stats(y, yhat, k=1)
        model = {"bias": bias, "coef": coef, "act_idx": act_idx}
        return model, stats

    if reg.lower() == "ols":
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Phi, y)
        yhat = lr.predict(Phi)
        coef = lr.coef_
        bias = float(lr.intercept_)
    elif reg.lower() == "lasso":
        lcv = LassoCV(alphas=None, cv=5, random_state=seed, max_iter=5000)
        lcv.fit(Phi, y)
        yhat = lcv.predict(Phi)
        coef = lcv.coef_
        bias = float(lcv.intercept_)
    else:
        raise ValueError("Unknown reg type. Use 'ols' or 'lasso'.")

    k = 1 + int(np.sum(np.abs(coef) > 1e-12))
    stats = compute_stats(y, yhat, k=k)
    model = {"bias": bias, "coef": coef, "act_idx": act_idx}
    return model, stats

def tournament_select(objs, k=2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(objs)
    cand = rng.choice(n, size=k, replace=False)
    best = cand[0]
    bestv = objs[best]
    for c in cand[1:]:
        if objs[c] < bestv:
            best = c; bestv = objs[c]
    return int(best)

def evolve(X, y, reg="lasso", pop=40, gens=25, K=12, seed=0, pmut=0.2, elitism=2):
    rng = np.random.default_rng(seed)
    dvars = X.shape[1]
    n_prims = len(PRIMITIVES)

    P = init_population(pop, K, dvars, n_prims, seed=seed)
    fits = [build_and_fit(ch, X, y, reg, seed=seed)[1] for ch in P]
    objs = [f.aicc for f in fits]

    best_hist = []

    for g in range(gens):
        elite_idx = np.argsort(objs)[:elitism]
        newP = [P[i].copy() for i in elite_idx]

        while len(newP) < pop:
            i = tournament_select(objs, k=3, seed=rng.integers(1e9))
            j = tournament_select(objs, k=3, seed=rng.integers(1e9))
            c1, c2 = crossover(P[i], P[j], seed=rng.integers(1e9))
            c1 = mutate(c1, dvars, n_prims, pmut=pmut, seed=rng.integers(1e9))
            c2 = mutate(c2, dvars, n_prims, pmut=pmut, seed=rng.integers(1e9))
            newP += [c1, c2]
        P = newP[:pop]

        fits = [build_and_fit(ch, X, y, reg, seed=seed)[1] for ch in P]
        objs = [f.aicc for f in fits]

        b = int(np.argmin(objs))
        best_hist.append((g, objs[b], fits[b]))

    b = int(np.argmin(objs))
    best_ch = P[b]
    best_model, best_stats = build_and_fit(best_ch, X, y, reg, seed=seed)
    return best_ch, best_model, best_stats, best_hist
