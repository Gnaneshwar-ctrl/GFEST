import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
try:
    from .genetics import evolve, build_and_fit
    from .models import eval_expression, build_feature_matrix
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from gfest.genetics import evolve, build_and_fit
    from gfest.models import eval_expression, build_feature_matrix


def parse_args():
    ap = argparse.ArgumentParser(description="GFEST: Genetic Feature Extraction + Statistical Testing (minimal).")
    ap.add_argument("--csv", required=True, help="CSV with columns y and features")
    ap.add_argument("--y", required=True, help="Name of the response column")
    ap.add_argument("--x", nargs="+", required=True, help="List of feature column names")
    ap.add_argument("--reg", choices=["ols","lasso"], default="lasso")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--terms", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cv", type=int, default=3, help="K-fold for post-hoc CV reporting")
    ap.add_argument("--sg_window", type=int, default=0, help="Savitzky–Golay window length (odd). 0 disables.")
    ap.add_argument("--sg_poly", type=int, default=3, help="Savitzky–Golay polynomial order.")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df[args.y].to_numpy().astype(float)
    X = df[args.x].to_numpy().astype(float)

    if args.sg_window and args.sg_window > 2 and args.sg_window % 2 == 1:
        if X.ndim == 2:
            Xs = np.copy(X)
            for j in range(X.shape[1]):
                Xs[:,j] = savgol_filter(X[:,j], window_length=args.sg_window, polyorder=args.sg_poly, mode="interp")
            X = Xs
        ys = savgol_filter(y, window_length=args.sg_window, polyorder=args.sg_poly, mode="interp")
        y = ys

    best_ch, best_model, best_stats, hist = evolve(
        X, y, reg=args.reg, pop=args.pop, gens=args.gens, K=args.terms, seed=args.seed
    )

    final_model, final_stats = build_and_fit(best_ch, X, y, args.reg, seed=args.seed)

    Phi, act_idx = build_feature_matrix(best_ch, X)
    coefs = final_model["coef"]
    bias = final_model["bias"]
    expr = best_ch.to_expression(args.x, coefs, bias)
    expr_tex = best_ch.to_latex(args.x, coefs, bias)

    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    cv_mse = []; cv_r2 = []
    for tr, te in kf.split(X):
        Xm, ym = X[tr], y[tr]
        Xt, yt = X[te], y[te]
        m, _ = build_and_fit(best_ch, Xm, ym, args.reg, seed=args.seed)
        yhat = eval_expression(best_ch, Xt, m["coef"], m["bias"])
        cv_mse.append(mean_squared_error(yt, yhat))
        cv_r2.append(r2_score(yt, yhat))

    out = {
        "expression": expr,
        "latex": expr_tex,
        "stats": final_stats.__dict__,
        "cv_mse_mean": float(np.mean(cv_mse)),
        "cv_r2_mean": float(np.mean(cv_r2)),
        "history_best_aicc": [float(h[1]) for h in hist],
        "args": vars(args),
    }
    (outdir/"model.json").write_text(json.dumps(out, indent=2))

    yhat_all = eval_expression(best_ch, X, final_model["coef"], final_model["bias"])
    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    plt.scatter(y, yhat_all, s=12)
    lims = [min(y.min(), yhat_all.min()), max(y.max(), yhat_all.max())]
    plt.plot(lims, lims)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Parity plot")
    fig1.savefig(outdir/"parity.png", dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(out["history_best_aicc"])
    plt.xlabel("Generation")
    plt.ylabel("Best AICc")
    plt.title("Convergence (AICc)")
    fig2.savefig(outdir/"aicc_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    print("==== GFEST RESULT ====")
    print(expr)
    print("LaTeX:", expr_tex)
    print("CV MSE (mean):", np.mean(cv_mse))
    print("CV R2  (mean):", np.mean(cv_r2))
    print("Saved to:", outdir)

if __name__ == "__main__":
    main()
