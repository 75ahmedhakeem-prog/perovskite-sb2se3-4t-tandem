"""
09_train_tandem.py
Train XGBoost and an artificial neural network (MLP) on the SCAPS-1D 4T tandem
dataset, compare them, explain the best tree model and search for the optimum.

Inputs (6): d_PVK_um, Nt_PVK, Nit_PVK_Spiro, d_SbSe_um, Nt_SbSe, Nit_CdS_SbSe
Defect densities enter the models as log10 values.

Usage
  pip install numpy pandas scikit-learn xgboost shap matplotlib scipy
  python 09_train_tandem.py --data tandem_dataset.csv --target PCE_4T
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import differential_evolution
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:  # only for testing the pipeline; never for reported results
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_XGB = False

FEATURES = ["d_PVK_um", "Nt_PVK", "Nit_PVK_Spiro", "d_SbSe_um", "Nt_SbSe", "Nit_CdS_SbSe"]
LOG = ["Nt_PVK", "Nit_PVK_Spiro", "Nt_SbSe", "Nit_CdS_SbSe"]
LABELS = {"d_PVK_um": "d$_{PVK}$", "Nt_PVK": "N$_{t,PVK}$", "Nit_PVK_Spiro": "N$_{it,PVK/Spiro}$",
          "d_SbSe_um": "d$_{Sb_2Se_3}$", "Nt_SbSe": "N$_{t,Sb_2Se_3}$", "Nit_CdS_SbSe": "N$_{it,CdS/Sb_2Se_3}$"}
SEED = 42
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.labelsize": 14, "axes.linewidth": 1.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})


def to_X(df):
    X = df[FEATURES].copy()
    for c in LOG:
        X[c] = np.log10(X[c])
    return X


def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p)}


def build_tree(X, y, n_iter):
    if HAVE_XGB:
        est = XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1)
        grid = {"n_estimators": [300, 600, 1000, 1500], "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5],
                "reg_lambda": [0.5, 1.0, 3.0]}
    else:
        print("  WARNING: xgboost not installed; sklearn HistGradientBoosting used for a pipeline test only")
        est = HistGradientBoostingRegressor(random_state=SEED)
        grid = {"max_iter": [300, 600, 1000], "max_depth": [3, 5, 8, None],
                "learning_rate": [0.03, 0.05, 0.1], "min_samples_leaf": [5, 10, 20]}
    s = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=5, scoring="r2", random_state=SEED, n_jobs=-1)
    s.fit(X, y)
    print("  best params:", s.best_params_)
    return s.best_estimator_


def build_ann(X, y, n_iter):
    base = make_pipeline(StandardScaler(), MLPRegressor(max_iter=4000, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=60, random_state=SEED))
    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    grid = {"regressor__mlpregressor__hidden_layer_sizes": [(64, 64), (128, 64), (128, 128, 64), (256, 128, 64)],
            "regressor__mlpregressor__activation": ["relu", "tanh"],
            "regressor__mlpregressor__alpha": [1e-5, 1e-4, 1e-3],
            "regressor__mlpregressor__learning_rate_init": [1e-3, 3e-3]}
    s = RandomizedSearchCV(model, grid, n_iter=n_iter, cv=5, scoring="r2", random_state=SEED, n_jobs=-1)
    s.fit(X, y)
    print("  best params:", s.best_params_)
    return s.best_estimator_


def parity(results, y, target, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    lo, hi = np.floor(y.min()), np.ceil(y.max())
    for ax, (name, r), col, lab in zip(axes, results.items(), ["#1F4FB4", "#D62728"], ["(a)", "(b)"]):
        ax.scatter(y, r["pred"], s=14, color=col, alpha=0.55, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"SCAPS-1D {target} (%)".replace("_", " ")); ax.set_ylabel(f"Predicted {target} (%)".replace("_", " "))
        m = r["test"]
        ax.text(0.05, 0.94, f"{lab} {name}", transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
        ax.text(0.95, 0.06, f"R$^2$ = {m['R2']:.4f}\nRMSE = {m['RMSE']:.3f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def importance(model, X, y, path):
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns)
        how = "Mean |SHAP value| (%)"
    except Exception:
        r = permutation_importance(model, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
        imp = pd.Series(r.importances_mean, index=X.columns)
        how = "Permutation importance (R$^2$ loss)"
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh([LABELS[c] for c in imp.index], imp.values, color="#1F4FB4", edgecolor="black", lw=1.2)
    ax.set_xlabel(how); ax.tick_params(axis="y", which="both", right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.savefig(path); plt.close(fig)
    print(f"  {how}:\n" + imp[::-1].round(4).to_string())
    return imp


def optimise(model, X, path):
    bounds = [(X[c].min(), X[c].max()) for c in FEATURES]
    f = lambda x: -float(model.predict(pd.DataFrame([x], columns=FEATURES))[0])
    res = differential_evolution(f, bounds, seed=SEED, maxiter=200, tol=1e-7, polish=False)
    best = {c: float(10 ** v if c in LOG else v) for c, v in zip(FEATURES, res.x)}
    best["predicted"] = -float(res.fun)
    json.dump(best, open(path, "w"), indent=2)
    print("  Optimum (verify in SCAPS-1D):", {k: round(v, 4) if v < 1e3 else f"{v:.3g}" for k, v in best.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tandem_dataset.csv")
    ap.add_argument("--target", default="PCE_4T")
    ap.add_argument("--n-iter", type=int, default=20)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    X, y = to_X(df), df[a.target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"{len(df)} samples: train {len(X_tr)}, test {len(X_te)}; target {a.target}")
    results = {}
    for name, builder in [("XGBoost" if HAVE_XGB else "GBM (test only)", build_tree), ("ANN", build_ann)]:
        print(f"\n{name}")
        model = builder(X_tr, y_tr, a.n_iter)
        cv = cross_val_score(model, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
        pred = model.predict(X_te)
        results[name] = {"model": model, "pred": pred, "cv": cv, "test": metrics(y_te, pred)}
        print(f"  CV R2 = {cv.mean():.5f} ± {cv.std():.5f}")
        print("  Test: " + ", ".join(f"{k} = {v:.5f}" for k, v in results[name]["test"].items()))
    tab = pd.DataFrame({n: {**r["test"], "CV_R2_mean": r["cv"].mean(), "CV_R2_std": r["cv"].std()}
                        for n, r in results.items()}).T
    tab.to_csv(f"model_comparison_{a.target}.csv"); print("\n" + tab.round(5).to_string())
    parity(results, y_te, a.target, f"parity_{a.target}.png")
    tree = list(results.values())[0]["model"]
    importance(tree, X_te, y_te, f"importance_{a.target}.png")
    best = max(results, key=lambda n: results[n]["test"]["R2"])
    print(f"\nBest model: {best}")
    optimise(results[best]["model"], X, f"optimum_{a.target}.json")
    print(f"Best sample in the dataset: {df[a.target].max():.3f} %")


if __name__ == "__main__":
    main()
