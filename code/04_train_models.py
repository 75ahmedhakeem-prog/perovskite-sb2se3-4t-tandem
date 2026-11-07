"""
04_train_models.py
Train and compare XGBoost vs. ANN (MLP) on the SCAPS tandem dataset,
explain the best model with SHAP, and search for the optimum design.

Usage:
    python 04_train_models.py --data dataset.csv --target PCE_4T
    python 04_train_models.py --demo        # synthetic toy data, only to test the pipeline

Requirements:
    pip install numpy pandas scikit-learn xgboost shap matplotlib scipy
"""
import os
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Features sampled on a log scale are fed to the models as log10(x)
LOG_FEATURES = ["pvk_NA_cm3", "pvk_Nt_cm3", "top_IL_Nt_cm2",
                "sbse_NA_cm3", "sbse_Nt_cm3", "bot_IL_Nt_cm2"]
FEATURES = ["pvk_thick_um", "pvk_Eg_eV", "pvk_NA_cm3", "pvk_Nt_cm3",
            "etl_thick_um", "htl_thick_um", "top_IL_Nt_cm2",
            "sbse_thick_um", "sbse_NA_cm3", "sbse_Nt_cm3",
            "cds_thick_um", "bot_IL_Nt_cm2"]
SEED = 42


# ---------------------------------------------------------------------------
def make_demo_data(n=1500, seed=SEED):
    """Synthetic toy data with plausible trends. NOT physics, only for testing."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_generate_samples.py"))
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    df = gen.lhs_samples(gen.SPACE, n, seed)
    rng = np.random.default_rng(seed)
    d, Eg = df.pvk_thick_um, df.pvk_Eg_eV
    absorb_top = 1 - np.exp(-d / 0.25)
    pce_top = 21.0 * absorb_top * (1 - 0.6 * (Eg - 1.7) ** 2) \
        - 1.2 * (np.log10(df.pvk_Nt_cm3) - 13) - 0.8 * (np.log10(df.top_IL_Nt_cm2) - 9) ** 1.2 / 2
    trans = np.exp(-d / 0.35) * 0.4 + (Eg - 1.55) * 1.5
    pce_bot = 8.0 * trans * (1 - np.exp(-df.sbse_thick_um / 0.6)) \
        - 0.6 * (np.log10(df.sbse_Nt_cm3) - 12) - 0.3 * (np.log10(df.bot_IL_Nt_cm2) - 9)
    df["PCE_4T"] = np.clip(pce_top, 0, None) + np.clip(pce_bot, 0, None) + rng.normal(0, 0.3, n)
    return df


def to_X(df):
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log10(X[c])
    return X


def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAE": mean_absolute_error(y, p)}


# ---------------------------------------------------------------------------
def build_xgb(X_tr, y_tr):
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1),
        param_distributions={
            "n_estimators": [300, 600, 1000],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [0.5, 1.0, 3.0],
        },
        n_iter=25, cv=5, scoring="r2", random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    print("  XGBoost best params:", search.best_params_)
    return search.best_estimator_


def build_ann(X_tr, y_tr):
    base = make_pipeline(StandardScaler(), MLPRegressor(
        max_iter=3000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=50, random_state=SEED))
    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    search = RandomizedSearchCV(
        model,
        param_distributions={
            "regressor__mlpregressor__hidden_layer_sizes": [(64, 64), (128, 64), (128, 128, 64), (256, 128, 64)],
            "regressor__mlpregressor__activation": ["relu", "tanh"],
            "regressor__mlpregressor__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "regressor__mlpregressor__learning_rate_init": [1e-3, 3e-3, 1e-2],
        },
        n_iter=15, cv=5, scoring="r2", random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    print("  ANN best params:", search.best_params_)
    return search.best_estimator_


# ---------------------------------------------------------------------------
def parity_plot(results, y_te, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    lo, hi = y_te.min(), y_te.max()
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        ax.scatter(y_te, r["pred"], s=10, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        m = r["test"]
        ax.set_title(f"{name}: R²={m['R2']:.3f}, RMSE={m['RMSE']:.2f}")
        ax.set_xlabel("SCAPS PCE (%)"); ax.set_ylabel("Predicted PCE (%)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def shap_analysis(model, X_te, path):
    try:
        import shap
    except ImportError:
        print("  shap not installed - skipping SHAP plot (pip install shap)")
        return
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    shap.summary_plot(sv, X_te, show=False)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=X_te.columns).sort_values(ascending=False)
    print("  Mean |SHAP| (feature importance):\n" + imp.round(3).to_string())


def optimise(model, X_all, path):
    """Global search (differential evolution) inside the sampled design space.
    The optimum is only trustworthy inside the training ranges: verify it in SCAPS."""
    bounds = [(X_all[c].min(), X_all[c].max()) for c in FEATURES]
    f = lambda x: -float(model.predict(pd.DataFrame([x], columns=FEATURES))[0])
    res = differential_evolution(f, bounds, seed=SEED, maxiter=200, tol=1e-6, polish=True)
    best = {c: (10 ** v if c in LOG_FEATURES else v) for c, v in zip(FEATURES, res.x)}
    best["predicted_PCE"] = -res.fun
    with open(path, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"  Optimum predicted PCE = {-res.fun:.2f} %  -> saved to {path}")
    for k, v in best.items():
        print(f"    {k:15s} {v:.4g}")
    return best


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.csv")
    ap.add_argument("--target", default="PCE_4T")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    df = make_demo_data() if a.demo else pd.read_csv(a.data)
    if a.demo:
        print("DEMO MODE: synthetic toy data, results have NO physical meaning.")
    X, y = to_X(df), df[a.target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train {len(X_tr)} / Test {len(X_te)} samples, target = {a.target}")

    results = {}
    for name, builder in [("XGBoost", build_xgb), ("ANN", build_ann)]:
        print(f"\nTraining {name} ...")
        model = builder(X_tr, y_tr)
        cv = cross_val_score(model, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
        pred = model.predict(X_te)
        results[name] = {"model": model, "pred": pred, "cv_r2": cv, "test": metrics(y_te, pred)}
        print(f"  CV R2 = {cv.mean():.4f} ± {cv.std():.4f}")
        print("  Test: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name]["test"].items()))

    # Simple stacking: average of both models
    avg = (results["XGBoost"]["pred"] + results["ANN"]["pred"]) / 2
    print("\nAverage ensemble test: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics(y_te, avg).items()))

    table = pd.DataFrame({n: {**r["test"], "CV_R2_mean": r["cv_r2"].mean()} for n, r in results.items()}).T
    table.to_csv("model_comparison.csv")
    print("\n" + table.round(4).to_string())

    parity_plot(results, y_te, "parity_plot.png")
    shap_analysis(results["XGBoost"]["model"], X_te, "shap_summary.png")

    best_name = max(results, key=lambda n: results[n]["test"]["R2"])
    print(f"\nBest model: {best_name}. Searching optimum design ...")
    optimise(results[best_name]["model"], X, "optimum_design.json")


if __name__ == "__main__":
    main()
