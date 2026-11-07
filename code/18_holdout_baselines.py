"""Level hold-out with ANN seed uncertainty and simple baselines (runs without XGBoost).
Baselines: training-mean, nearest level (piecewise constant, as a tree), linear interpolation
between the two neighbouring levels on the grid, global linear regression, cubic polynomial ridge."""
import os
import importlib.util, sys, time, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_train_tandem.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
df = pd.read_csv("../data/ml/tandem_dataset_v2.csv")
X, y = t.to_X(df), df["PCE_4T"].values
F = t.FEATURES
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]
def ann(seed):
    return t.TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), t.MLPRegressor(
        hidden_layer_sizes=(128, 128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-3, max_iter=4000,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=60, random_state=seed)), transformer=StandardScaler())
def grid_baselines(f, lv, levels, te):
    others = [c for c in F if c != f]
    key = X[others].round(6).astype(str).agg("|".join, axis=1)
    tab = pd.DataFrame({"k": key, "v": X[f].values, "y": y})
    piv = tab.pivot_table(index="k", columns="v", values="y")
    i = list(levels).index(lv); lo, hi = levels[i - 1], levels[i + 1]
    w = (lv - lo) / (hi - lo)
    lin = piv[lo] + w * (piv[hi] - piv[lo])
    near = piv[lo] if (lv - lo) <= (hi - lv) else piv[hi]
    k = key[te]
    return near.loc[k].values, lin.loc[k].values
rows = []
for f in F:
    levels = np.sort(X[f].unique())
    for lv in levels[1:-1]:
        te = np.isclose(X[f].values, lv); yt = y[te]
        val = 10 ** lv if f in t.LOG else lv
        preds = {}
        if SEEDS == [0]:
            near, lin = grid_baselines(f, lv, levels, te)
            preds["Mean"] = np.full(te.sum(), y[~te].mean())
            preds["Nearest level"] = near
            preds["Linear interp."] = lin
            preds["Linear regr."] = LinearRegression().fit(X[~te], y[~te]).predict(X[te])
            preds["Cubic poly."] = make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(1e-3)).fit(X[~te], y[~te]).predict(X[te])
        else:
            for s in SEEDS:
                t0 = time.time(); preds[f"ANN_s{s}"] = ann(s).fit(X[~te], y[~te]).predict(X[te])
                print(f, f"{val:.3g}", s, f"{time.time()-t0:.0f}s", flush=True)
        for name, p in preds.items():
            rows.append({"feature": f, "held_out_level": f"{val:.3g}", "model": name, **t.metrics(yt, p)})
out = "../data/ml/holdout_" + ("baselines" if SEEDS == [0] else "ann_" + "_".join(map(str, SEEDS))) + ".csv"
pd.DataFrame(rows).to_csv(out, index=False); print("saved", out)
print(pd.DataFrame(rows).groupby("model")[["R2", "RMSE", "MAE"]].mean().round(4))
