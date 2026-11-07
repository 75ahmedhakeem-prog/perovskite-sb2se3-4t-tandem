"""
10_level_holdout.py
Interpolation test for a grid dataset: for each feature, every INTERIOR grid level is
removed from training in turn and predicted by a model trained on the remaining levels.
This measures how well the models predict designs that lie between the sampled levels,
which a random train/test split of a grid cannot show.

Usage
  python 10_level_holdout.py --data tandem_dataset.csv --target PCE_4T
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_train_tandem.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="tandem_dataset.csv")
ap.add_argument("--target", default="PCE_4T")
a = ap.parse_args()
df = pd.read_csv(a.data)
X, y = t.to_X(df), df[a.target].values

if t.HAVE_XGB:
    tree = lambda: t.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.85,
                                  colsample_bytree=1.0, random_state=t.SEED, n_jobs=-1)
    tname = "XGBoost"
else:
    tree = lambda: t.HistGradientBoostingRegressor(max_iter=1000, max_depth=8, learning_rate=0.05,
                                                    min_samples_leaf=20, random_state=t.SEED)
    tname = "GBM (test only)"
ann = lambda: t.TransformedTargetRegressor(regressor=t.make_pipeline(t.StandardScaler(), t.MLPRegressor(
    hidden_layer_sizes=(128, 128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-3, max_iter=4000,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=60, random_state=t.SEED)),
    transformer=t.StandardScaler())

rows = []
for f in t.FEATURES:
    levels = np.sort(X[f].unique())
    for lv in levels[1:-1]:
        te = np.isclose(X[f].values, lv)
        for name, mk in [(tname, tree), ("ANN", ann)]:
            m = mk().fit(X[~te], y[~te])
            r = t.metrics(y[te], m.predict(X[te]))
            val = 10 ** lv if f in t.LOG else lv
            rows.append({"feature": f, "held_out_level": f"{val:.3g}", "model": name, **r})
            print(f"{f:14s} level {val:9.3g}  {name:16s} R2 = {r['R2']:.4f}  RMSE = {r['RMSE']:.3f}  MAE = {r['MAE']:.3f}")
res = pd.DataFrame(rows)
res.to_csv(f"level_holdout_{a.target}.csv", index=False)
print("\nMean over all held-out levels:")
print(res.groupby("model")[["R2", "RMSE", "MAE"]].mean().round(4).to_string())
