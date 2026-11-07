"""
11_ann_design.py
Use the ANN (best interpolating model) to find the optimum layer thicknesses of the
4T tandem at fixed, realistic defect densities, and list the point to verify in SCAPS-1D.

Usage
  python 11_ann_design.py --data tandem_dataset_v2.csv --Nt-pvk 1e15 --Nit-pvk 1e9 --Nt-sbse 1e16 --Nit-cds 2.85e10
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_train_tandem.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="tandem_dataset_v2.csv")
ap.add_argument("--Nt-pvk", type=float, default=1e15)
ap.add_argument("--Nit-pvk", type=float, default=1e9)
ap.add_argument("--Nt-sbse", type=float, default=1e16)
ap.add_argument("--Nit-cds", type=float, default=2.85e10)
a = ap.parse_args()

df = pd.read_csv(a.data)
X = t.to_X(df)
models = {}
for tgt in ["PCE_top", "PCE_bot", "PCE_4T"]:
    m = t.TransformedTargetRegressor(regressor=t.make_pipeline(t.StandardScaler(), t.MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-3, max_iter=4000,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=60, random_state=t.SEED)),
        transformer=t.StandardScaler()).fit(X, df[tgt].values)
    models[tgt] = m

dp = np.linspace(0.3, 0.7, 81); ds = np.linspace(0.2, 1.0, 161)
P, S = np.meshgrid(dp, ds, indexing="ij")
G = pd.DataFrame({"d_PVK_um": P.ravel(), "Nt_PVK": np.log10(a.Nt_pvk), "Nit_PVK_Spiro": np.log10(a.Nit_pvk),
                  "d_SbSe_um": S.ravel(), "Nt_SbSe": np.log10(a.Nt_sbse), "Nit_CdS_SbSe": np.log10(a.Nit_cds)})
pred = {k: m.predict(G).reshape(P.shape) for k, m in models.items()}
i, j = np.unravel_index(pred["PCE_4T"].argmax(), P.shape)
best = {"d_PVK_um": float(dp[i]), "d_SbSe_um": float(ds[j]), "Nt_PVK": a.Nt_pvk, "Nit_PVK_Spiro": a.Nit_pvk,
        "Nt_SbSe": a.Nt_sbse, "Nit_CdS_SbSe": a.Nit_cds,
        **{f"pred_{k}": float(v[i, j]) for k, v in pred.items()}}
json.dump(best, open("ann_design_optimum.json", "w"), indent=2)
np.savez("ann_design_map.npz", dp=dp, ds=ds, **pred)
print(json.dumps(best, indent=2))
