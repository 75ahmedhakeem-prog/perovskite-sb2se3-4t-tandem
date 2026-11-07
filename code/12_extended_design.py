"""
12_extended_design.py
ANN design on the extended dataset (perovskite 0.3-1.0 um) and the improved-top-cell scenario.
 - ANN (256-128-64, tanh) trained on tandem_dataset_v3.csv for PCE_top and PCE_bot
 - accuracy: random 80/20 split and hold-out of the new 0.85 um level
 - design maps at fixed defects; T_par scaling of the bottom cell calibrated on Stage D:
     Jsc ~ T_par,  Voc + n kT/q ln(T/T0),  FF + k_FF ln(T/T0)
"""
import os
import importlib.util, json, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
spec = importlib.util.spec_from_file_location("t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_train_tandem.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

def ann():
    return t.TransformedTargetRegressor(regressor=t.make_pipeline(t.StandardScaler(), t.MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-3, max_iter=4000,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=60, random_state=t.SEED)), transformer=t.StandardScaler())

df = pd.read_csv("tandem_dataset_v3.csv"); X = t.to_X(df)
acc = {}
rng = np.random.RandomState(42); idx = rng.permutation(len(df)); ntr = int(.8 * len(df))
for tg in ["PCE_top", "PCE_bot"]:
    y = df[tg].values
    m = ann().fit(X.iloc[idx[:ntr]], y[idx[:ntr]]); p = m.predict(X.iloc[idx[ntr:]])
    acc[f"{tg}_random_R2"] = r2_score(y[idx[ntr:]], p); acc[f"{tg}_random_MAE"] = mean_absolute_error(y[idx[ntr:]], p)
    h = np.isclose(df.d_PVK_um, 0.85)
    m = ann().fit(X[~h], y[~h]); p = m.predict(X[h])
    acc[f"{tg}_holdout085_R2"] = r2_score(y[h], p); acc[f"{tg}_holdout085_MAE"] = mean_absolute_error(y[h], p)
print(json.dumps(acc, indent=1))

M = {tg: ann().fit(X, df[tg].values) for tg in ["PCE_top", "PCE_bot"]}
nkT = (0.331261 - 0.324105) / np.log(0.79 / 0.67); kFF = (56.2593 - 55.6204) / np.log(0.79 / 0.67)
def g(T, V=0.324105, F=55.6204):
    r = T / 0.67; return r * (V + nkT * np.log(r)) / V * (F + kFF * np.log(r)) / F
dp = np.linspace(0.3, 1.0, 141); ds = np.linspace(0.2, 1.0, 161); P, S = np.meshgrid(dp, ds, indexing="ij")
def grid(NtP, NitP, NtS=1e16, NitC=2.85e10):
    G = pd.DataFrame({"d_PVK_um": P.ravel(), "Nt_PVK": np.log10(NtP), "Nit_PVK_Spiro": np.log10(NitP), "d_SbSe_um": S.ravel(),
                      "Nt_SbSe": np.log10(NtS), "Nit_CdS_SbSe": np.log10(NitC)})
    return M["PCE_top"].predict(G).reshape(P.shape), M["PCE_bot"].predict(G).reshape(P.shape)
scen = [("Calibrated design", 1e15, 0.67), ("T_par = 0.90", 1e15, 0.90), ("T_par = 0.90 + N_t,PVK = 1e14", 1e14, 0.90)]
out = {"accuracy": acc, "nkT": nkT, "kFF": kFF, "g090": g(0.9), "scenarios": []}; maps = {}
for name, NtP, T in scen:
    pt, pb = grid(NtP, 1e9); pb = pb * g(T); tot = pt + pb
    i, j = np.unravel_index(tot.argmax(), P.shape)
    out["scenarios"].append({"name": name, "Nt_PVK": NtP, "T_par": T, "d_PVK_um": float(dp[i]), "d_SbSe_um": float(ds[j]),
                             "PCE_top": float(pt[i, j]), "PCE_bot": float(pb[i, j]), "PCE_4T": float(tot[i, j]),
                             "PCE_4T_at_d07": float(tot[np.argmin(abs(dp - .7)), :].max())})
    maps[name] = tot
json.dump(out, open("ext_design.json", "w"), indent=1)
np.savez("ext_design_maps.npz", dp=dp, ds=ds, **{f"m{k}": v for k, v in enumerate(maps.values())})
print(json.dumps(out["scenarios"], indent=1)); print("g(0.9)=", g(0.9))
