"""
14_uncertainty.py
Monte Carlo propagation of the calibration uncertainty (JCGM 101 approach) through the ANN surrogate.

For each verified design, the calibrated inputs are sampled from the distributions below; each sample is
evaluated with a randomly chosen member of a 5-network ensemble (top and bottom cell), so that the
interval contains both the input uncertainty and the surrogate (initialisation) uncertainty.

Inputs (truncated to the sampled grid range to avoid extrapolation)
  bulk defect densities   : log-uniform, calibrated value x/÷ 10^0.5
  interface defect dens.  : log-uniform, calibrated value x/÷ 2
  thicknesses             : uniform, ±10 %
  T_par (calibrated)      : uniform 0.67–0.79 (bounds inferred from the two measured top cells)
  T_par (improved)        : uniform 0.85–0.93 (TMM: low-loss TCO without / with antireflection)
Bottom-cell PCE at T_par follows Eqs. (12)–(14): PCE_bot(T) = PCE_bot(T0) * g(T).
"""
import os
import importlib.util, json
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_train_tandem.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

df = pd.read_csv("../data/ml/tandem_dataset_v3.csv"); X = t.to_X(df)
LIM = {c: (X[c].min(), X[c].max()) for c in X.columns}

def ann(seed):
    return t.TransformedTargetRegressor(regressor=t.make_pipeline(t.StandardScaler(), t.MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-3, max_iter=4000,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=60, random_state=seed)), transformer=t.StandardScaler())

SEEDS = [42, 1, 2, 3, 4]
ENS = {tg: [ann(s).fit(X, df[tg].values) for s in SEEDS] for tg in ["PCE_top", "PCE_bot"]}

nkT = (0.331261 - 0.324105) / np.log(0.79 / 0.67); kFF = (56.2593 - 55.6204) / np.log(0.79 / 0.67)
def g(T, V=0.324105, F=55.6204):
    r = T / 0.67; return r * (V + nkT * np.log(r)) / V * (F + kFF * np.log(r)) / F

DESIGNS = {
    "ANN design (0.70/0.755 um)": dict(d_PVK_um=0.70, Nt_PVK=1e15, Nit_PVK_Spiro=1.5e9, d_SbSe_um=0.755, Nt_SbSe=1e16, Nit_CdS_SbSe=2.85e10, T=(0.67, 0.79)),
    "Extended design (1.0/0.875 um)": dict(d_PVK_um=1.0, Nt_PVK=1e15, Nit_PVK_Spiro=1.5e9, d_SbSe_um=0.875, Nt_SbSe=1e16, Nit_CdS_SbSe=2.85e10, T=(0.67, 0.79)),
    "Improved top cell (1.0/0.865 um)": dict(d_PVK_um=1.0, Nt_PVK=1e14, Nit_PVK_Spiro=1.5e9, d_SbSe_um=0.865, Nt_SbSe=1e16, Nit_CdS_SbSe=2.85e10, T=(0.85, 0.93)),
}
N = 20000
rng = np.random.default_rng(2026)
out = {}
for name, d in DESIGNS.items():
    S = pd.DataFrame({
        "d_PVK_um": d["d_PVK_um"] * rng.uniform(0.9, 1.1, N),
        "Nt_PVK": np.log10(d["Nt_PVK"]) + rng.uniform(-0.5, 0.5, N),
        "Nit_PVK_Spiro": np.log10(d["Nit_PVK_Spiro"]) + rng.uniform(-np.log10(2), np.log10(2), N),
        "d_SbSe_um": d["d_SbSe_um"] * rng.uniform(0.9, 1.1, N),
        "Nt_SbSe": np.log10(d["Nt_SbSe"]) + rng.uniform(-0.5, 0.5, N),
        "Nit_CdS_SbSe": np.log10(d["Nit_CdS_SbSe"]) + rng.uniform(-np.log10(2), np.log10(2), N),
    })
    for c in S.columns: S[c] = S[c].clip(*LIM[c])
    T = rng.uniform(*d["T"], N)
    member = rng.integers(0, len(SEEDS), N)
    top = np.empty(N); bot = np.empty(N)
    for m in range(len(SEEDS)):
        k = member == m
        top[k] = ENS["PCE_top"][m].predict(S[k]); bot[k] = ENS["PCE_bot"][m].predict(S[k])
    pce = top + bot * g(T)
    # nominal (ensemble mean, central inputs, T at its lower/central value)
    nom = pd.DataFrame({c: [np.log10(d[c]) if c in t.LOG else d[c]] for c in S.columns})
    for c in nom.columns: nom[c] = nom[c].clip(*LIM[c])
    nom_top = np.mean([e.predict(nom)[0] for e in ENS["PCE_top"]]); nom_bot = np.mean([e.predict(nom)[0] for e in ENS["PCE_bot"]])
    # variance shares from standardised regression coefficients (SRC^2)
    Z = pd.concat([S, pd.Series(T, name="T_par")], axis=1)
    Zs = (Z - Z.mean()) / Z.std(); ys = (pce - pce.mean()) / pce.std()
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(N), Zs.values], ys, rcond=None)
    src2 = dict(zip(Z.columns, (beta[1:] ** 2).round(4)))
    r2_lin = 1 - np.sum((ys - np.c_[np.ones(N), Zs.values] @ beta) ** 2) / np.sum(ys ** 2)
    # surrogate-only spread at the nominal point
    ens_spread = np.std([a.predict(nom)[0] + b.predict(nom)[0] * g(np.mean(d["T"])) for a, b in zip(ENS["PCE_top"], ENS["PCE_bot"])])
    out[name] = {"median": float(np.median(pce)), "p2.5": float(np.percentile(pce, 2.5)), "p97.5": float(np.percentile(pce, 97.5)),
                 "mean": float(pce.mean()), "sd": float(pce.std()), "nominal_top": float(nom_top), "nominal_bot_T0": float(nom_bot),
                 "SRC2": src2, "SRC_R2": float(r2_lin), "ensemble_sd_nominal": float(ens_spread),
                 "P(PCE>17)": float(np.mean(pce > 17)), "P(PCE>15)": float(np.mean(pce > 15))}
    np.save(f"../data/ml/uq_{list(DESIGNS).index(name)}.npy", pce)
json.dump(out, open("../data/ml/uq_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
