"""
15_detailed_balance.py
Detailed-balance (Shockley–Queisser) efficiency limit of the 4T tandem versus the top-cell bandgap.
Same spectrum (SCAPS AM1.5G, 1000 W m^-2) and the same SQ method as the validation script (05):
Jsc from all photons above Eg, J0 from black-body emission above Eg (300 K), FF from Green's expression.
4T: the ideal top cell absorbs every photon above Eg,top and transmits the rest; the bottom cell
absorbs the photons between Eg,top and Eg,bot; the two sub-cells are operated independently.
"""
import json
import numpy as np

Q, H, C, KB = 1.602176634e-19, 6.62607015e-34, 2.99792458e8, 1.380649e-23
HC = 1239.84193
SPE = "../data/spectra/AM1_5G 1 sun.spe"

def spectrum():
    lam, p = [], []
    for line in open(SPE, encoding="latin-1"):
        s = line.split()
        if line.startswith(">") or len(s) < 2: continue
        try: lam.append(float(s[0])); p.append(float(s[1]))
        except ValueError: pass
    lam, p = np.array(lam), np.array(p)
    return lam, p * lam * 1e-9 / (H * C)  # photon flux per bin (m^-2 s^-1)

LAM, PHI = spectrum()

def j0(eg, T=300.0):
    E = np.linspace(eg, eg + 2.0, 20000) * Q
    return Q * 2 * np.pi / (H ** 3 * C ** 2) * np.trapezoid(E ** 2 / np.expm1(E / (KB * T)), E) / 10.0  # mA cm^-2

def cell(jsc, eg, T=300.0):
    voc = KB * T / Q * np.log(jsc / j0(eg, T) + 1)
    v = voc / (KB * T / Q); ff = (v - np.log(v + 0.72)) / (v + 1)
    return {"Jsc": jsc, "Voc": voc, "FF": ff * 100, "PCE": jsc * voc * ff}

# each spectrum bin is spread uniformly over its wavelength interval -> smooth cumulative flux
_edges = np.r_[LAM[0] - (LAM[1] - LAM[0]) / 2, (LAM[1:] + LAM[:-1]) / 2, LAM[-1] + (LAM[-1] - LAM[-2]) / 2]
_cum = np.r_[0.0, np.cumsum(PHI)]
def n_below(lam_nm):
    """photon flux with wavelength below lam_nm (m^-2 s^-1)"""
    return np.interp(lam_nm, _edges, _cum)
def jsc_band(e_hi, e_lo):
    """photons with e_lo <= E < e_hi (e_hi = inf for the top cell)"""
    lo = n_below(HC / e_hi) if np.isfinite(e_hi) else 0.0
    return Q * (n_below(HC / e_lo) - lo) / 10.0

def tandem(eg_top, eg_bot):
    t = cell(jsc_band(np.inf, eg_top), eg_top)
    b = cell(jsc_band(eg_top, eg_bot), eg_bot) if eg_bot < eg_top else {"PCE": 0.0, "Jsc": 0.0, "Voc": 0.0, "FF": 0.0}
    return t, b

if __name__ == "__main__":
    out = {}
    out["single_1.20"] = cell(jsc_band(np.inf, 1.20), 1.20)
    out["single_1.68"] = cell(jsc_band(np.inf, 1.68), 1.68)
    tops = np.round(np.arange(1.40, 2.301, 0.01), 2)
    curve = []
    for e in tops:
        t, b = tandem(e, 1.20); curve.append((e, t["PCE"], b["PCE"], t["PCE"] + b["PCE"]))
    curve = np.array(curve)
    k = int(np.argmax(curve[:, 3]))
    out["opt_top_for_1.20"] = {"Eg_top": float(curve[k, 0]), "PCE_4T": float(curve[k, 3]), "top": float(curve[k, 1]), "bot": float(curve[k, 2])}
    for e in (1.60, 1.68, 1.77):
        t, b = tandem(e, 1.20)
        out[f"Eg_top_{e:.2f}"] = {"top": t, "bot": b, "PCE_4T": t["PCE"] + b["PCE"]}
    # flatness: range of Eg_top within 1% abs of the optimum
    ok = curve[curve[:, 3] >= curve[k, 3] - 1.0, 0]
    out["within_1pct_abs"] = [float(ok.min()), float(ok.max())]
    # 2D map
    E1 = np.round(np.arange(1.40, 2.301, 0.02), 2); E2 = np.round(np.arange(0.90, 1.501, 0.02), 2)
    Z = np.full((len(E2), len(E1)), np.nan)
    for i, eb in enumerate(E2):
        for j, et in enumerate(E1):
            if eb < et:
                t, b = tandem(et, eb); Z[i, j] = t["PCE"] + b["PCE"]
    iz = np.unravel_index(np.nanargmax(Z), Z.shape)
    out["global_opt"] = {"Eg_top": float(E1[iz[1]]), "Eg_bot": float(E2[iz[0]]), "PCE_4T": float(Z[iz])}
    # fraction of the limit reached by the calibrated 1.68 eV design (SCAPS-1D values of the manuscript)
    lim = out["Eg_top_1.68"]
    out["fraction_of_limit"] = {"top_13.23": 13.23 / lim["top"]["PCE"], "bot_1.42_1.73": [1.42 / lim["bot"]["PCE"], 1.73 / lim["bot"]["PCE"]],
                                "tandem_14.65_14.96": [14.65 / lim["PCE_4T"], 14.96 / lim["PCE_4T"]], "improved_17.42": 17.42 / lim["PCE_4T"]}
    # the same fractions applied across Eg_top: where would a 'realistic' tandem peak?
    ft = out["fraction_of_limit"]["top_13.23"]; fb = 1.42 / lim["bot"]["PCE"]
    real = curve[:, 1] * ft + curve[:, 2] * fb
    kr = int(np.argmax(real)); out["realistic_scaled_opt"] = {"Eg_top": float(curve[kr, 0]), "PCE_4T": float(real[kr]), "ft": ft, "fb": fb}
    json.dump(out, open("../data/ml/db_results.json", "w"), indent=1, default=float)
    np.savez("../data/ml/db_curves.npz", tops=curve[:, 0], top=curve[:, 1], bot=curve[:, 2], tot=curve[:, 3], real=real, E1=E1, E2=E2, Z=Z)
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=lambda x: round(float(x), 3)))
