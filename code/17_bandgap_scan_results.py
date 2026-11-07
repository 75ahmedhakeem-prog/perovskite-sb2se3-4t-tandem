"""
17_bandgap_scan_results.py
Read the filled run table of the bandgap scan (bandgap_scan_runs.xlsx, sheets 'Top cell' and
'Bottom cell'), compute PCE against 100 mW cm-2 (Eq. 5) and the 4T efficiency (Eq. 6),
check the scan against the calibrated and measured devices, and draw the figure.
Optional: --top-iv / --bottom-iv read SCAPS-1D batch .iv files instead of the typed values
(columns are matched by run order).

Usage
  python 17_bandgap_scan_results.py --xlsx bandgap_scan_runs.xlsx --out egscan_out
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import tstyle  # noqa: F401  (minor ticks, tick padding; optional)
except ImportError:
    pass

CHECKS = {  # reference values: (source, Voc V, Jsc mA cm-2, FF %, PCE %)
    "cal_1.60": ("calibrated SCAPS-1D, 1.60 eV", 1.129, 18.94, 67.65, 14.46),
    "meas_1.60": ("measured 1.60 eV top cell [19]", 1.13, 19.01, 67.59, 14.53),
    "design_1.68": ("1.68 eV design, Table 4", 1.130, 16.94, 69.12, 13.23),
    "meas_1.77": ("measured 1.77 eV top cell [19]", 1.16, None, None, 13.0),
}
SCEN = [  # name, HTL case, stack, Nt, T_par
    ("Calibrated (0.5/0.33 µm, T$_{par}$ = 0.67)", "P", "S1", "1e+15", 0.67),
    ("Design (1.0/0.875 µm, T$_{par}$ = 0.67)", "P", "S2", "1e+15", 0.67),
    ("Improved (N$_t$ = 10$^{14}$, T$_{par}$ = 0.90)", "P", "S2", "1e+14", 0.90),
    ("Improved, aligned HTL", "A", "S2", "1e+14", 0.90),
]

def load(xlsx):
    top = pd.read_excel(xlsx, "Top cell"); bot = pd.read_excel(xlsx, "Bottom cell")
    for df in (top, bot):
        df.rename(columns={"Voc (V)": "Voc", "Jsc (mA/cm2)": "Jsc", "FF (%)": "FF"}, inplace=True)
        df["PCE"] = df.Voc * df.Jsc * df.FF / 100.0
    top["stack"] = np.where(top["d PVK (um)"] < 0.75, "S1", "S2")
    top["Nt"] = ["1e+%d" % round(np.log10(float(v))) for v in top["Nt PVK (cm-3)"]]
    bot["stack"] = np.where(bot["d PVK (um)"] < 0.75, "S1", "S2")
    return top, bot

def from_iv(df, path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    sp = importlib.util.spec_from_file_location("p07", os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_parse_batch.py"))
    p07 = importlib.util.module_from_spec(sp); sp.loader.exec_module(p07)
    iv = p07.parse(path)
    assert len(iv) == len(df), f"{path}: {len(iv)} runs, table has {len(df)}"
    for k in ("Voc", "Jsc", "FF"): df[k] = iv[k].values
    df["PCE"] = df.Voc * df.Jsc * df.FF / 100.0
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True); ap.add_argument("--out", default="egscan_out")
    ap.add_argument("--top-iv"); ap.add_argument("--bottom-iv")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    top, bot = load(a.xlsx)
    if a.top_iv: top = from_iv(top, a.top_iv)
    if a.bottom_iv: bot = from_iv(bot, a.bottom_iv)
    miss = top.PCE.isna().sum() + bot.PCE.isna().sum()
    if miss: print(f"warning: {miss} runs without results; they are skipped")
    EG = "Eg PVK (eV)"
    res = {"scenarios": {}, "checks": {}}
    # 4T scenarios
    curves = {}
    for name, case, st, nt, tp in SCEN:
        t = top[(top["HTL case"] == case) & (top["stack"] == st) & (top.Nt == nt)].set_index(EG)
        b = bot[(bot["stack"] == st) & (np.isclose(bot.T_par, tp))].set_index(EG)
        eg = sorted(set(t.index) & set(b.index))
        c = pd.DataFrame({"top": t.loc[eg, "PCE"], "bot": b.loc[eg, "PCE"]}).dropna()
        c["4T"] = c.top + c.bot; curves[name] = c
        if len(c):
            k = c["4T"].idxmax()
            res["scenarios"][name] = {"Eg_opt": float(k), "PCE_4T_opt": float(c.loc[k, "4T"]),
                                      "PCE_4T_1.68": float(c["4T"].get(1.68, np.nan)),
                                      "curve": {f"{e:.2f}": [round(r.top, 3), round(r.bot, 3), round(r["4T"], 3)] for e, r in c.iterrows()}}
    # checks against the calibrated and measured devices (case P, stack S1, Nt 1e15)
    ref = top[(top["HTL case"] == "P") & (top["stack"] == "S1") & (top.Nt == "1e+15")].set_index(EG)
    alg = top[(top["HTL case"] == "A") & (top.Nt == "1e+15")].set_index(EG)
    for key, (src, voc, jsc, ff, pce) in CHECKS.items():
        eg = float(key.split("_")[1])
        if eg in ref.index and not np.isnan(ref.loc[eg, "PCE"]):
            r = ref.loc[eg]
            res["checks"][key] = {"source": src, "model": [round(r.Voc, 4), round(r.Jsc, 3), round(r.FF, 3), round(r.PCE, 3)],
                                  "dev_%": [None if v is None else round(100 * (m / v - 1), 2) for m, v in zip([r.Voc, r.Jsc, r.FF, r.PCE], [voc, jsc, ff, pce])]}
    if 1.77 in alg.index:
        res["checks"]["aligned_1.77_Voc_vs_meas_%"] = round(100 * (alg.loc[1.77, "Voc"] / 1.16 - 1), 2)
    # bottom-cell check: 1.68 eV, stack S1, T_par 0.67 -> 1.42% in the manuscript
    bb = bot[(bot["stack"] == "S1") & np.isclose(bot.T_par, 0.67)].set_index(EG)
    if 1.68 in bb.index: res["checks"]["bottom_1.68_S1_T067"] = {"model_PCE": round(bb.loc[1.68, "PCE"], 3), "manuscript": 1.42}
    json.dump(res, open(os.path.join(a.out, "egscan_results.json"), "w"), indent=1, default=float)
    top.to_csv(os.path.join(a.out, "egscan_top.csv"), index=False); bot.to_csv(os.path.join(a.out, "egscan_bottom.csv"), index=False)
    fig(top, bot, curves, a.out)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float)[:4000])

def fig(top, bot, curves, out):
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 16.9, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.labelsize": 18.2, "axes.linewidth": 1.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3,
        "ytick.minor.size": 3, "xtick.major.width": 1.4, "ytick.major.width": 1.4, "legend.frameon": False,
        "legend.fontsize": 13.0, "lines.linewidth": 2.0, "lines.markersize": 7, "savefig.dpi": 600,
        "savefig.bbox": "tight", "mathtext.default": "regular"})
    BLUE, RED, GREEN, PURPLE, ORANGE, BLACK, GREY = "#1F4FB4", "#D62728", "#2CA02C", "#6A3D9A", "#E67E22", "#000000", "#7F7F7F"
    EG = "Eg PVK (eV)"
    f, ax = plt.subplots(2, 2, figsize=(13.2, 10.6))
    a_, b_, c_, d_ = ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1]   # (a) bottom-left, (b) bottom-right, (c) above (a), (d) above (b)
    def lab(axx, s): axx.text(0.04, 0.96, s, transform=axx.transAxes, ha="left", va="top", fontsize=19.5, fontweight="bold")
    def sel(case, st, nt):
        return top[(top["HTL case"] == case) & (top["stack"] == st) & (top.Nt == nt)].sort_values(EG)
    # (a) top-cell Voc
    for (case, st), col, mk, ls, nm in [(("P", "S1"), BLUE, "o", "--", "Pinned, 0.5 µm"), (("P", "S2"), BLUE, "o", "-", "Pinned, 1.0 µm"),
                                         (("A", "S2"), RED, "s", "-", "Aligned, 1.0 µm")]:
        t = sel(case, st, "1e+15"); a_.plot(t[EG], t.Voc, ls, marker=mk, color=col, label=nm)
    a_.plot([1.60, 1.77], [1.13, 1.16], "*", ms=16, color=BLACK, label="Measured [19]")
    a_.set_ylim(1.0, 1.5); a_.set_ylabel("Top-cell V$_{oc}$ (V)")
    a_.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0)); lab(a_, "(a)")
    # (b) top-cell PCE
    for (case, st, nt), col, mk, ls, nm in [(("P", "S1", "1e+15"), BLUE, "o", "--", "Pinned, 0.5 µm"),
                                            (("P", "S2", "1e+15"), BLUE, "o", "-", "Pinned, 1.0 µm"),
                                            (("P", "S2", "1e+14"), GREEN, "^", "-", "Pinned, 1.0 µm, N$_t$ = 10$^{14}$"),
                                            (("A", "S2", "1e+15"), RED, "s", "-", "Aligned, 1.0 µm"),
                                            (("A", "S2", "1e+14"), ORANGE, "D", "-", "Aligned, 1.0 µm, N$_t$ = 10$^{14}$")]:
        t = sel(case, st, nt); b_.plot(t[EG], t.PCE, ls, marker=mk, color=col, label=nm)
    b_.plot([1.60, 1.77], [14.53, 13.0], "*", ms=16, color=BLACK)
    b_.set_ylim(6, 19); b_.set_ylabel("Top-cell PCE (%)")
    b_.legend(loc="lower left", fontsize=12.3); lab(b_, "(b)")
    # (c) bottom-cell PCE
    for (st, tp), col, mk, ls, nm in [(("S1", 0.67), PURPLE, "o", "--", "0.5/0.33 µm, T$_{par}$ = 0.67"), (("S1", 0.90), ORANGE, "s", "--", "0.5/0.33 µm, T$_{par}$ = 0.90"),
                                      (("S2", 0.67), PURPLE, "o", "-", "1.0/0.875 µm, T$_{par}$ = 0.67"), (("S2", 0.90), ORANGE, "s", "-", "1.0/0.875 µm, T$_{par}$ = 0.90")]:
        b = bot[(bot["stack"] == st) & np.isclose(bot.T_par, tp)].sort_values(EG)
        c_.plot(b[EG], b.PCE, ls, marker=mk, color=col, label=nm)
    c_.set_ylim(0.6, 3.6); c_.set_ylabel("Bottom-cell PCE (%)")
    c_.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=12.3); lab(c_, "(c)")
    # (d) 4T
    for (name, c), col, mk in zip(curves.items(), [PURPLE, BLUE, GREEN, RED], ["o", "o", "^", "s"]):
        if not len(c): continue
        d_.plot(c.index, c["4T"], "-", marker=mk, color=col, label=name)
        k = c["4T"].idxmax(); d_.plot(k, c.loc[k, "4T"], "*", ms=18, color=col, mec=BLACK, mew=1.0, zorder=5)
    d_.set_ylim(10, 21); d_.set_ylabel("4T PCE (%)")
    d_.legend(loc="lower left", fontsize=12.3); lab(d_, "(d)")
    for axx in (a_, b_, c_, d_):
        axx.set_xlim(1.52, 1.88); axx.set_xticks([1.55, 1.65, 1.75, 1.85]); axx.set_xlabel("Perovskite bandgap (eV)")
    f.tight_layout(h_pad=1.2, w_pad=1.4); f.savefig(os.path.join(out, "fig_egscan.png")); plt.close(f)

if __name__ == "__main__":
    main()
