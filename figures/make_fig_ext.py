"""Fig. 14: extended thickness range and improved-top-cell scenarios (Origin style, 600 dpi)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tstyle  # noqa: F401
import numpy as np
from matplotlib.ticker import AutoMinorLocator
exec(open("make_fig_ml.py").read().split("# ---------- graphical abstract")[0])
z = np.load("ml/ext_design_maps.npz"); dp = z["dp"]
fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.4))
a = ax[0]
names = ["Calibrated design", r"+ T$_{par}$ = 0.90", r"+ N$_{t,PVK}$ = 10$^{14}$ cm$^{-3}$"]
cols = [BLUE, GREEN, ORANGE]; scaps = [15.630, 16.233, 17.422]
for k in range(3):
    a.plot(dp, z[f"m{k}"].max(1), "-", color=cols[k], label=names[k])
    a.plot(1.0, scaps[k], "s", color=cols[k], mfc="white", mew=1.8, ms=9)
a.plot(0.70, 15.290, "o", color=BLUE, mfc="white", mew=1.8, ms=9)
a.axvline(0.70, ymin=0.22, color=GREY, ls="--", lw=1.4); a.text(0.685, 17.72, "previous limit", ha="right", fontsize=13.0, color=GREY)
a.plot([], [], "s", color=BLACK, mfc="white", mew=1.8, ms=8, label="SCAPS-1D check")
a.set_xlabel(r"Perovskite thickness d$_{PVK}$ (µm)"); a.set_ylabel(r"Max. PCE$_{4T}$ (%)")
a.set_xlim(0.28, 1.03); a.set_ylim(13.2, 18.0)
a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
a.legend(loc="lower right", ncol=2, fontsize=12.0, columnspacing=0.8, handlelength=1.5); lab(a, "(a)")
b = ax[1]
steps = [("Design\n0.70 µm", 15.290), ("d$_{PVK}$\n1.0 µm", 15.630), ("T$_{par}$\n0.90", 16.233), ("N$_{t,PVK}$\n10$^{14}$", 17.422)]
x = np.arange(4); prev = 14.0
for i, (s, v) in enumerate(steps):
    base = 14.0 if i == 0 else steps[i - 1][1]
    b.bar(i, v - base, bottom=base, width=0.6, color=[BLUE, PURPLE, GREEN, ORANGE][i], edgecolor=BLACK, lw=1.4)
    b.text(i, v + 0.07, f"{v:.2f}" if i == 0 else f"+{v - base:.2f}", ha="center", va="bottom", fontsize=14.3, fontweight="bold")
    if i < 3: b.plot([i + 0.3, i + 0.7], [v, v], color=BLACK, lw=1.2, ls=":")
b.bar(4, 17.422 - 14.0, bottom=14.0, width=0.6, color="white", edgecolor=BLACK, lw=1.6, hatch="//")
b.text(4, 17.49, "17.42", ha="center", va="bottom", fontsize=14.3, fontweight="bold")
b.set_xticks(list(x) + [4]); b.set_xticklabels([s for s, _ in steps] + ["Improved\ntandem"], fontsize=13.7)
b.set_ylabel(r"PCE$_{4T}$ (SCAPS-1D, %)"); b.set_ylim(14.0, 18.0); b.set_xlim(-0.55, 4.55)
b.yaxis.set_minor_locator(AutoMinorLocator(2)); b.tick_params(axis="x", which="both", top=False); lab(b, "(b)")
fig.tight_layout(); fig.savefig("figs/fig_ext_design.png"); plt.close(fig)
