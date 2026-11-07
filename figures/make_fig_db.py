"""Fig. 16: detailed-balance limit of the 4T tandem (Origin style)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tstyle  # noqa: F401
import numpy as np
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 16.9, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.labelsize": 18.2, "axes.linewidth": 1.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "legend.frameon": False, "legend.fontsize": 13.7, "lines.linewidth": 2.0, "savefig.dpi": 600, "savefig.bbox": "tight"})
BLUE, RED, GREEN, BLACK, ORANGE, PURPLE, GREY = "#1F4FB4", "#D62728", "#2CA02C", "#000000", "#E67E22", "#6A3D9A", "#7F7F7F"
def lab(ax, s, x=0.04, ha="left", y=0.96, **k):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19.5, fontweight="bold", **k)

d = np.load("ml/db_curves.npz"); r = json.load(open("ml/db_results.json"))
tops, top, bot, tot = d["tops"], d["top"], d["bot"], d["tot"]
fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 5.2))

a.plot(tops, tot, color=BLACK, lw=2.4, label="4T limit")
a.plot(tops, top, color=BLUE, lw=2.0, label="Top cell")
a.plot(tops, bot, color=RED, lw=2.0, label="Sb$_2$Se$_3$ bottom cell")
cal = 0.4541 * top + 0.1043 * bot; imp = 0.5220 * top + 0.1504 * bot
a.plot(tops, imp, color=ORANGE, lw=2.0, ls="--", label="Scaled, improved")
a.plot(tops, cal, color=PURPLE, lw=2.0, ls="--", label="Scaled, calibrated")
for e in (1.60, 1.68, 1.77):
    a.axvline(e, ymin=0.0, ymax=0.08, color=GREY, lw=1.6)
k = int(np.argmax(tot)); a.plot(tops[k], tot[k], "*", ms=15, color=BLACK)
a.annotate(f"{tot[k]:.1f}% at {tops[k]:.2f} eV", (tops[k], tot[k]), textcoords="offset points", xytext=(6, 6), fontsize=13.5)
i68 = int(np.argmin(abs(tops - 1.68)))
a.plot(1.68, tot[i68], "o", ms=8, mfc="white", mec=BLACK, mew=1.6)
a.annotate(f"{tot[i68]:.1f}%", (1.68, tot[i68]), textcoords="offset points", xytext=(-8, -20), ha="right", fontsize=13.5)
a.plot(1.68, 14.65, "s", ms=8, color=PURPLE); a.plot(1.68, 17.26, "s", ms=8, color=ORANGE)
a.set_xlim(1.40, 2.30); a.set_ylim(0, 60); a.set_yticks(range(0, 51, 10))
a.set_xlabel("Top-cell bandgap (eV)"); a.set_ylabel("Efficiency (%)")
a.legend(loc="upper right", ncol=2, fontsize=12.3, handlelength=1.6, columnspacing=0.9)
lab(a, "(a)", x=0.03, y=0.97)

E1, E2, Z = d["E1"], d["E2"], d["Z"]
cs = b.contourf(E1, E2, Z, levels=np.arange(28, 47, 2), cmap="viridis")
cl = b.contour(E1, E2, Z, levels=[38, 40, 42, 44, 46], colors="white", linewidths=1.0)
b.clabel(cl, fmt="%d%%", fontsize=12)
cb = fig.colorbar(cs, ax=b, pad=0.02); cb.set_label("4T limit (%)", labelpad=6)
b.plot(1.68, 1.20, "o", ms=10, mfc="white", mec=BLACK, mew=1.8)
b.annotate("this work", (1.68, 1.20), textcoords="offset points", xytext=(8, 6), fontsize=13.5, color="white")
go = r["global_opt"]; b.plot(go["Eg_top"], go["Eg_bot"], "*", ms=16, color="white", mec=BLACK)
b.set_xlim(1.40, 2.30); b.set_ylim(0.90, 1.50)
b.set_xlabel("Top-cell bandgap (eV)"); b.set_ylabel("Bottom-cell bandgap (eV)")
lab(b, "(b)", x=0.03, y=0.97)
fig.tight_layout(w_pad=1.0)
fig.savefig("figs/fig_db.png"); plt.close(fig)
print("saved")
