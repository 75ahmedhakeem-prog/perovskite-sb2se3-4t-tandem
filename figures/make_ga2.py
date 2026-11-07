"""Graphical abstract v2: calibrated model -> ML interpolation test -> bandgap / HTL alignment result."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({"font.family": "Liberation Sans", "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 2.0, "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 600, "mathtext.default": "regular"})
BLUE, RED, GREEN, ORANGE, BLACK = "#1F4FB4", "#D62728", "#2CA02C", "#E67E22", "#000000"
r = json.load(open("egscan/out/egscan_results.json"))["scenarios"]
pin = [k for k in r if k.startswith("Improved (")][0]; ali = "Improved, aligned HTL"
fig = plt.figure(figsize=(13.0, 5.2))
# panel 1: device
ax = fig.add_axes([0.0, 0.0, 0.29, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(5, 9.55, "Calibrated SCAPS-1D", ha="center", va="top", fontsize=21)
def layer(y, h, c, t, fs=15):
    ax.add_patch(Rectangle((1.4, y), 7.0, h, fc=c, ec=BLACK, lw=1.6)); ax.text(4.9, y + h / 2, t, ha="center", va="center", fontsize=fs)
layer(5.75, 1.25, "#F6C27A", "Perovskite 1.55–1.85 eV", 14.5); layer(5.25, 0.5, "#B7E1A1", "HTL / ITO", 12)
layer(1.85, 1.25, "#C9A27E", "Sb$_2$Se$_3$ 1.2 eV", 15.5); layer(1.35, 0.5, "#E6C84F", "Au", 12)
for x in [2.5, 4.9, 7.3]:
    ax.add_patch(FancyArrowPatch((x, 8.25), (x, 7.05), arrowstyle="-|>", mutation_scale=22, color=BLUE, lw=3))
    ax.add_patch(FancyArrowPatch((x, 5.15), (x, 3.15), arrowstyle="-|>", mutation_scale=22, color=RED, lw=3))
ax.text(5, 0.55, "Error vs. experiment < 3%", ha="center", fontsize=16, color=BLUE)
for x0 in [0.287, 0.612]:
    fig.patches.append(FancyArrowPatch((x0, 0.47), (x0 + 0.04, 0.47), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=34, color=BLACK, lw=3.5))
# panel 2: hold-out R2
b = fig.add_axes([0.395, 0.17, 0.19, 0.60])
b.bar([0, 1], [0.823, 0.989], 0.6, color=[BLUE, RED], edgecolor=BLACK, lw=1.8)
for x, v in [(0, 0.823), (1, 0.989)]: b.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=18)
b.set_xticks([0, 1]); b.set_xticklabels(["XGBoost", "ANN"], fontsize=18); b.set_ylim(0.6, 1.08); b.set_yticks([0.6, 0.8, 1.0])
b.yaxis.set_minor_locator(AutoMinorLocator(2)); b.tick_params(which="minor", length=3.5, width=1.4)
b.tick_params(labelsize=16, width=1.8, length=6, pad=3); b.tick_params(axis="x", length=0)
b.set_ylabel("Hold-out R$^2$", fontsize=18, labelpad=2)
fig.text(0.49, 0.955, "ML surrogate", ha="center", va="top", fontsize=21)
fig.text(0.49, 0.86, "interpolation test", ha="center", va="top", fontsize=16, color="#444444")
# panel 3: bandgap scan
c = fig.add_axes([0.735, 0.17, 0.255, 0.60])
for key, col, mk, lab, y in [(ali, RED, "s", "Aligned HTL", 0.90), (pin, GREEN, "^", "Pinned HTL", 0.10)]:
    cur = r[key]["curve"]; e = [float(k) for k in cur]; v = [cur[k][2] for k in cur]
    c.plot(e, v, "-", marker=mk, color=col, lw=2.6, ms=8)
k = r[ali]; c.plot(k["Eg_opt"], k["PCE_4T_opt"], "*", ms=22, color=RED, mec=BLACK, mew=1.2, zorder=5)
c.text(1.735, 18.98, f"{k['PCE_4T_opt']:.1f}%", ha="left", va="bottom", fontsize=17)
c.text(1.545, 20.2, "Aligned HTL", color=RED, fontsize=15.5, ha="left", va="top")
c.text(1.545, 14.75, "Pinned HTL", color=GREEN, fontsize=15.5, ha="left", va="bottom")
c.set_xlim(1.52, 1.88); c.set_ylim(14.5, 20.5); c.set_xticks([1.55, 1.65, 1.75, 1.85]); c.set_yticks([15, 17, 19])
c.xaxis.set_minor_locator(AutoMinorLocator(2)); c.yaxis.set_minor_locator(AutoMinorLocator(2))
c.tick_params(labelsize=15, width=1.8, length=6, pad=3); c.tick_params(which="minor", length=3.5, width=1.4)
c.set_xlabel("Perovskite E$_g$ (eV)", fontsize=17, labelpad=2); c.set_ylabel("PCE$_{4T}$ (%)", fontsize=18, labelpad=2)
fig.text(0.862, 0.955, "Bandgap design", ha="center", va="top", fontsize=21)
fig.text(0.862, 0.86, "needs matched HTL", ha="center", va="top", fontsize=16, color="#444444")
fig.savefig("figs/graphical_abstract_v2.png", dpi=300); fig.savefig("figs/graphical_abstract_v2.pdf"); plt.close(fig); print("ok")
