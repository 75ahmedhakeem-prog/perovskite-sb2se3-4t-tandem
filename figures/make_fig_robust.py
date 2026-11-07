"""Fig. 15: transfer-matrix optics and Monte Carlo uncertainty (Origin style; (a),(b) bottom row, (c),(d) top row)."""
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

tm = json.load(open("ml/tmm_results.json")); sp = np.load("ml/tmm_spectra.npz"); uq = json.load(open("ml/uq_results.json"))
fig, axs = plt.subplots(2, 2, figsize=(12.6, 9.6))
a, b, c, d = axs[1, 0], axs[1, 1], axs[0, 0], axs[0, 1]

# (a) transmittance spectra
lam = sp["lam"]
for Eg, col in ((1.60, BLUE), (1.68, GREEN), (1.77, RED)):
    a.plot(lam, sp[f"TMM_{Eg}"], color=col, lw=2.0, label=f"E$_g$ = {Eg:.2f} eV")
    a.plot(lam, sp[f"BL_{Eg}"], color=col, lw=1.6, ls="--")
a.plot([], [], color=BLACK, ls="--", lw=1.6, label="Beer–Lambert, T$_{par}$ = 1")
a.set_xlim(500, 1200); a.set_ylim(0, 1.1); a.set_yticks(np.arange(0, 1.01, 0.2)); a._xpad = 10
a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Transmittance")
a.legend(loc="lower right", fontsize=12.5, handlelength=1.4, borderaxespad=0.3)
lab(a, "(a)")

# (b) effective T_par
sens = [v["Tpar_1.68"] for v in tm["sensitivity"].values()]
nom = tm["Eg1.68"]["Tpar_eff"]; ub = tm["upper_bounds"]
items = [("Fitted\n1.60 eV", 0.67, None, BLUE), ("Inferred\n1.77 eV", 0.79, None, RED),
         ("TMM\nITO", nom, (min(sens), max(sens)), GREEN), ("TMM\nITO + AR", ub["ITO_plus_AR"], None, GREEN),
         ("TMM\nlow-loss\nTCO", ub["lossless_TCO"], None, ORANGE), ("TMM\nlow-loss\nTCO + AR", ub["lossless_TCO_plus_AR"], None, ORANGE)]
for i, (name, v, rng, col) in enumerate(items):
    b.bar(i, v, width=0.6, color=col, edgecolor=BLACK, lw=1.2)
    if rng: b.errorbar(i, v, yerr=[[v - rng[0]], [rng[1] - v]], color=BLACK, capsize=6, lw=1.6)
    b.text(i, (rng[1] if rng else v) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=13.5)
b.axhline(0.90, color=GREY, ls=":", lw=1.6); b.text(-0.45, 0.905, "scenario T$_{par}$ = 0.90", ha="left", va="bottom", fontsize=12.5, color=GREY)
b.set_xticks(range(len(items))); b.set_xticklabels([n for n, *_ in items], fontsize=11.8)
b.tick_params(axis="x", length=0)
b.set_ylim(0.5, 1.08); b.set_xlim(-0.6, 5.6); b.set_ylabel("Effective T$_{par}$ (1.68 eV)")
lab(b, "(b)")

# (c) Monte Carlo distributions
names = list(uq.keys()); cols = [BLUE, PURPLE, ORANGE]; short = ["0.70 µm design", "1.0 µm design", "Improved top cell"]
bins = np.arange(13.4, 18.3, 0.08)
for k, (nm, col, sh) in enumerate(zip(names, cols, short)):
    x = np.load(f"ml/uq_{k}.npy")
    h, e = np.histogram(x, bins=bins, density=True)
    c.step(e[:-1], h, where="post", color=col, lw=2.0, label=f"{sh}: {uq[nm]['median']:.2f}% ({uq[nm]['p2.5']:.1f}–{uq[nm]['p97.5']:.1f})")
for v, col in zip((15.29, 15.63, 17.42), cols):
    c.axvline(v, ymax=0.58, color=col, ls="--", lw=1.4)
c.set_xlim(13.4, 18.3); c.set_ylim(0, 2.1)
c.set_xlabel("PCE$_{4T}$ (%)"); c.set_ylabel("Probability density (%$^{-1}$)")
c.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=12.3, handlelength=1.2)
lab(c, "(c)")

# (d) variance shares (SRC^2)
keys = ["Nt_PVK", "Nit_PVK_Spiro", "Nt_SbSe", "Nit_CdS_SbSe", "T_par", "d_PVK_um"]
klab = ["N$_{t,PVK}$", "N$_{it,PVK/Spiro}$", "N$_{t,Sb_2Se_3}$", "N$_{it,CdS/Sb_2Se_3}$", "T$_{par}$", "d$_{PVK}$"]
kcol = [BLUE, GREEN, PURPLE, ORANGE, RED, GREY]
xs = np.arange(3); bottom = np.zeros(3)
for key, kl, col in zip(keys, klab, kcol):
    v = np.array([uq[nm]["SRC2"][key] for nm in names]) * 100
    d.bar(xs, v, bottom=bottom, width=0.55, color=col, edgecolor=BLACK, lw=1.0, label=kl); bottom += v
d.set_xticks(xs); d.set_xticklabels(["0.70 µm\ndesign", "1.0 µm\ndesign", "Improved\ntop cell"], fontsize=13.5)
d.tick_params(axis="x", length=0)
d.set_xlim(-0.5, 4.3); d.set_ylim(0, 112); d.set_yticks(range(0, 101, 20)); d.set_ylabel("Share of variance (%)")
d.legend(loc="center right", fontsize=12.8, handlelength=1.0, borderaxespad=0.2)
lab(d, "(d)")

fig.tight_layout(h_pad=1.2, w_pad=1.2)
fig.savefig("figs/fig_robust.png"); plt.close(fig)
print("saved")
