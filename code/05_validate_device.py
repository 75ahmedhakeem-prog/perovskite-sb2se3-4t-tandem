"""
05_validate_device.py
Validate a single-junction SCAPS-1D device (e.g. the perovskite top cell).

Checks performed
  1. J-V parameters (Voc, Jsc, FF, PCE) extracted from the J-V curve itself.
  2. Relative error against a benchmark (published simulation or experiment).
  3. Shockley-Queisser (SQ) limits for the absorber bandgap, computed from the
     same AM1.5G spectrum file used in SCAPS: Jsc, Voc and FF must stay below them.
  4. Voltage deficit  Eg/q - Voc  (values below ~0.3 V are suspicious).
  5. Consistency of Jsc with the EQE integrated over the AM1.5G photon flux.
It also writes Origin-style J-V and EQE figures.

Usage
  python 05_validate_device.py --iv top.iv --qe top.qe --spectrum "AM1_5G 1 sun.spe" \
         --eg 1.60 --target 1.23 21.05 75.0 19.62

  --iv / --qe accept SCAPS result files or simple two-column text/CSV files
  (V [V], J [mA/cm2]) and (wavelength [nm], EQE [%]).
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

Q, H, C, KB = 1.602176634e-19, 6.62607015e-34, 2.99792458e8, 1.380649e-23
HC_EV_NM = 1239.84193


# --------------------------------------------------------------------------- reading
def numeric_blocks(path):
    """Return the largest block of consecutive numeric rows (>= 2 columns)."""
    blocks, cur = [], []
    num = re.compile(r"^[\s]*[-+]?\d")
    with open(path, errors="ignore") as f:
        for line in f:
            parts = re.split(r"[\s,;]+", line.strip())
            if num.match(line) and len(parts) >= 2:
                try:
                    cur.append([float(p) for p in parts if p != ""])
                    continue
                except ValueError:
                    pass
            if cur:
                blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    if not blocks:
        raise ValueError(f"No numeric data found in {path}")
    best = max(blocks, key=len)
    ncol = min(len(r) for r in best)
    return np.array([r[:ncol] for r in best])


def read_iv(path, vcol=0, jcol=1):
    d = numeric_blocks(path)
    V, J = d[:, vcol], d[:, jcol]
    o = np.argsort(V); V, J = V[o], J[o]
    # SCAPS gives negative current under illumination at V = 0; use positive photocurrent
    if np.interp(0.0, V, J) < 0:
        J = -J
    return V, J


def read_qe(path, lcol=0, qcol=1):
    """SCAPS .qe columns: lambda (nm), QE (%), photon energy (eV)."""
    d = numeric_blocks(path)
    lam, eqe = d[:, lcol], d[:, qcol]
    o = np.argsort(lam); lam, eqe = lam[o], eqe[o]
    if eqe.max() <= 1.5:          # fraction -> percent
        eqe = eqe * 100
    return lam, eqe


def read_spectrum(path):
    """Return wavelength (nm) and incident power per wavelength bin (W m^-2).
    Accepts SCAPS .spe files (power per interval, summing to ~1000 W/m2) or
    spectral irradiance files (W m^-2 nm^-1, integrating to ~1000 W/m2)."""
    d = numeric_blocks(path)
    lam, p = d[:, 0], d[:, 1]
    o = np.argsort(lam); lam, p = lam[o], p[o]
    if 800 < p.sum() < 1200:          # SCAPS style: power per interval
        pbin = p
    else:                              # spectral irradiance -> power per bin
        pbin = p * np.gradient(lam)
    print(f"Spectrum: {lam.min():.0f}-{lam.max():.0f} nm, total {pbin.sum():.1f} W/m2")
    return lam, pbin


# --------------------------------------------------------------------------- physics
def jv_parameters(V, J):
    Jsc = float(np.interp(0.0, V, J))
    sign = np.where(np.diff(np.sign(J)) != 0)[0]
    if len(sign) == 0:
        raise ValueError("J-V curve does not cross J = 0: extend the voltage sweep.")
    i = sign[0]
    Voc = float(V[i] - J[i] * (V[i + 1] - V[i]) / (J[i + 1] - J[i]))
    mask = (V >= 0) & (V <= Voc)
    Vf = np.linspace(0, Voc, 2000)
    P = Vf * np.interp(Vf, V[mask], J[mask])
    k = int(np.argmax(P))
    Pmax = float(P[k])
    FF = Pmax / (Voc * Jsc) * 100
    PCE = Pmax / 100.0 * 100           # Pin = 100 mW/cm2 -> PCE (%) = Pmax (mW/cm2)
    return {"Voc": Voc, "Jsc": Jsc, "FF": FF, "PCE": PCE, "Vmp": Vf[k], "Jmp": Pmax / Vf[k]}


def photon_flux(lam, pbin):
    """Photon flux per wavelength bin (m^-2 s^-1)."""
    return pbin * lam * 1e-9 / (H * C)


def sq_limit(eg, lam, pbin, T=300.0):
    phi = photon_flux(lam, pbin)
    lam_g = HC_EV_NM / eg
    jsc = Q * phi[lam <= lam_g].sum() / 10.0                             # A/m2 -> mA/cm2
    E = np.linspace(eg, eg + 2.0, 20000) * Q
    j0 = Q * 2 * np.pi / (H ** 3 * C ** 2) * np.trapezoid(E ** 2 / np.expm1(E / (KB * T)), E) / 10.0
    voc = KB * T / Q * np.log(jsc / j0 + 1)
    v = voc / (KB * T / Q)
    ff = (v - np.log(v + 0.72)) / (v + 1) * 100
    return {"Jsc": jsc, "Voc": voc, "FF": ff, "PCE": jsc * voc * ff / 100}


def jsc_from_eqe(lam_q, eqe, lam, pbin):
    """Integrate EQE over the photon flux of the spectrum bins inside the QE range."""
    m = (lam >= lam_q.min()) & (lam <= lam_q.max())
    e = np.interp(lam[m], lam_q, eqe) / 100
    return Q * (photon_flux(lam[m], pbin[m]) * e).sum() / 10.0


def jph_outside(lam_q, lam, pbin, eg):
    """Photocurrent available (EQE = 100 %) from spectrum bins below the QE range."""
    m = (lam < lam_q.min()) & (lam <= HC_EV_NM / eg)
    return Q * photon_flux(lam[m], pbin[m]).sum() / 10.0


# --------------------------------------------------------------------------- plotting
def origin_style():
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 13, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.labelsize": 14, "axes.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4, "legend.frameon": False,
        "lines.linewidth": 2.2, "savefig.dpi": 600, "savefig.bbox": "tight"})


def plot(V, J, prm, qe, out):
    origin_style()
    n = 2 if qe is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.3))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    m = (V >= 0) & (J >= -1)
    ax.plot(V[m], J[m], color="#1F4FB4")
    ax.plot(prm["Vmp"], prm["Jmp"], "o", ms=7, mfc="none", mec="#D62728", mew=1.8)
    ax.set_xlim(0, prm["Voc"] * 1.08); ax.set_ylim(0, prm["Jsc"] * 1.15)
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel(r"Current density (mA cm$^{-2}$)")
    ax.text(0.05, 0.08, f"V$_{{oc}}$ = {prm['Voc']:.3f} V\nJ$_{{sc}}$ = {prm['Jsc']:.2f} mA cm$^{{-2}}$\n"
            f"FF = {prm['FF']:.1f}%\nPCE = {prm['PCE']:.2f}%", transform=ax.transAxes, fontsize=11.5)
    if n == 2:
        ax.text(0.95, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontsize=15, fontweight="bold")
        lam_q, eqe = qe
        b = axes[1]
        b.plot(lam_q, eqe, color="#2CA02C")
        b.set_xlim(lam_q.min(), lam_q.max()); b.set_ylim(0, 100)
        b.set_xlabel("Wavelength (nm)"); b.set_ylabel("EQE (%)")
        b.text(0.95, 0.95, "(b)", transform=b.transAxes, ha="right", va="top", fontsize=15, fontweight="bold")
    for a in axes:
        a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", required=True)
    ap.add_argument("--qe")
    ap.add_argument("--spectrum", help="the AM1.5G file used in SCAPS")
    ap.add_argument("--eg", type=float, required=True, help="absorber bandgap (eV)")
    ap.add_argument("--target", type=float, nargs=4, metavar=("VOC", "JSC", "FF", "PCE"))
    ap.add_argument("--tol", type=float, default=5.0, help="accepted relative error (%%)")
    ap.add_argument("--out", default="device_validation.png")
    a = ap.parse_args()

    V, J = read_iv(a.iv)
    prm = jv_parameters(V, J)
    print("\n=== Extracted J-V parameters ===")
    print(f"Voc = {prm['Voc']:.4f} V | Jsc = {prm['Jsc']:.3f} mA/cm2 | FF = {prm['FF']:.2f} % | PCE = {prm['PCE']:.3f} %")
    ok = True

    if a.target:
        print("\n=== Benchmark comparison ===")
        for k, t in zip(["Voc", "Jsc", "FF", "PCE"], a.target):
            err = (prm[k] - t) / t * 100
            flag = "PASS" if abs(err) <= a.tol else "CHECK"
            ok &= flag == "PASS"
            print(f"{k:4s}: simulated {prm[k]:8.3f} | target {t:8.3f} | error {err:+6.2f} %  [{flag}]")

    vdef = a.eg - prm["Voc"]
    print(f"\nVoltage deficit Eg/q - Voc = {vdef:.3f} V  "
          f"[{'PASS' if vdef >= 0.3 else 'CHECK: unusually small, verify defects and band offsets'}]")
    ok &= vdef >= 0.3

    qe = None
    if a.spectrum:
        lam, p = read_spectrum(a.spectrum)
        sq = sq_limit(a.eg, lam, p)
        print(f"\n=== Shockley-Queisser limits for Eg = {a.eg:.2f} eV ===")
        for k in ["Voc", "Jsc", "FF", "PCE"]:
            frac = prm[k] / sq[k] * 100
            flag = "PASS" if prm[k] < sq[k] else "FAIL: exceeds the SQ limit"
            ok &= prm[k] < sq[k]
            print(f"{k:4s}: SQ {sq[k]:8.3f} | device {prm[k]:8.3f} ({frac:5.1f} % of SQ)  [{flag}]")
        if a.qe:
            lam_q, eqe = read_qe(a.qe)
            qe = (lam_q, eqe)
            jq = jsc_from_eqe(lam_q, eqe, lam, p)
            d = (jq - prm["Jsc"]) / prm["Jsc"] * 100
            flag = "PASS" if abs(d) <= 2 else "CHECK: extend the QE range or check the spectrum file"
            ok &= abs(d) <= 2
            print(f"\nJsc from EQE = {jq:.3f} mA/cm2 vs J-V Jsc = {prm['Jsc']:.3f} ({d:+.2f} %)  [{flag}]")
            jo = jph_outside(lam_q, lam, p, a.eg)
            if jo > 0:
                print(f"  (photons below {lam_q.min():.0f} nm could add at most {jo:.3f} mA/cm2; not in the QE range)")
            edge = lam_q[np.where(eqe >= 5)[0].max()] if (eqe >= 5).any() else np.nan
            print(f"EQE cut-off (last point >= 5 %) at {edge:.0f} nm; band edge from Eg = {HC_EV_NM / a.eg:.0f} nm")
    elif a.qe:
        lam_q, eqe = read_qe(a.qe); qe = (lam_q, eqe)

    plot(V, J, prm, qe, a.out)
    print(f"\nFigure written to {a.out}")
    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS NEED ATTENTION")


if __name__ == "__main__":
    main()
