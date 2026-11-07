"""
06_filtered_spectrum_scaps.py
Create the filtered AM1.5G spectrum that reaches the bottom cell of a 4T tandem,
directly in the SCAPS-1D .spe format (power per wavelength interval, W/m2).

    P_filt(lambda) = P_AM1.5G(lambda) * T_par * exp( -alpha_PVK(lambda) * d_PVK )

alpha_PVK follows the square-root law of SCAPS-1D 3.3.x used for the top cell,
alpha = (alpha0 + beta0*Eg/hv) * sqrt(hv/Eg - 1), with an Urbach tail below Eg.
This is the same law as the traditional form (A + B/hv)*sqrt(hv - Eg) with
A = alpha0/sqrt(Eg) and B = beta0*sqrt(Eg); SCAPS shows both in its absorption panel.
T_par lumps reflection and parasitic absorption of the top-cell stack
(glass, transparent electrodes, transport layers) for sub-bandgap light.

Usage
  python 06_filtered_spectrum_scaps.py --spe "AM1_5G 1 sun.spe" --eg 1.60 --d 500 --tpar 0.30 \
         --out "filtered_1p60eV.spe"

Copy the output file to the SCAPS 'spectrum' folder and select it in the
Action panel ("Select spectrum file") when simulating the bottom cell.
SCAPS then reports eta relative to the filtered power; the bottom-cell PCE
must be recomputed against 100 mW/cm2: PCE = Voc*Jsc*FF (script 05 does this).
"""
import argparse

import numpy as np

HC_EV_NM = 1239.84193
Q, H, C = 1.602176634e-19, 6.62607015e-34, 2.99792458e8


def read_scaps_spe(path):
    lam, p, header = [], [], []
    with open(path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(">") or s.startswith("//"):
                header.append(line.rstrip("\n"))
                continue
            parts = s.replace(",", " ").split()
            try:
                lam.append(float(parts[0])); p.append(float(parts[1]))
            except (ValueError, IndexError):
                header.append(line.rstrip("\n"))
    return np.array(lam), np.array(p), header


def alpha_sqrt(lam_nm, eg, alpha0=1e5, beta0=0.0, e_urbach=0.015):
    """Absorption coefficient (cm^-1), SCAPS 3.3.x square-root law + Urbach tail."""
    E = HC_EV_NM / lam_nm
    above = (alpha0 + beta0 * eg / E) * np.sqrt(np.clip(E / eg - 1, 0, None))
    a0 = (alpha0 + beta0 * eg / (eg + e_urbach)) * np.sqrt(e_urbach / eg)
    tail = a0 * np.exp((E - eg - e_urbach) / e_urbach)
    return np.where(E > eg + e_urbach, above, np.minimum(tail, a0))


def jph(lam, pbin, lam_max=None):
    """Maximum photocurrent (mA/cm2) of the bins, optionally below lam_max."""
    m = np.ones_like(lam, bool) if lam_max is None else lam <= lam_max
    return Q * (pbin[m] * lam[m] * 1e-9 / (H * C)).sum() / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spe", required=True, help="SCAPS AM1.5G .spe file")
    ap.add_argument("--eg", type=float, required=True, help="top-cell perovskite bandgap (eV)")
    ap.add_argument("--d", type=float, default=500.0, help="perovskite thickness (nm)")
    ap.add_argument("--tpar", type=float, default=1.0, help="sub-bandgap transmission of the top-cell stack")
    ap.add_argument("--alpha0", type=float, default=1e5, help="SCAPS alpha_0 of the perovskite (1/cm)")
    ap.add_argument("--beta0", type=float, default=0.0, help="SCAPS beta_0 of the perovskite (1/cm)")
    ap.add_argument("--eg-bottom", type=float, default=1.20, help="bottom absorber bandgap for the summary")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lam, p, header = read_scaps_spe(a.spe)
    T = a.tpar * np.exp(-alpha_sqrt(lam, a.eg, a.alpha0, a.beta0) * a.d * 1e-7)
    pf = p * T
    with open(a.out, "w") as f:
        f.write(f"> Filtered AM1.5G spectrum for the bottom cell of a 4T tandem\n")
        f.write(f"> Top-cell perovskite: Eg = {a.eg:.3f} eV, d = {a.d:.0f} nm, alpha0 = {a.alpha0:.3g}, beta0 = {a.beta0:.3g}\n")
        f.write(f"> Sub-bandgap transmission of the top-cell stack T_par = {a.tpar:.4f}\n")
        f.write(f"> second column: incident power per wavelength interval (W/m2), as in the SCAPS source file\n")
        f.write(f"> total filtered power = {pf.sum():.3f} W/m2 (unfiltered {p.sum():.3f} W/m2)\n")
        f.write(">\n>lambda (nm)\tP(W/m2)\n")
        for l, v in zip(lam, pf):
            f.write(f"{l:.1f}\t{v:.5f}\n")
    lg_b = HC_EV_NM / a.eg_bottom
    print(f"Filtered power: {pf.sum():.2f} W/m2 of {p.sum():.2f} W/m2")
    print(f"Max photocurrent below {lg_b:.0f} nm: unfiltered {jph(lam, p, lg_b):.2f} mA/cm2, "
          f"filtered {jph(lam, pf, lg_b):.2f} mA/cm2")
    print(f"Written: {a.out}")


if __name__ == "__main__":
    main()
