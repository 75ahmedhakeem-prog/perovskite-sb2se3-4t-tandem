"""
02_filtered_spectrum.py
Build the filtered spectrum that reaches the Sb2Se3 bottom cell in a 4T tandem:

    S_filtered(lambda) = S_AM1.5G(lambda) * T_par * exp(-alpha_pvk(lambda) * d_pvk)

alpha_pvk uses the SCAPS "sqrt model":  alpha = A * sqrt(hv - Eg)  for hv > Eg,
plus an Urbach tail below Eg. T_par is a lumped transmission of the
transparent contacts / transport layers (ITO, SnO2, HTL), typically 0.85-0.92.

Input : AM1.5G file with two columns: wavelength_nm, irradiance_W_m2_nm
        (e.g. ASTM G173 "Global tilt" column).
Output: one .spe file per sample (wavelength_nm, W/m2/nm) for SCAPS.
        Check the header/format expected by your SCAPS version (see the
        example spectrum files shipped in SCAPS 'spectrum' folder).

Usage:
    python 02_filtered_spectrum.py --am15 am15g.csv --samples samples.csv --outdir spectra
"""
import argparse
import os

import numpy as np
import pandas as pd

HC_EV_NM = 1239.84193


def alpha_sqrt(lam_nm, Eg, A=1e5, E_urbach=0.015):
    """Absorption coefficient (cm^-1): sqrt model above Eg, Urbach tail below."""
    E = HC_EV_NM / lam_nm
    alpha = np.where(E > Eg, A * np.sqrt(np.clip(E - Eg, 0, None)), 0.0)
    # Urbach tail, matched to the value at Eg + E_urbach
    a0 = A * np.sqrt(E_urbach)
    tail = a0 * np.exp((E - Eg - E_urbach) / E_urbach)
    return np.where(E > Eg + E_urbach, alpha, np.minimum(tail, a0))


def filtered_spectrum(lam, S, Eg, d_um, T_par=0.90, A=1e5):
    d_cm = d_um * 1e-4
    return S * T_par * np.exp(-alpha_sqrt(lam, Eg, A) * d_cm)


def write_spe(path, lam, S):
    with open(path, "w") as f:
        f.write("// filtered spectrum for SCAPS: lambda(nm)  power(W/m2/nm)\n")
        for l, s in zip(lam, S):
            f.write(f"{l:.1f}\t{s:.6e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--am15", required=True)
    ap.add_argument("--samples", default="samples.csv")
    ap.add_argument("--outdir", default="spectra")
    ap.add_argument("--Tpar", type=float, default=0.90)
    a = ap.parse_args()

    ref = pd.read_csv(a.am15, comment="#", header=None, names=["lam", "S"])
    ref = ref[(ref.lam >= 300) & (ref.lam <= 1300)]
    lam, S = ref.lam.values, ref.S.values

    df = pd.read_csv(a.samples)
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for r in df.itertuples(index=False):
        Sf = filtered_spectrum(lam, S, r.pvk_Eg_eV, r.pvk_thick_um, a.Tpar)
        write_spe(os.path.join(a.outdir, f"filtered_{int(r.sample_id):05d}.spe"), lam, Sf)
        rows.append((int(r.sample_id), np.trapezoid(Sf, lam)))
    pd.DataFrame(rows, columns=["sample_id", "P_filtered_W_m2"]).to_csv(
        os.path.join(a.outdir, "filtered_power.csv"), index=False)
    print(f"Wrote {len(df)} filtered spectra to {a.outdir}/  "
          f"(input power {np.trapezoid(S, lam):.1f} W/m2 in 300-1300 nm)")


if __name__ == "__main__":
    main()
