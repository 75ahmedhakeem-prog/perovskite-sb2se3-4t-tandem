"""
01_generate_samples.py
Generate design samples (Latin Hypercube) for a 4T Perovskite / Sb2Se3 tandem
and write one SCAPS-1D script file per sample.

Usage:
    python 01_generate_samples.py --n 2000 --out samples.csv --scripts scaps_scripts

IMPORTANT: layer numbers (layer1, layer2, ...) must match the order of layers
in YOUR SCAPS .def file. Check the exact script-command syntax in the SCAPS
manual (chapter "Script commands") for your SCAPS version before running.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Design space: name -> (min, max, scale)   scale = "lin" or "log"
# Thickness in micrometres (SCAPS unit), densities in cm^-3, interface in cm^-2
# ---------------------------------------------------------------------------
TOP_SPACE = {
    "pvk_thick_um":   (0.30, 1.00, "lin"),   # wide-gap perovskite absorber
    "pvk_Eg_eV":      (1.60, 1.80, "lin"),
    "pvk_NA_cm3":     (1e13, 1e17, "log"),
    "pvk_Nt_cm3":     (1e13, 1e17, "log"),
    "etl_thick_um":   (0.02, 0.10, "lin"),   # SnO2
    "htl_thick_um":   (0.05, 0.25, "lin"),   # Spiro-OMeTAD / NiOx
    "top_IL_Nt_cm2":  (1e9,  1e13, "log"),   # perovskite/ETL interface
}
BOT_SPACE = {
    "sbse_thick_um":  (0.50, 2.50, "lin"),   # Sb2Se3 absorber
    "sbse_NA_cm3":    (1e13, 1e17, "log"),
    "sbse_Nt_cm3":    (1e12, 1e16, "log"),
    "cds_thick_um":   (0.02, 0.10, "lin"),   # CdS buffer
    "bot_IL_Nt_cm2":  (1e9,  1e13, "log"),   # CdS/Sb2Se3 interface
}
SPACE = {**TOP_SPACE, **BOT_SPACE}

# Map design variables to SCAPS layer numbers in your .def files
# (edit to match your own structure!)
TOP_LAYERS = {"etl": 2, "pvk": 3, "htl": 4}   # e.g. ITO(1)/SnO2(2)/PVK(3)/HTL(4)/ITO(5)
BOT_LAYERS = {"cds": 2, "sbse": 3}            # e.g. FTO(1)/CdS(2)/Sb2Se3(3)/CuSCN(4)


def lhs_samples(space: dict, n: int, seed: int = 42) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(space), seed=seed)
    u = sampler.random(n)
    cols = {}
    for i, (name, (lo, hi, scale)) in enumerate(space.items()):
        if scale == "log":
            cols[name] = 10 ** (np.log10(lo) + u[:, i] * (np.log10(hi) - np.log10(lo)))
        else:
            cols[name] = lo + u[:, i] * (hi - lo)
    df = pd.DataFrame(cols)
    df.insert(0, "sample_id", np.arange(n))
    return df


def top_script(row, def_file="top_pvk.def"):
    L = TOP_LAYERS
    return "\n".join([
        f"load definitionfile {def_file}",
        f"set layer{L['etl']}.thickness {row.etl_thick_um:.4f}",
        f"set layer{L['pvk']}.thickness {row.pvk_thick_um:.4f}",
        f"set layer{L['pvk']}.Eg {row.pvk_Eg_eV:.4f}",
        f"set layer{L['pvk']}.Na {row.pvk_NA_cm3:.3e}",
        f"set layer{L['pvk']}.defect1.Ntotal {row.pvk_Nt_cm3:.3e}",
        f"set layer{L['htl']}.thickness {row.htl_thick_um:.4f}",
        f"set interface{L['etl']}.IFdefect1.Ntotal {row.top_IL_Nt_cm2:.3e}",
        "action iv.doiv",
        "calculate",
        f"save results.iv top_{int(row.sample_id):05d}.iv",
        "clear all",
    ])


def bottom_script(row, def_file="bottom_sbse.def",
                  spectrum="filtered_{id:05d}.spe"):
    L = BOT_LAYERS
    spe = spectrum.format(id=int(row.sample_id))
    return "\n".join([
        f"load definitionfile {def_file}",
        f"load illuminationfile {spe}",
        f"set layer{L['cds']}.thickness {row.cds_thick_um:.4f}",
        f"set layer{L['sbse']}.thickness {row.sbse_thick_um:.4f}",
        f"set layer{L['sbse']}.Na {row.sbse_NA_cm3:.3e}",
        f"set layer{L['sbse']}.defect1.Ntotal {row.sbse_Nt_cm3:.3e}",
        f"set interface{L['cds']}.IFdefect1.Ntotal {row.bot_IL_Nt_cm2:.3e}",
        "action iv.doiv",
        "calculate",
        f"save results.iv bot_{int(row.sample_id):05d}.iv",
        "clear all",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="samples.csv")
    ap.add_argument("--scripts", default="scaps_scripts")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = lhs_samples(SPACE, a.n, a.seed)
    df.to_csv(a.out, index=False)
    os.makedirs(a.scripts, exist_ok=True)
    for row in df.itertuples(index=False):
        sid = int(row.sample_id)
        with open(os.path.join(a.scripts, f"top_{sid:05d}.script"), "w") as f:
            f.write(top_script(row) + "\n")
        with open(os.path.join(a.scripts, f"bot_{sid:05d}.script"), "w") as f:
            f.write(bottom_script(row) + "\n")
    print(f"Wrote {len(df)} samples to {a.out} and {2*len(df)} scripts to {a.scripts}/")


if __name__ == "__main__":
    main()
