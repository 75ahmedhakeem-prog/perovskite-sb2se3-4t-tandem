"""
08_build_tandem_dataset.py
Combine the top-cell batch (AM1.5G) with the bottom-cell batches (filtered spectra)
into the 4T tandem dataset. The bottom cell depends on the top cell only through the
filtered spectrum, which is set by the perovskite thickness (Eg and T_par fixed).
    PCE_4T = PCE_top + PCE_bot      (Eq. 3; PCE_bot against 100 mW/cm2, Eq. 2)

Usage
  python 08_build_tandem_dataset.py --top top_ML.csv --bottom bottom_ML.csv --out tandem_dataset.csv
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--top", required=True)
ap.add_argument("--bottom", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

t = pd.read_csv(a.top)
b = pd.read_csv(a.bottom)
t = t.rename(columns={"thickness_um": "d_PVK_um", "Nt_bulk": "Nt_PVK", "Nt_interface": "Nit_PVK_Spiro",
                      "Voc": "Voc_top", "Jsc": "Jsc_top", "FF": "FF_top", "PCE": "PCE_top"})
b = b.rename(columns={"thickness_um": "d_SbSe_um", "Nt_bulk": "Nt_SbSe", "Nt_interface": "Nit_CdS_SbSe",
                      "Voc": "Voc_bot", "Jsc": "Jsc_bot", "FF": "FF_bot", "PCE": "PCE_bot"})
t["d_top_nm"] = (t.d_PVK_um * 1000).round().astype(int)
cols_t = ["d_PVK_um", "Nt_PVK", "Nit_PVK_Spiro", "d_top_nm", "Voc_top", "Jsc_top", "FF_top", "PCE_top"]
cols_b = ["d_SbSe_um", "Nt_SbSe", "Nit_CdS_SbSe", "d_top_nm", "Voc_bot", "Jsc_bot", "FF_bot", "PCE_bot"]
d = t[cols_t].merge(b[cols_b], on="d_top_nm").drop(columns="d_top_nm")
d["PCE_4T"] = d.PCE_top + d.PCE_bot
d.to_csv(a.out, index=False)
print(f"{len(t)} top x bottom runs -> {len(d)} tandem samples written to {a.out}")
print(d[["PCE_top", "PCE_bot", "PCE_4T"]].describe().round(2))
best = d.loc[d.PCE_4T.idxmax()]
print("Best tandem in the dataset:\n", best.round(4).to_string())
