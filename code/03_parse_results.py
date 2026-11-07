"""
03_parse_results.py
Read SCAPS .iv result files for the top and bottom cells, merge them with the
design samples and compute the 4T tandem efficiency.

Bottom-cell PCE is recomputed against the FULL AM1.5G power (100 mW/cm2),
because SCAPS normalises eta to the power of the (filtered) input spectrum:
    PCE(%) = Voc(V) * Jsc(mA/cm2) * FF(fraction)          [Pin = 100 mW/cm2]
    PCE_4T = PCE_top + PCE_bottom

Usage:
    python 03_parse_results.py --samples samples.csv --results scaps_results --out dataset.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
PATTERNS = {
    "Voc": re.compile(r"\bVoc\s*=\s*" + NUM),
    "Jsc": re.compile(r"\bJsc\s*=\s*" + NUM),
    "FF":  re.compile(r"\bFF\s*=\s*" + NUM),
    "eta": re.compile(r"\beta\s*=\s*" + NUM),
}


def parse_iv(path):
    """Return dict Voc[V], Jsc[mA/cm2], FF[%], eta[%] from a SCAPS .iv file.
    The regexes look for lines like 'Voc = 1.1234 V'. Open one of your own
    .iv files and adjust PATTERNS if your SCAPS version writes them differently."""
    out = {k: np.nan for k in PATTERNS}
    if not os.path.exists(path):
        return out
    with open(path, errors="ignore") as f:
        txt = f.read()
    for k, pat in PATTERNS.items():
        m = pat.search(txt)
        if m:
            out[k] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples.csv")
    ap.add_argument("--results", default="scaps_results")
    ap.add_argument("--out", default="dataset.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.samples)
    recs = []
    for sid in df.sample_id:
        t = parse_iv(os.path.join(a.results, f"top_{sid:05d}.iv"))
        b = parse_iv(os.path.join(a.results, f"bot_{sid:05d}.iv"))
        recs.append({
            "sample_id": sid,
            "Voc_top": t["Voc"], "Jsc_top": t["Jsc"], "FF_top": t["FF"],
            "PCE_top": t["Voc"] * t["Jsc"] * t["FF"] / 100.0,
            "Voc_bot": b["Voc"], "Jsc_bot": b["Jsc"], "FF_bot": b["FF"],
            "PCE_bot": b["Voc"] * b["Jsc"] * b["FF"] / 100.0,
        })
    res = pd.DataFrame(recs)
    res["PCE_4T"] = res.PCE_top + res.PCE_bot
    data = df.merge(res, on="sample_id").dropna()
    data.to_csv(a.out, index=False)
    print(f"{len(data)} complete samples written to {a.out}")
    print(data[["PCE_top", "PCE_bot", "PCE_4T"]].describe().round(2))


if __name__ == "__main__":
    main()
