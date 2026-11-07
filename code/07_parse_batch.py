"""
07_parse_batch.py
Convert a SCAPS-1D batch result file (.iv) into a CSV table.
Each simulation block gives the batch parameters (one or several) and the
J-V parameters. PCE is recomputed against 100 mW/cm2 (Eq. 2 of the protocol),
because SCAPS-1D normalises eta to the power of the spectrum file.

Usage
  python 07_parse_batch.py --iv top_ML.iv --out top_ML.csv
"""
import argparse
import re

import pandas as pd

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def short(name):
    """Readable column name from a SCAPS batch-parameter label."""
    n = re.sub(r"\[.*?\]", "", name)
    n = n.replace(">>", "|").strip()
    if "thickness" in n:
        return "thickness_um"
    if "/" in n.split("|")[0] or re.search(r"\(I\d+\)", n.split("|")[0]):
        return "Nt_interface"
    if "defect" in n.lower():
        return "Nt_bulk"
    if "acceptor" in n.lower():
        return "NA"
    if "donor" in n.lower():
        return "ND"
    return re.sub(r"\W+", "_", n).strip("_")


def parse(path):
    txt = open(path, errors="ignore").read()
    rows = []
    for blk in re.split(r"Simulation of I-V curves", txt)[1:]:
        rec = {}
        m = re.search(r"\*\*Batch parameters\*\*\s*\n(.*?)\n\s*\n", blk, re.S)
        if m:
            for line in m.group(1).splitlines():
                mm = re.match(r"(.*?):\s*" + NUM + r"\s*$", line.strip())
                if mm:
                    rec[short(mm.group(1))] = float(mm.group(2))
        for k in ("Voc", "Jsc", "FF", "eta"):
            mm = re.search(r"\b" + k + r"\s*=\s*" + NUM, blk)
            rec[k] = float(mm.group(1)) if mm else float("nan")
        sp = re.search(r"spectrum file :\s*\n\s*(\S.*?\.spe)", blk)
        rec["spectrum"] = sp.group(1).split("\\")[-1] if sp else ""
        rec["PCE"] = rec["Voc"] * rec["Jsc"] * rec["FF"] / 100.0
        if m:  # keep batch runs only; single-shot checks are skipped
            rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = parse(a.iv)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} simulations -> {a.out}")
    print(df.head().to_string())
