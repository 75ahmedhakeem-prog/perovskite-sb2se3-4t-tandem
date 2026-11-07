"""
16_bandgap_scan_inputs.py
Inputs for the SCAPS-1D scan of the perovskite bandgap (1.55-1.85 eV).
1) Filtered spectra (.spe) for the bottom cell: Eq. (4) with the same alpha law as the top cell,
   for two top-cell thicknesses (0.5 and 1.0 um) and two T_par values (0.67 and 0.90).
2) Run table (xlsx) with the SCAPS-1D settings of every run and empty result columns.
Top-cell cases
  P (pinned HTL, as in the manuscript): Eg widened by lowering the valence band, chi_PVK = 3.90 eV,
    Spiro-OMeTAD unchanged -> the interface gap PVK(CB)/Spiro(VB) stays 1.36 eV.
  A (aligned HTL): chi_Spiro = 2.20 + (Eg - 1.60) eV, so the Spiro valence band follows the perovskite
    valence band with the 0.24 eV offset of the calibrated 1.60 eV device (identical to P at 1.60 eV).
"""
import importlib.util, os, sys
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("f06", os.path.join(HERE, "06_filtered_spectrum_scaps.py"))
f06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f06)

SPE = sys.argv[1]; OUT = sys.argv[2]
EGS = [1.55, 1.60, 1.65, 1.68, 1.72, 1.77, 1.85]
STACKS = [("S1", 0.5, 0.330), ("S2", 1.0, 0.875)]   # (stack, d_PVK um, d_Sb2Se3 um)
TPARS = [0.67, 0.90]
os.makedirs(os.path.join(OUT, "spectra"), exist_ok=True)
lam, p, _ = f06.read_scaps_spe(SPE)

def tag(x, n=2): return f"{x:.{n}f}".replace(".", "p")
summary = []
for eg in EGS:
    for st, d, _ in STACKS:
        for tp in TPARS:
            name = f"F_Eg{tag(eg)}_d{int(d*1000)}_T{int(round(tp*100)):03d}.spe"
            T = tp * np.exp(-f06.alpha_sqrt(lam, eg) * d * 1e3 * 1e-7)
            pf = p * T
            with open(os.path.join(OUT, "spectra", name), "w") as f:
                f.write("> Filtered AM1.5G spectrum for the Sb2Se3 bottom cell (4T tandem)\n")
                f.write(f"> Top-cell perovskite: Eg = {eg:.3f} eV, d = {d*1000:.0f} nm, alpha0 = 1e+05, beta0 = 0\n")
                f.write(f"> Sub-bandgap transmission of the top-cell stack T_par = {tp:.4f}\n")
                f.write("> second column: incident power per wavelength interval (W/m2), as in the SCAPS source file\n")
                f.write(f"> total filtered power = {pf.sum():.3f} W/m2 (unfiltered {p.sum():.3f} W/m2)\n>\n>lambda (nm)\tP(W/m2)\n")
                for l, v in zip(lam, pf): f.write(f"{l:.1f}\t{v:.5f}\n")
            summary.append((name, eg, d, tp, pf.sum(), f06.jph(lam, pf, 1239.84193 / 1.20)))

wb = Workbook()
def sheet(ws, head, rows, widths):
    ws.append(head)
    for r in rows: ws.append(r)
    for c in ws[1]: c.font = Font(bold=True); c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment = Alignment(horizontal="center")
    for i, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
RES = ["Voc (V)", "Jsc (mA/cm2)", "FF (%)"]
top = []
k = 0
for case in ("P", "A"):
    for eg in EGS:
        for st, d, _ in STACKS:
            if case == "A" and st == "S1": continue
            for nt in (1e15, 1e14):
                k += 1
                chi = 2.20 if case == "P" else round(2.20 + (eg - 1.60), 3)
                top.append([f"T{k:02d}", case, eg, 3.90, round(3.90 + eg, 3), d, f"{nt:.0e}", chi, 3.06,
                            round(2.20 + 3.06 if case == "P" else chi + 3.06, 3), None, None, None])
ws = wb.active; ws.title = "Top cell"
sheet(ws, ["Run", "HTL case", "Eg PVK (eV)", "chi PVK (eV)", "VB PVK (eV)", "d PVK (um)", "Nt PVK (cm-3)",
           "chi Spiro (eV)", "Eg Spiro (eV)", "VB Spiro (eV)"] + RES, top, [7, 9, 11, 11, 11, 10, 13, 13, 12, 12, 10, 13, 9])
bot = []
k = 0
for eg in EGS:
    for st, d, dsb in STACKS:
        for tp in TPARS:
            k += 1
            bot.append([f"B{k:02d}", eg, d, tp, f"F_Eg{tag(eg)}_d{int(d*1000)}_T{int(round(tp*100)):03d}.spe", dsb, None, None, None])
ws2 = wb.create_sheet("Bottom cell")
sheet(ws2, ["Run", "Eg PVK (eV)", "d PVK (um)", "T_par", "Spectrum file", "d Sb2Se3 (um)"] + RES, bot, [7, 11, 10, 8, 30, 13, 10, 13, 9])
ws3 = wb.create_sheet("Spectra")
sheet(ws3, ["Spectrum file", "Eg PVK (eV)", "d PVK (um)", "T_par", "Filtered power (W/m2)", "Max Jph below 1033 nm (mA/cm2)"],
      [[n, e, d, t, round(P, 2), round(J, 2)] for n, e, d, t, P, J in summary], [30, 11, 10, 8, 18, 22])
wb.save(os.path.join(OUT, "bandgap_scan_runs.xlsx"))
print(len(top), "top runs,", len(bot), "bottom runs,", len(summary), "spectra")
for s in summary: print(s[0], round(s[4], 1), round(s[5], 2))
