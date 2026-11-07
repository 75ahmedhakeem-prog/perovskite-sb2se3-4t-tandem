# SCAPS-1D 3.3.12 definition files

| File | Device |
|---|---|
| `top_cell_1p60eV_calibrated.scaps` | top cell calibrated against the measured 1.60 eV device (Table 4) |
| `top_cell_1p68eV_design_Eg_scan.scaps` | 1.68 eV top-cell design (0.5 µm perovskite); starting point of the bandgap scan |
| `bottom_cell_Sb2Se3_calibrated.scaps` | final calibrated Sb2Se3 bottom cell (0.33 µm, Rs = 2.36 Ω cm², Rsh = 215 Ω cm²) |

For the bandgap scan, the perovskite bandgap was varied in batch mode (1.55–1.85 eV).
In the aligned case, the electron affinity of Spiro-OMeTAD and the work function of the
right (rear) contact were varied simultaneously by E_g − 1.60 eV.
The bottom cell is run under the filtered spectra in `data/spectra/`.
