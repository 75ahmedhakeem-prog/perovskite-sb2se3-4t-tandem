# Perovskite/Sb2Se3 four-terminal tandem: calibrated SCAPS-1D and machine-learning design

Code and data for the manuscript *"Bandgap is not enough: hole-transport alignment in perovskite/Sb2Se3 tandems"* (S. A. Ahmed, H. M. Altass, R. S. Jassas, S. S. A. Natto, A. Timoumi, A. M. Abdel Hakeem).

## Contents

| Folder | What it holds |
|---|---|
| `scaps_definitions/` | SCAPS-1D 3.3.12 definition files of the calibrated top and bottom cells |
| `data/spectra/` | AM1.5G file of SCAPS-1D and all filtered spectra (Eq. 4) used for the bottom cell |
| `data/scaps_runs/` | SCAPS-1D result files of the calibration, blind test and extended design |
| `data/ml/` | 9600- and 13 440-sample tandem datasets, model results, hold-out tests, TMM, Monte Carlo and detailed-balance results |
| `data/bandgap_scan/` | perovskite bandgap scan (42 top-cell and 28 bottom-cell SCAPS-1D runs): raw `.iv` files, filled run table, parsed results |
| `data/nk/` | optical constants used by the transfer-matrix model (refractiveindex.info database) |
| `code/` | Python scripts, numbered in the order of the workflow |
| `figures/` | scripts for selected figures |

## Scripts

| Script | Purpose |
|---|---|
| `01_generate_samples.py` | design grid of the dataset |
| `02_filtered_spectrum.py`, `06_filtered_spectrum_scaps.py` | filtered spectrum of Eq. (4) in SCAPS-1D format |
| `03_parse_results.py`, `07_parse_batch.py` | read SCAPS-1D `.iv` result files |
| `04_train_models.py`, `08_build_tandem_dataset.py`, `09_train_tandem.py` | dataset and XGBoost/ANN training |
| `10_level_holdout.py`, `18_holdout_baselines.py` | level hold-out test and baselines |
| `11_ann_design.py`, `12_extended_design.py` | ANN design and extended thickness range |
| `13_tmm_tpar.py` | transfer-matrix transmittance of the top-cell stack |
| `14_uncertainty.py` | Monte Carlo propagation of the calibration uncertainty |
| `15_detailed_balance.py` | detailed-balance limit of the 4T tandem |
| `16_bandgap_scan_inputs.py`, `17_bandgap_scan_results.py` | inputs and analysis of the perovskite bandgap scan |
| `05_validate_device.py` | comparison with measured devices |

Run the scripts from the `code/` folder; paths are relative to it. Python 3.10+ with numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, openpyxl and pyyaml.

## Notes

* SCAPS-1D reports the efficiency relative to the power of the spectrum file. Under filtered spectra, PCE is recomputed against 100 mW cm⁻² (Eq. 5).
* In the bandgap scan, the *pinned* case (P) widens the bandgap by lowering the valence band at constant electron affinity. In the *aligned* case (A), the electron affinity of Spiro-OMeTAD and the work function of the rear contact are raised by E_g − 1.60 eV.

## Licence

Code: MIT licence (see `LICENSE`). Data: CC BY 4.0. Please cite the article when using this material.
