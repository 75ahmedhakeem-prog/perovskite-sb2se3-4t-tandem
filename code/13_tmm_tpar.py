"""
13_tmm_tpar.py
Transfer-matrix (TMM) optics of the perovskite top-cell stack.

Stack (light enters through the glass):
  air | glass (1.1 mm, incoherent) | ITO | SnO2 | perovskite | Spiro-OMeTAD | ITO | air
The thin films are treated coherently (characteristic-matrix method); the glass is
treated incoherently (intensity addition of multiple reflections).

Optical data
  ITO          : sputtered ITO on glass, Minenkov et al., ACS AMI 16, 9517 (2024)  (refractiveindex.info)
  perovskite n : MAPbI3, Phillips et al., Data in Brief 5, 926 (2015)            (refractiveindex.info)
  perovskite k : k = alpha*lambda/(4 pi), alpha = SCAPS sqrt law (A = 1e5 cm^-1 eV^-1/2) + Urbach tail,
                 i.e. the same absorption that the SCAPS model and Eq. (4) use
  SnO2, Spiro  : constant n = 1.95 and 1.75, k = 0 (assumption, varied in the sensitivity test)
  glass        : n = 1.52

Outputs
  - T(lambda) of the stack for Eg = 1.60, 1.68, 1.77 eV
  - bottom-cell Jsc = q * sum(Phi * T * EQE_bot) with the calibrated bottom-cell EQE
  - effective T_par = Jsc(TMM) / Jsc(Beer-Lambert, T_par = 1)
  - current ratio R_J = Jsc(1.77 eV filter) / Jsc(1.60 eV filter)
"""
import json
import numpy as np
import yaml

HC = 1239.84193  # eV nm
Q = 1.602176634e-19
DB = "../data/nk/"
SPE = "../data/spectra/AM1_5G 1 sun.spe"
EQE = "../data/scaps_runs/bottomB/B_final.qe"


def load_nk(path):
    d = yaml.safe_load(open(DB + path))
    a = np.array([[float(x) for x in l.split()] for l in d["DATA"][0]["data"].strip().splitlines()])
    return a[:, 0] * 1000, a[:, 1], a[:, 2]


def interp_nk(path, lam):
    l, n, k = load_nk(path)
    return np.interp(lam, l, n) + 1j * np.interp(lam, l, k)


def alpha_sqrt(lam, Eg, A=1e5, Eu=0.015):
    E = HC / lam
    al = np.where(E > Eg, A * np.sqrt(np.clip(E - Eg, 0, None)), 0.0)
    a0 = A * np.sqrt(Eu)
    tail = a0 * np.exp((E - Eg - Eu) / Eu)
    return np.where(E > Eg + Eu, al, np.minimum(tail, a0))  # cm^-1


def coherent_rt(nlist, dlist, lam):
    """Normal-incidence characteristic-matrix method. nlist includes the two
    semi-infinite media; dlist (nm) the thin films. Returns r, t, R, T."""
    nlist = [np.conj(n) for n in nlist]  # Macleod convention N = n - ik for the matrix below
    n0, ns = nlist[0], nlist[-1]
    M = np.array([[1, 0], [0, 1]], dtype=complex)
    for n, d in zip(nlist[1:-1], dlist):
        delta = 2 * np.pi * n * d / lam
        M = M @ np.array([[np.cos(delta), 1j * np.sin(delta) / n], [1j * n * np.sin(delta), np.cos(delta)]])
    B, C = M @ np.array([1, ns])
    r = (n0 * B - C) / (n0 * B + C)
    t = 2 * n0 / (n0 * B + C)
    return r, t, abs(r) ** 2, (ns.real / n0.real) * abs(t) ** 2


def stack_T(lam, Eg, d_pvk=500, d_ito_f=150, d_sno2=100, d_spiro=250, d_ito_r=150,
            n_sno2=1.95, n_spiro=1.75, dn_pvk=0.0, n_glass=1.52):
    lam = np.atleast_1d(lam).astype(float)
    ito = interp_nk("other/mixed crystals/In2O3-SnO2/nk/Minenkov-glass.yml", lam)
    npvk = interp_nk("other/perovskite/CH3NH3PbI3/nk/Phillips.yml", lam).real + dn_pvk
    kpvk = alpha_sqrt(lam, Eg) * lam * 1e-7 / (4 * np.pi)
    pvk = npvk + 1j * kpvk
    T = np.zeros_like(lam)
    for i, l in enumerate(lam):
        films = [ito[i], n_sno2 + 0j, pvk[i], n_spiro + 0j, ito[i]]
        dl = [d_ito_f, d_sno2, d_pvk, d_spiro, d_ito_r]
        # glass -> films -> air (forward) and air <- films <- glass (for the glass-side reflectance)
        _, _, Rgf, Tgf = coherent_rt([n_glass + 0j] + films + [1 + 0j], dl, l)
        # air/glass interface (incoherent)
        Rag = ((1 - n_glass) / (1 + n_glass)) ** 2
        Tag = 1 - Rag
        # incoherent glass slab (no glass absorption): multiple reflections between air/glass and glass/films
        T[i] = Tag * Tgf / (1 - Rag * Rgf)
    return T


def load_spectrum():
    lam, P = [], []
    for line in open(SPE, encoding="latin-1"):
        s = line.strip()
        if not s or s.startswith(">") or s.startswith("/"):
            continue
        p = s.split()
        try:
            lam.append(float(p[0])); P.append(float(p[1]))
        except ValueError:
            pass
    return np.array(lam), np.array(P)  # W m^-2 per interval


def load_eqe():
    lam, qe = [], []
    rd = False
    for line in open(EQE, encoding="latin-1"):
        p = line.split()
        if line.strip().startswith("lambda(nm)"):
            rd = True; continue
        if rd and len(p) >= 2:
            try:
                lam.append(float(p[0])); qe.append(float(p[1]) / 100)
            except ValueError:
                if lam: break
    return np.array(lam), np.array(qe)


def jsc_bottom(T_of_lam):
    lam, P = load_spectrum()
    le, qe = load_eqe()
    eqe = np.interp(lam, le, qe, left=0, right=0)
    flux = P / (HC / lam * Q)  # photons m^-2 s^-1 per interval
    return Q * np.sum(flux * T_of_lam(lam) * eqe) / 10  # mA cm^-2


def bl_T(lam, Eg, d_pvk=500, T_par=1.0):
    return T_par * np.exp(-alpha_sqrt(lam, Eg) * d_pvk * 1e-7)


if __name__ == "__main__":
    out = {}
    j0 = jsc_bottom(lambda l: np.ones_like(l))
    out["Jsc_bottom_unfiltered"] = j0
    lamg = np.arange(350, 1201, 2.0)
    spectra = {}
    for Eg in (1.60, 1.68, 1.77):
        jt = jsc_bottom(lambda l, Eg=Eg: stack_T(l, Eg))
        jb = jsc_bottom(lambda l, Eg=Eg: bl_T(l, Eg))
        out[f"Eg{Eg}"] = {"Jsc_TMM": jt, "Jsc_BL_T1": jb, "Tpar_eff": jt / jb}
        spectra[Eg] = (stack_T(lamg, Eg), bl_T(lamg, Eg))
    out["RJ_TMM"] = out["Eg1.77"]["Jsc_TMM"] / out["Eg1.6"]["Jsc_TMM"]
    out["RJ_BL"] = out["Eg1.77"]["Jsc_BL_T1"] / out["Eg1.6"]["Jsc_BL_T1"]
    # sensitivity of T_par (1.68 eV, 500 nm) and of R_J
    sens = {}
    base = dict()
    cases = {"ITO_rear_100": dict(d_ito_r=100), "ITO_rear_250": dict(d_ito_r=250), "ITO_front_100": dict(d_ito_f=100),
             "ITO_front_250": dict(d_ito_f=250), "nSnO2_1.80": dict(n_sno2=1.80), "nSnO2_2.10": dict(n_sno2=2.10),
             "nSpiro_1.60": dict(n_spiro=1.60), "nSpiro_1.90": dict(n_spiro=1.90), "nPVK_-0.15": dict(dn_pvk=-0.15),
             "Spiro_150": dict(d_spiro=150), "Spiro_350": dict(d_spiro=350)}
    jb68 = out["Eg1.68"]["Jsc_BL_T1"]
    for k, kw in cases.items():
        t68 = jsc_bottom(lambda l: stack_T(l, 1.68, **kw)) / jb68
        rj = jsc_bottom(lambda l: stack_T(l, 1.77, **kw)) / jsc_bottom(lambda l: stack_T(l, 1.60, **kw))
        sens[k] = {"Tpar_1.68": t68, "RJ": rj}
    out["sensitivity"] = sens
    # T_par versus perovskite thickness (1.68 eV)
    out["Tpar_vs_dpvk"] = {int(d): jsc_bottom(lambda l, d=d: stack_T(l, 1.68, d_pvk=d)) / jsc_bottom(lambda l, d=d: bl_T(l, 1.68, d))
                           for d in (300, 500, 700, 900, 1000)}
    # thicker 1.77 eV absorber (blind-test hypothesis): R_J if d_pvk(1.77) differs
    out["RJ_vs_d177"] = {int(d): jsc_bottom(lambda l, d=d: stack_T(l, 1.77, d_pvk=d)) / out["Eg1.6"]["Jsc_TMM"] for d in (300, 400, 500, 600)}
    json.dump(out, open("../data/ml/tmm_results.json", "w"), indent=1)
    np.savez("../data/ml/tmm_spectra.npz", lam=lamg, **{f"TMM_{k}": v[0] for k, v in spectra.items()},
             **{f"BL_{k}": v[1] for k, v in spectra.items()})
    print(json.dumps(out, indent=1))
