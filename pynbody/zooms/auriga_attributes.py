import numpy as np
from ..array import SimArray
from .. import units

from .auriga import AurigaLikeHDFSnap

HubbleTime = 13.815


@AurigaLikeHDFSnap.derived_quantity
def temp(sim) -> SimArray:
    """Xe Electron Abundance"""
    # Temperature estimate from:
    # https://www.tng-project.org/data/forum/topic/338/cold-and-hot-gas/#c3
    Xe = sim["ElectronAbundance"].v
    internalenergy = sim["u"].v

    XH = 0.76  # the hydrogen mass fraction
    gamma = 5.0 / 3.0  # the adiabatic index
    KB = 1.3807e-16  # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    mp = 1.6726e-24  # the proton mass  [g]
    mu = (4 * mp) / (1 + 3 * XH + 4 * XH * Xe)

    temp = (gamma - 1) * (internalenergy / KB) * mu * 1e10
    temp = SimArray(temp)
    temp.units = units.K
    temp.sim = sim

    return temp


@AurigaLikeHDFSnap.derived_quantity
def metallicity(sim) -> SimArray:
    met = np.log10(sim["metals"].v / 0.0127)
    met[np.isnan(met)] = -12
    met[np.isinf(met)] = -12
    met = SimArray(met)
    met.sim = sim
    return met


# Chemistry
met_ind = {"H": 0, "He": 1, "C": 2, "N": 3, "O": 4, "Ne": 5, "Mg": 6, "Si": 7, "Fe": 8}
metals = ["C", "N", "O", "Ne", "Mg", "Si", "Fe"]
alpha_metals = ["C", "N", "O", "Ne", "Mg", "Si"]
Solar_abun = {"H": 12, "He": 10.93, "C": 8.39, "N": 7.78, "O": 8.66, "Ne": 7.84, "Mg": 7.53, "Si": 7.51, "Fe": 7.45}
# http://articles.adsabs.harvard.edu/pdf/2005ASPC..336...25A

atomic_mass = {
    "H": 1.008,
    "He": 4.0026,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ne": 20.180,
    "Mg": 24.305,
    "Si": 28.085,
    "Fe": 55.845,
}


def Chem_X_H_Calc(x, mass_frac_X, mass_frac_H) -> np.ndarray:
    """element x"""

    n_X = mass_frac_X / atomic_mass[x]
    n_H = mass_frac_H / atomic_mass["H"]

    X_H = np.full_like(n_X, -12)
    valid_X = (n_X / n_H) > 0

    X_H[valid_X] = np.log10((n_X / n_H)[valid_X]) - (Solar_abun[x] - Solar_abun["H"])

    X_H[X_H < -12] = -12
    X_H[np.logical_not(np.isfinite(X_H))] = -12

    return X_H


def GFM_Chemistry(GFM_Metals) -> tuple:
    m_H = {m: Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]) for m in metals}

    m_Fe = {m: m_H[m] - m_H["Fe"] for m in alpha_metals}
    alpha_Fe = np.mean([m_Fe[m] for m in alpha_metals], axis=0)
    return m_H, m_Fe, alpha_Fe


@AurigaLikeHDFSnap.derived_quantity
def C_H(sim) -> SimArray:
    m = "C"
    GFM_Metals = sim["GFM_Metals"]
    C_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    C_H.sim = sim
    return C_H


@AurigaLikeHDFSnap.derived_quantity
def Fe_H(sim) -> SimArray:
    m = "Fe"
    GFM_Metals = sim["GFM_Metals"]
    Fe_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    Fe_H.sim = sim
    return Fe_H


@AurigaLikeHDFSnap.derived_quantity
def N_H(sim) -> SimArray:
    m = "N"
    GFM_Metals = sim["GFM_Metals"]
    N_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    N_H.sim = sim
    return N_H


@AurigaLikeHDFSnap.derived_quantity
def O_H(sim) -> SimArray:
    m = "O"
    GFM_Metals = sim["GFM_Metals"]
    O_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    O_H.sim = sim
    return O_H


@AurigaLikeHDFSnap.derived_quantity
def Ne_H(sim) -> SimArray:
    m = "Ne"
    GFM_Metals = sim["GFM_Metals"]
    Ne_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    Ne_H.sim = sim
    return Ne_H


@AurigaLikeHDFSnap.derived_quantity
def Mg_H(sim) -> SimArray:
    m = "Mg"
    GFM_Metals = sim["GFM_Metals"]
    Mg_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    Mg_H.sim = sim
    return Mg_H


@AurigaLikeHDFSnap.derived_quantity
def Si_H(sim) -> SimArray:
    m = "Si"
    GFM_Metals = sim["GFM_Metals"]
    Si_H = SimArray(Chem_X_H_Calc(m, GFM_Metals[:, met_ind[m]], GFM_Metals[:, met_ind["H"]]))
    Si_H.sim = sim
    return Si_H


@AurigaLikeHDFSnap.derived_quantity
def C_Fe(sim) -> SimArray:
    C_Fe = sim["C_H"] - sim["Fe_H"]
    return C_Fe


@AurigaLikeHDFSnap.derived_quantity
def N_Fe(sim) -> SimArray:
    N_Fe = sim["N_H"] - sim["Fe_H"]
    return N_Fe


@AurigaLikeHDFSnap.derived_quantity
def O_Fe(sim) -> SimArray:
    O_Fe = sim["O_H"] - sim["Fe_H"]
    return O_Fe


@AurigaLikeHDFSnap.derived_quantity
def Ne_Fe(sim) -> SimArray:
    Ne_Fe = sim["Ne_H"] - sim["Fe_H"]
    return Ne_Fe


@AurigaLikeHDFSnap.derived_quantity
def Mg_Fe(sim) -> SimArray:
    Mg_Fe = sim["Mg_H"] - sim["Fe_H"]
    return Mg_Fe


@AurigaLikeHDFSnap.derived_quantity
def Si_Fe(sim) -> SimArray:
    Si_Fe = sim["Si_H"] - sim["Fe_H"]
    return Si_Fe


@AurigaLikeHDFSnap.derived_quantity
def alpha_Fe(sim) -> SimArray:
    alpha_Fe = SimArray(np.mean([sim[f"{m}_Fe"].view(np.ndarray) for m in alpha_metals], axis=0))
    alpha_Fe.sim = sim
    return alpha_Fe


from pynbody import units


@AurigaLikeHDFSnap.derived_quantity
def n_HI(sim) -> SimArray:
    proton_mass = SimArray(1.673e-24, units.g)
    rho_gas = sim["rho"]  # Msol kpc^-3
    X_Hcell = sim["GFM_Metals"][:, 0]  # hydrogen-mass fraction
    f_HI = sim["NeutralHydrogenAbundance"]  # neutral fraction of H mass
    f_HI.units = units.Unit(1)
    X_Hcell.units = units.Unit(1)

    n_HI = (rho_gas * X_Hcell * f_HI) / proton_mass

    return n_HI


@AurigaLikeHDFSnap.derived_quantity
def n_H2(sim) -> SimArray:
    cold_dense = (sim["temp"] < 300) & (sim["n_HI"] > 10 * units.cm**-3)
    f_H2 = SimArray(np.where(cold_dense, 0.33, 0.0), units.Unit(1))
    n_H2 = 0.5 * sim["n_HI"] * f_H2  # 0.5 × because H2 has 2 H atoms
    return n_H2


@AurigaLikeHDFSnap.derived_quantity
def n_effH(sim) -> SimArray:
    n_effH = (sim["n_HI"] + 2 * sim["n_H2"]) * (sim["metals"] / 0.0127)
    print(n_effH.units)
    print(n_effH)
    return n_effH


@AurigaLikeHDFSnap.derived_quantity
def n_effH_DeVis(sim) -> SimArray:
    # https://arxiv.org/pdf/1901.09040 DeVis19
    DGR = (9.6e-3) * ((sim["metals"] / 0.0127) ** (2.45))
    n_effH = (sim["n_HI"] + 2 * sim["n_H2"]) * DGR / 0.01
    return n_effH


@AurigaLikeHDFSnap.derived_quantity
def n_effH_RemyRuyer(sim) -> SimArray:
    # RemyRuyer14
    """
    Effective neutral-H number density [cm^-3]:
      (n_HI + 2 n_H2) * (DGR / 0.01),
    with DGR(Z) from broken power‐law.
    """
    # total metal mass fraction
    met = sim["metallicity"]

    # dust-to-gas ratio (Mdust/Mgas)
    DGR = _dgr_broken(met)  # dim‐less

    # combine HI + H2
    nHI = sim["n_HI"]
    nH2mol = sim["n_H2"]
    # scale relative to 1% Milky-Way reference
    n_eff = (nHI + 2.0 * nH2mol) * (DGR / 0.01)

    return n_eff


def _dgr_broken(met):
    """
    Mdust/Mgas as a function of log10(Z/Z_sun),
    broken‐power‐law from Rémy‐Ruyer 14 / De Vis 19.
    """
    logDGR = np.where(
        met <= -0.59,
        0.96 + 3.10 * met,  # low‐Z branch
        2.21 + 1.00 * met,
    )  # high‐Z branch
    DGR = 10 ** (-logDGR)
    return np.clip(DGR, 0, 0.02)  # cap at 2 %
