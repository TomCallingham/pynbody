import numpy as np
from .zoom import ZoomSnap
from .property_cache import cache_prop, multiple_read
from .agama_potential import agama_vcirc, calc_action_angles
from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot import FamilySubSnap


kms = units.km / units.s
kms2 = kms * kms

# TODO: Have units check? Or assume always physical units input


@ZoomSnap.derived_quantity
@cache_prop
def U(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    U_pot = SimArray(pot.potential(sim["pos"].view(np.ndarray)))
    U_pot.sim, U_pot.units = sim, kms2
    return U_pot


@ZoomSnap.derived_quantity
@cache_prop
def E(sim) -> SimArray:
    return sim["U"] + sim["ke"]


@ZoomSnap.derived_quantity
def L(sim) -> SimArray:
    '''Magnitude of  Angular Momentum'''
    L = SimArray(np.linalg.norm(sim["j"], axis=1))
    L.sim, L.units = sim, sim["j"].units
    return L


@ZoomSnap.derived_quantity
def Lperp(sim) -> SimArray:
    '''Perpendicular component of Angular Momentum'''
    return np.sqrt((sim['L']**2) - (sim['jz']**2))


def calc_peri_apo(sim) -> dict[str, SimArray]:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E, L = sim['E'].view(np.ndarray), sim['L'].view(np.ndarray)
    peri = SimArray(pot.Rperiapo(np.column_stack((E, L)))[:, 0])
    peri.sim, peri.units = sim, units.kpc
    apo = SimArray(pot.Rapoapo(np.column_stack((E, L)))[:, 1])
    apo.sim, apo.units = sim, units.kpc
    extreme_dict = {"peri": peri, "apo": apo}
    return extreme_dict


@ZoomSnap.derived_quantity
@cache_prop
def peri(sim) -> SimArray:
    extreme_dict = calc_peri_apo(sim)
    return multiple_read(sim, extreme_dict, read_key="peri")


@ZoomSnap.derived_quantity
@cache_prop
def apo(sim) -> SimArray:
    extreme_dict = calc_peri_apo(sim)
    return multiple_read(sim, extreme_dict, read_key="apo")


@ZoomSnap.derived_quantity
def ecc(sim) -> SimArray:
    ecc = (sim['apo'] - sim['peri']) / (sim['apo'] + sim['peri'])
    return ecc


@ZoomSnap.derived_quantity
def RcircE(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    RcircE = SimArray(pot.Rcirc(E=E))
    return RcircE


@ZoomSnap.derived_quantity
def RcircL(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    L = sim["L"].view(np.ndarray)
    RcircL = SimArray(pot.Rcirc(L=L))
    return RcircL


@ZoomSnap.derived_quantity
@cache_prop
def circ(sim) -> SimArray:
    circ = sim['Lz'] / sim['Lcirc_E']
    return circ


@ZoomSnap.derived_quantity
@cache_prop
def Tcirc(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    Tcirc = SimArray(pot.Tcirc(E))
    Tcirc.sim, Tcirc.units = sim, units.Gyr
    return Tcirc


@ZoomSnap.derived_quantity
def Lcirc_E(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    NR = 1000
    Rspace = np.linspace(0, 500, NR)
    vc_space = agama_vcirc(Rspace, pot)
    Lspace = Rspace * vc_space
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    pot_space = pot.potential(cart_space)
    Ecspace = pot_space + ((vc_space**2) / 2)
    Lcirc_E = SimArray(np.interp(E, Ecspace, Lspace))
    Lcirc_E = SimArray(pot.Tcirc(E))
    Lcirc_E.sim, Lcirc_E.units = sim, sim["Lz"].units
    return Lcirc_E

# ActionAngles
# Calculate seperately in agama_potential.action_angle_calc


@ZoomSnap.derived_quantity
@cache_prop
def JR(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "JR", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Jz(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "Jz", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Jphi(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "Jphi", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def AR(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "AR", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Az(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Az", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Aphi(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Aphi", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def OR(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "OR", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Oz(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Oz", save=True)


@ZoomSnap.derived_quantity
@cache_prop
def Ophi(sim) -> SimArray:
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Ophi", save=True)


# Derived from Action Angles


@ZoomSnap.derived_quantity
def Jtot(sim) -> SimArray:
    Jtot = sim["JR"] + sim["Jz"] + sim["Jphi"].abs()
    return Jtot


@ZoomSnap.derived_quantity
def J_diamond(sim) -> SimArray:
    Jd = (sim["Jz"] - sim["JR"]) / sim["Jtot"]
    return Jd


@ZoomSnap.derived_quantity
def Jphi_Jtot(sim) -> SimArray:
    '''Used in J_diamond'''
    return sim["Jphi"] / sim["Jtot"]


@ZoomSnap.derived_quantity
def JR_Jtot(sim) -> SimArray:
    '''Used in J_diamond'''
    return sim["JR"] / sim["Jtot"]


@ZoomSnap.derived_quantity
def Jz_Jtot(sim) -> SimArray:
    '''Used in J_diamond'''
    return sim["Jz"] / sim["Jtot"]
