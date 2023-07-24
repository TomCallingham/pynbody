import numpy as np
from .zoom import ZoomSnap
from .property_cache import cache_prop
from .agama_potential import agama_vcirc
from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot import FamilySubSnap


kms = units.km / units.s
kms2 = kms * kms

# Zoom


@ZoomSnap.derived_quantity
@cache_prop
def U(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    U_pot = SimArray(pot.potential(sim["pos"].view(np.ndarray)))  # units kpc check?
    U_pot.sim, U_pot.units = sim, kms2
    return U_pot


@ZoomSnap.derived_quantity
@cache_prop
def E(sim) -> SimArray:
    return sim["U"] + sim["ke"]


@ZoomSnap.derived_quantity
def Lp(sim) -> SimArray:
    return np.sqrt((sim['L']**2) - (sim['Lz']**2))


@cache_prop
@ZoomSnap.derived_quantity
def peri(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E, L = sim['E'].view(np.ndarray), sim['L'].view(np.ndarray)
    peri = SimArray(pot.Rperiapo(np.column_stack((E, L)))[:, 0])
    peri.sim, peri.units = sim, units.kpc
    return peri


@cache_prop
@ZoomSnap.derived_quantity
def apo(sim) -> SimArray:
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    pot = base.potential
    E, L = sim['E'].view(np.ndarray), sim['L'].view(np.ndarray)
    apo = SimArray(pot.Rperiapo(np.column_stack((E, L)))[:, 1])
    apo.sim, apo.units = sim, units.kpc
    return apo


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


@cache_prop
@ZoomSnap.derived_quantity
def circ(sim) -> SimArray:
    circ = sim['Lz'] / sim['Lcirc_E']
    return circ


@cache_prop
@ZoomSnap.derived_quantity
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
def JR(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Jz(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Jphi(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def AR(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Az(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Aphi(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def OR(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Oz(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
@cache_prop
def Ophi(_) -> SimArray:
    print("Never Used")


@ZoomSnap.derived_quantity
def Jtot(sim) -> SimArray:
    Jtot = sim["JR"] + sim["Jz"] + sim["Jphi"].abs()
    return Jtot


@ZoomSnap.derived_quantity
def Jd(sim) -> SimArray:
    Jd = (sim["Jz"] - sim["JR"]) / sim["Jtot"]
    return Jd


@ZoomSnap.derived_quantity
def Jphi_Jtot(sim) -> SimArray:
    return sim["Jphi"] / sim["Jtot"]

# Auriga
