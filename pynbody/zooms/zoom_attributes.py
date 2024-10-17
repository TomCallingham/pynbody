import numpy as np
from .. import units
from ..array import SimArray

from .orientation import single_rotation
from .zoom import ZoomSnap
from .property_cache import cache_prop
from .zoom_utils import top_hierarchy_family,multiple_read

kms = units.km / units.s
kms2 = kms * kms


@ZoomSnap.derived_array
def pos(sim) -> SimArray:
    '''lazy translation!'''
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    if not base.orientate:
        return sim["raw_pos"]
    orientation_dic = base.orientation
    x_cen, z_Rot = orientation_dic.get("x_cen"), orientation_dic.get("z_Rot")

    o_pos = sim["raw_pos"].v.copy() #If not copy, can try to orientate twice and change data.
    u = sim["raw_pos"].units
    if x_cen is not None:
        o_pos-=x_cen
    if z_Rot is not None:
        o_pos = single_rotation(z_Rot,o_pos)
    o_pos = SimArray(o_pos)
    o_pos.sim, o_pos.units = sim,u
    try:
        del sim["raw_pos"]#, sim["raw_x"], sim["raw_y"], sim["raw_z"],
    except Exception: #as e:
        pass
        # # print("Could not delete raw_pos")
        # print(e)
    return o_pos

@ZoomSnap.derived_array
def vel(sim) -> SimArray:
    '''lazy translation!'''
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    if not base.orientate:
        return sim["raw_vel"]
    orientation_dic = base.orientation
    v_cen, z_Rot = orientation_dic.get("v_cen"), orientation_dic.get("z_Rot")
    o_vel = sim["raw_vel"].v
    u = sim["raw_vel"].units
    if v_cen is not None:
        o_vel-=v_cen
    if z_Rot is not None:
        o_vel = single_rotation(z_Rot,o_vel)
    o_vel = SimArray(o_vel)
    try:
        del sim["raw_o_vel"]
    except Exception:
        pass
    o_vel.sim, o_vel.units = sim,u
    return o_vel


## convenience
@ZoomSnap.derived_array
@top_hierarchy_family
def sub_id(sim) -> SimArray:
    """Convenient function"""
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    fam = sim._unifamily if hasattr(sim, "_unifamily") else None
    sub_id = SimArray(base.halos(subhalos=True).get_group_array(family=fam))
    sub_id.sim = sim
    return sub_id

@ZoomSnap.derived_array
@top_hierarchy_family
def group_id(sim) -> SimArray:
    """Convenient function"""
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    fam = sim._unifamily if hasattr(sim, "_unifamily") else None
    group_id = SimArray(base.halos(subhalos=False).get_group_array(family=fam))
    group_id.sim = sim
    return group_id



@ZoomSnap.derived_array
@cache_prop
def U(sim) -> SimArray:
    # print("U!")
    # base = sim.base if isinstance(sim, FamilySubSnap) else sim
    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    pos = sim["pos"].view(np.ndarray)
    U_pot = SimArray(pot.potential(pos))
    U_pot.sim, U_pot.units = sim, kms2
    return U_pot


@ZoomSnap.derived_array
@cache_prop
def E(sim) -> SimArray:
    return sim["U"] + sim["ke"]


@ZoomSnap.derived_array
def L(sim) -> SimArray:
    """Magnitude of  Angular Momentum"""
    L = SimArray(np.linalg.norm(sim["j"], axis=1))
    L.sim, L.units = sim, sim["j"].units
    return L


@ZoomSnap.derived_array
def Lperp(sim) -> SimArray:
    """Perpendicular component of Angular Momentum"""
    return np.sqrt((sim["L"] ** 2) - (sim["jz"] ** 2))


def calc_peri_apo(sim) -> dict[str, SimArray]:
    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    E, L = sim["E"].view(np.ndarray), sim["L"].view(np.ndarray)
    peri_apo = pot.Rperiapo(np.column_stack((E, L)))
    peri = SimArray(peri_apo[:, 0])
    peri.sim, peri.units = sim, units.kpc
    apo = SimArray(peri_apo[:, 1])
    apo.sim, apo.units = sim, units.kpc
    extreme_dict = {"peri": peri, "apo": apo}
    return extreme_dict


@ZoomSnap.derived_array
@cache_prop
def peri(sim) -> SimArray:
    extreme_dict = calc_peri_apo(sim)
    return multiple_read(sim, extreme_dict, read_key="peri")


@ZoomSnap.derived_array
@cache_prop
def apo(sim) -> SimArray:
    extreme_dict = calc_peri_apo(sim)
    return multiple_read(sim, extreme_dict, read_key="apo")


@ZoomSnap.derived_array
def ecc(sim) -> SimArray:
    ecc = (sim["apo"] - sim["peri"]) / (sim["apo"] + sim["peri"])
    return ecc


@ZoomSnap.derived_array
def RcircE(sim) -> SimArray:
    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    RcircE = SimArray(pot.Rcirc(E=E))
    return RcircE


@ZoomSnap.derived_array
def RcircL(sim) -> SimArray:
    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    L = sim["L"].view(np.ndarray)
    RcircL = SimArray(pot.Rcirc(L=L))
    return RcircL


@ZoomSnap.derived_array
@cache_prop
def circ(sim) -> SimArray:
    circ = sim["jz"] / sim["Lcirc_E"]
    return circ


@ZoomSnap.derived_array
@cache_prop
def Tcirc(sim) -> SimArray:
    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    Tcirc = SimArray(pot.Tcirc(E))
    Tcirc.sim, Tcirc.units = sim, units.Gyr
    return Tcirc


@ZoomSnap.derived_array
def Lcirc_E(sim) -> SimArray:

    base = sim.ancestor if  hasattr(sim, "ancestor") else sim
    pot = base.potential
    E = sim["E"].view(np.ndarray)
    NR = 1000
    Rspace = np.linspace(0, 500, NR)

    from .agama_potential import agama_vcirc
    vc_space = agama_vcirc(Rspace, pot)
    Lspace = Rspace * vc_space
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    pot_space = pot.potential(cart_space)
    Ecspace = pot_space + ((vc_space**2) / 2)
    Lcirc_E = SimArray(np.interp(E, Ecspace, Lspace))
    Lcirc_E.sim, Lcirc_E.units = sim, units.kpc*kms
    return Lcirc_E


# ActionAngles
# Calculate seperately in agama_potential.action_angle_calc


@ZoomSnap.derived_array
@cache_prop
def JR(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "JR")


@ZoomSnap.derived_array
@cache_prop
def Jz(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "Jz")


@ZoomSnap.derived_array
@cache_prop
def Jphi(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=False)
    return multiple_read(sim, aa_dict, "Jphi")


@ZoomSnap.derived_array
@cache_prop
def AR(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "AR")


@ZoomSnap.derived_array
@cache_prop
def Az(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Az")


@ZoomSnap.derived_array
@cache_prop
def Aphi(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Aphi")


@ZoomSnap.derived_array
@cache_prop
def OR(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "OR")


@ZoomSnap.derived_array
@cache_prop
def Oz(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Oz")


@ZoomSnap.derived_array
@cache_prop
def Ophi(sim) -> SimArray:
    from .agama_potential import calc_action_angles
    aa_dict = calc_action_angles(sim, angles=True)
    return multiple_read(sim, aa_dict, "Ophi")


# Derived from Action Angles


@ZoomSnap.derived_array
def Jtot(sim) -> SimArray:
    Jtot = sim["JR"] + sim["Jz"] + sim["Jphi"].abs()
    return Jtot


@ZoomSnap.derived_array
def J_diamond(sim) -> SimArray:
    Jd = (sim["Jz"] - sim["JR"]) / sim["Jtot"]
    return Jd


@ZoomSnap.derived_array
def Jphi_Jtot(sim) -> SimArray:
    """Used in J_diamond"""
    return sim["Jphi"] / sim["Jtot"]


@ZoomSnap.derived_array
def JR_Jtot(sim) -> SimArray:
    """Used in J_diamond"""
    return sim["JR"] / sim["Jtot"]


@ZoomSnap.derived_array
def Jz_Jtot(sim) -> SimArray:
    """Used in J_diamond"""
    return sim["Jz"] / sim["Jtot"]

