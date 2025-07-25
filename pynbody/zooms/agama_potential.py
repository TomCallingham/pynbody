import agama
import os
import numpy as np
from .. import units, snapshot
from ..array import SimArray

# define the physical units used in the code: the choice below corresponds to
# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1.0, length=1.0, velocity=1)

symmlabel = {"a": "axi", "s": "sph", "t": "triax", "n": "none"}


action_angles_props = ["JR", "Jz", "Jphi", "AR", "Az", "Aphi", "OR", "Oz", "Ophi"]


def agama_pynbody_load(Sim) -> agama.Potential:
    symmetry = Sim.pot_symm
    if symmetry == "axi":
        return agama_pynbody_load_axi(Sim)
    elif symmetry == "spherical":
        return agama_pynbody_load_sph(Sim)
    else:
        raise KeyError(f"Potential symmetry not recognised: {symmetry}")


def agama_pynbody_load_axi(Sim, sub_id: None | int = None) -> agama.Potential:
    # print("Loading Axi!")
    f_sphere = f"{Sim.analysis_folder}axi_sphere_{Sim.orientation_name}.coef_mul"
    f_disc = f"{Sim.analysis_folder}axi_disc_{Sim.orientation_name}.coef_cylsp"

    if os.path.isfile(f_sphere) or os.path.isfile(f_disc):
        pot_sphere = agama.Potential(f_sphere)  # type:ignore
        pot_disc = agama.Potential(f_disc)  # type:ignore
        return agama.Potential(pot_sphere, pot_disc)  # type:ignore

    print("No potential found, creating")
    if sub_id is None:
        try:
            sub_id = int(Sim.orientation_name.split("_")[-1])
        except Exception:
            print("No sub_id found, using 0")
            sub_id = 0
    print(f"Using sub_id:{sub_id}")
    pot_sphere, pot_disc = agama_pynbody_save_axi(Sim, sub_id=sub_id)
    return agama.Potential(pot_sphere, pot_disc)  # type:ignore


def agama_pynbody_save_axi(Sim, sub_id=0, n_max=16) -> tuple:
    f_sphere = f"{Sim.analysis_folder}axi_sphere_{Sim.orientation_name}.coef_mul"
    f_disc = f"{Sim.analysis_folder}axi_disc_{Sim.orientation_name}.coef_cylsp"

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=n_max, user_api="openmp"):
        print(f"Using threadpool_limits: {n_max}")
        pot_sphere, pot_disc = agama_pynbody_calc_axi(Sim, sub_id=sub_id)
    pot_sphere.export(f_sphere)
    pot_disc.export(f_disc)
    print("Potentials Saved")
    return pot_sphere, pot_disc


def agama_pynbody_save_sph(Sim) -> None:
    fname = Sim.analysis_folder + "sph.coef_mul"
    print("Agama Spherical pynbody calc!")
    pot = agama_pynbody_calc_sph(Sim)
    pot.export(fname)
    print("Potential Saved")


def agama_pynbody_calc_axi(Sim, rcut: float = 500, sub_id: None | int = 0) -> tuple:
    """
    Fits axisymmetric potential to given Sim. Subhalo/Orientation should already be selected
    constructs a hybrid two-component basis expansion model of the potential for Auriga.
    dark matter and hot gas are represented by an expansion in spherical harmonics.
    remaining baryons (stars and cold gas) are represented by an azimuthal harmonic expansion in
    phi and a quintic spline in (R,z). (see Agama docs, sections 2.2.2 and 2.2.3 for more details).
    Adapted from an example AGAMA script by Robyn Sanderson, with contributions from Andrew Wetzel, Eugene Vasiliev,
    by TomCallingham
    """

    if sub_id is not None:
        print(f"Getting subhalo {sub_id}")
        Sim = Sim.halos(subhalos=True)[sub_id]
    Sim = Sim[Sim["r"] < rcut]
    print(f"reading {Sim}")
    Gas = Sim.gas
    Stars = Sim.stars
    DM = Sim.dm
    # TODO: Auriga Specific with wind!
    Wind = Sim.winds
    # separate cold gas in disk (modeled with cylspline) from hot gas in halo
    # (modeled with multipole)
    cold_gas_filt = np.log10(Gas["temp"].v) < 4.5
    hot_gas_filt = np.invert(cold_gas_filt)

    # combine components that will be fed to the cylspline part
    if len(Stars) > 0:
        disc_pos = np.vstack((Stars["pos"].v, Gas["pos"].v[cold_gas_filt]))
        disc_mass = np.hstack((Stars["mass"].v, Gas["mass"].v[cold_gas_filt]))
    else:
        disc_pos = Gas["pos"].v[cold_gas_filt]
        disc_mass = Gas["mass"].v[cold_gas_filt]

    # combine components that will be fed to the multipol part
    if len(Wind) > 0:
        sphere_pos = np.vstack((DM["pos"].v, Gas["pos"].v[hot_gas_filt], Wind["pos"].v))
        sphere_mass = np.hstack((DM["mass"].v, Gas["mass"].v[hot_gas_filt], Wind["mass"].v))
    else:
        sphere_pos = np.vstack((DM["pos"].v, Gas["pos"].v[hot_gas_filt]))
        sphere_mass = np.hstack((DM["mass"].v, Gas["mass"].v[hot_gas_filt]))

    print("Computing multipole expansion coefficients for dark matter/hot gas component")
    pot_sphere = agama.Potential(
        type="multipole", particles=(sphere_pos, sphere_mass), lmax=8, symmetry="axi", rmin=0.1, rmax=rcut
    )

    print("Computing cylindrical spline coefficients for stellar/cold gas component")

    pot_disc = agama.Potential(
        type="cylspline",
        particles=(disc_pos, disc_mass),
        mmax=8,
        symmetry="axi",
        gridsizer=40,
        gridsizez=40,
        rmin=0.1,
        rmax=rcut,
    )
    print("Potential Created")

    return pot_sphere, pot_disc


def agama_pynbody_calc_sph(Sim, rcut=500) -> agama.Potential:
    """ """
    print(f"reading {Sim}")

    pos = Sim["pos"].v
    mass = Sim["mass"].v

    print("Computing Spherical Pot")
    pot = agama.Potential(type="multipole", particles=(pos, mass), lmax=8, symmetry="sph", rmin=0.1, rmax=rcut)
    print("Potential Created")

    return pot


def agama_vcirc(Rspace, pot) -> np.ndarray:
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    force = pot.force(cart_space)
    vcirc = np.sqrt(-cart_space[:, 0] * force[:, 0])
    return vcirc


def calc_action_angles(sim, angles=True) -> dict:
    pos, vel = sim["pos"].v, sim["vel"].v
    xyz = np.column_stack((pos, vel))

    pot = sim.ancestor.potential if hasattr(sim, "ancestor") else sim.potential
    J_finder = agama.ActionFinder(pot, interp=True)
    dyn = {}
    if angles:
        print("Calculating Actions, Angles and Frequencies...")
        Js, As, Os = J_finder(xyz, angles=True)
        dyn["AR"], dyn["Az"], dyn["Aphi"] = As.T
        dyn["OR"], dyn["Oz"], dyn["Ophi"] = Os.T
    else:
        print("Calculating just Actions...")
        Js = J_finder(xyz, angles=False)
    dyn["JR"], dyn["Jz"], dyn["Jphi"] = Js.T
    j_units = units.kpc * units.km / units.s
    o_units = 1 / units.Gyr

    for p in list(dyn.keys()):
        x = SimArray(dyn[p])
        x.sim = sim
        if p in ["JR", "Jz", "Jphi"]:
            x.units = j_units
        elif p in ["OR", "Oz", "Ophi"]:
            x.units = o_units
        dyn[p] = x
    print("Calculated!")
    return dyn
