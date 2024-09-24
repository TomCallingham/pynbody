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


def agama_pynbody_load(self, symm="axi") -> agama.Potential:
    f_sphere = self.analysis_folder + f"{symm}_sphere.coef_mul"
    f_disc = self.analysis_folder + f"{symm}_disc.coef_cylsp"

    if not os.path.isfile(f_sphere) or not os.path.isfile(f_disc):
        print("No potential found, creating")
        agama_pynbody_save(self, symm)

    pot_sphere = agama.Potential(f_sphere)  # type:ignore
    pot_disc = agama.Potential(f_disc)  # type:ignore
    Pot = agama.Potential(pot_sphere, pot_disc)  # type:ignore
    return Pot


def agama_pynbody_save(self, symm="axi") -> None:
    f_sphere = self.analysis_folder + f"{symm}_sphere.coef_mul"
    f_disc = self.analysis_folder + f"{symm}_disc.coef_cylsp"
    print("Agama pynbody calc!")
    pot_sphere, pot_disc = agama_pynbody_calc(self, symm="axi")
    pot_sphere.export(f_sphere)
    pot_disc.export(f_disc)
    print("Potentials Saved")


def agama_pynbody_calc(self, symm="axi", rcut=500) -> tuple:
    """
    Fits axisymmetric potential to main subhalo in sim
    constructs a hybrid two-component basis expansion model of the potential for Auriga.
    dark matter and hot gas are represented by an expansion in spherical harmonics.
    remaining baryons (stars and cold gas) are represented by an azimuthal harmonic expansion in
    phi and a quintic spline in (R,z). (see Agama docs, sections 2.2.2 and 2.2.3 for more details).
    Adapted from an example AGAMA script by Robyn Sanderson, with contributions from Andrew Wetzel, Eugene Vasiliev,
    by TomCallingham
    """
    print(f"reading {self}")

    h0 = self.halos()[0].sub[0]
    Gas = h0.gas
    Stars = h0.stars
    DM = h0.dm
    Wind = h0.winds
    # separate cold gas in disk (modeled with cylspline) from hot gas in halo
    # (modeled with multipole)
    cold_gas_filt = np.log10(Gas["temp"].v) < 4.5
    hot_gas_filt = np.invert(cold_gas_filt)

    # combine components that will be fed to the cylspline part
    disc_pos = np.vstack((Stars["pos"].v, Gas["pos"].v[cold_gas_filt]))
    disc_mass = np.hstack((Stars["mass"].v, Gas["mass"].v[cold_gas_filt]))
    # combine components that will be fed to the multipol part
    sphere_pos = np.vstack((DM["pos"].v, Gas["pos"].v[hot_gas_filt], Wind["pos"].v))
    sphere_mass = np.hstack((DM["mass"].v, Gas["mass"].v[hot_gas_filt], Wind["mass"].v))

    print("Computing multipole expansion coefficients for dark matter/hot gas component")
    pot_sphere = agama.Potential(
        type="multipole", particles=(sphere_pos, sphere_mass), lmax=8, symmetry=symm, rmin=0.1, rmax=rcut
    )

    print("Computing cylindrical spline coefficients for stellar/cold gas component")

    pot_disc = agama.Potential(
        type="cylspline",
        particles=(disc_pos, disc_mass),
        mmax=8,
        symmetry=symm,
        gridsizer=40,
        gridsizez=40,
        rmin=0.1,
        rmax=rcut,
    )
    print("Potential Created")

    return pot_sphere, pot_disc


def agama_vcirc(Rspace, pot) -> np.ndarray:
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    force = pot.force(cart_space)
    vcirc = np.sqrt(-cart_space[:, 0] * force[:, 0])
    return vcirc


def calc_action_angles(sim, angles=True) -> dict:
    pos, vel = sim["pos"].v, sim["vel"].v
    xyz = np.column_stack((pos, vel))

    base = sim.base if isinstance(sim, snapshot.FamilySubSnap) else sim
    pot = base.potential
    J_finder = agama.ActionFinder(pot, interp=True)
    dyn = {}
    if angles:
        print("Calculating Actions and Angles...")
        Js, As, Os = J_finder(xyz, angles=True)
        dyn["AR"], dyn["Az"], dyn["Aphi"] = As.T
        dyn["OR"], dyn["Oz"], dyn["Ophi"] = Os.T
    else:
        print("Calculating just Actions...")
        Js = J_finder(xyz, angles=False)
    dyn["JR"], dyn["Jz"], dyn["Jphi"] = Js.T
    # j_units = str(sim["pos"].units * sim["vel"].units)
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
