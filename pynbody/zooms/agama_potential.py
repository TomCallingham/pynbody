import agama
import os
import numpy as np
import h5py
from pynbody import units
import pynbody

# define the physical units used in the code: the choice below corresponds to
# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1., length=1., velocity=1)

symmlabel = {'a': 'axi', 's': 'sph', 't': 'triax', 'n': 'none'}


action_angles_props = ["JR", "Jz", "Jphi", "AR", "Az", "Aphi", "OR", "Oz", "Ophi"]


def agama_pynbody_load(self, symm="axi") -> agama.Potential:
    f_sphere = self.analysis_folder + f'{symm}_sphere.coef_mul'
    f_disc = self.analysis_folder + f'{symm}_disc.coef_cylsp'

    if not os.path.isfile(f_sphere):
        print('No potential found, creating')
        agama_pynbody_save(self, symm)

    pot_sphere = agama.Potential(f_sphere)  # type:ignore
    pot_disc = agama.Potential(f_disc)  # type:ignore
    Pot = agama.Potential(pot_sphere, pot_disc)  # type:ignore
    return Pot


def agama_pynbody_save(self, symm="axi") -> None:
    f_sphere = self.analysis_folder + f'{symm}_sphere.coef_mul'
    f_disc = self.analysis_folder + f'{symm}_disc.coef_cylsp'
    pot_sphere, pot_disc = agama_pynbody_calc(self, symm="axi")
    pot_sphere.export(f_sphere)
    pot_disc.export(f_disc)
    print('Potentials Saved')


def agama_pynbody_calc(self, symm="axi", rcut=500) -> tuple:
    '''
    constructs a hybrid two-component basis expansion model of the potential for Auriga.
    dark matter and hot gas are represented by an expansion in spherical harmonics.
    remaining baryons (stars and cold gas) are represented by an azimuthal harmonic expansion in
    phi and a quintic spline in (R,z). (see Agama docs, sections 2.2.2 and 2.2.3 for more details).
    Arguments:
    Adapted from an example AGAMA script by Robyn Sanderson, with contributions from Andrew Wetzel, Eugene Vasiliev.
    '''
    print(f'reading {self}')

    # h0_data = sim_snap.halos()[0].properties
    # if rcut_R200:
    #     rcut = rcut * Host['R200']
    h0 = self.halos()[0].sub[0]
    Gas = h0.gas
    Stars = h0.stars
    DM = h0.dm
    Wind = h0.winds
    # separate cold gas in disk (modeled with cylspline) from hot gas in halo
    # (modeled with multipole)
    cold_gas_filt = (np.log10(Gas['temp'].v) < 4.5)
    hot_gas_filt = np.invert(cold_gas_filt)

    # combine components that will be fed to the cylspline part
    disc_pos = np.vstack((Stars['pos'], Gas['pos'][cold_gas_filt]))
    disc_mass = np.hstack((Stars['mass'], Gas['mass'][cold_gas_filt]))
    # combine components that will be fed to the multipol part
    sphere_pos = np.vstack((DM['pos'], Gas['pos'][hot_gas_filt], Wind['pos']))
    sphere_mass = np.hstack((DM['mass'], Gas['mass'][hot_gas_filt], Wind['mass']))

    print('Computing multipole expansion coefficients for dark matter/hot gas component')
    pot_sphere = agama.Potential(type='multipole',
                                 particles=(sphere_pos, sphere_mass),
                                 lmax=8, symmetry=symm,
                                 rmin=0.1, rmax=rcut)

    print('Computing cylindrical spline coefficients for stellar/cold gas component')

    pot_disc = agama.Potential(type='cylspline',
                               particles=(disc_pos, disc_mass),
                               mmax=8, symmetry=symm,
                               gridsizer=40, gridsizez=40,
                               rmin=0.1, rmax=rcut)
    print('Potential Created')

    return pot_sphere, pot_disc


def agama_vcirc(Rspace, pot) -> np.ndarray:
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    force = pot.force(cart_space)
    vcirc = np.sqrt(-cart_space[:, 0] * force[:, 0])
    return vcirc


def calc_action_angles(sim, fam, save=True) -> dict:
    print("Calculating Action Angles")

    pos, vel = sim["pos"].view(np.ndarray), sim["vel"].view(np.ndarray)
    xyz = np.column_stack((pos, vel))

    base = sim.base if isinstance(sim, pynbody.snapshot.FamilySubSnap) else sim
    pot = base.potential
    J_finder = agama.ActionFinder(pot, interp=True)
    dyn = {}
    Js, As, Os = J_finder(xyz, angles=True)
    dyn['JR'], dyn["Jz"], dyn["Jphi"] = Js.T
    dyn['AR'], dyn["Az"], dyn["Aphi"] = As.T
    dyn['OR'], dyn["Oz"], dyn["Ophi"] = Os.T
    aa_units = {}
    j_units = str(sim["pos"].units * sim["vel"].units)
    o_units = str(1 / units.Gyr)
    for p in ["JR", "Jz", "Jphi"]:
        aa_units[p] = j_units
    for p in ["AR", "Az", "Aphi"]:
        aa_units[p] = "NoUnit()"
    for p in ["OR", "Oz", "Ophi"]:
        aa_units[p] = o_units
    if save:
        save_action_angles(sim, fam, dyn, aa_units)
    return dyn


def save_action_angles(sim, fam, dyn, aa_units) -> None:
    print("Saving Action Angles")
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    file = f"{base.analysis_folder}{fam}_cached_properties.hdf5"
    with h5py.File(file, "a") as hf:
        for xkey in action_angles_props:
            dset = hf.create_dataset(f"{xkey}", data=dyn[xkey])
            dset.attrs["units"] = aa_units[xkey]
    base.cached_props[fam] += action_angles_props
