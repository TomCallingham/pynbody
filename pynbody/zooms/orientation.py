import os
from .. import analysis, filt, transformation
import numpy as np
import h5py


def load_orientation(Sim, sub_id: int = 0) -> dict[str, np.ndarray]:
    fname = f"{Sim.analysis_folder}pynbody_orientation_{Sim.orientation_name}.hdf5"
    if os.path.isfile(fname):
        with h5py.File(fname) as hf:
            orientation_dic = {p: np.asarray(hf[p][:]) for p in hf.keys()}
    else:
        orientation_dic = calc_apply_pynbody_orientation(Sim, sub_id)
        save_pynbody_orientation(fname, orientation_dic)

    return orientation_dic


def calc_apply_pynbody_orientation(Sim, sub_id: int = 0, cen_size: str = "1 kpc", disk_size: str = "5 kpc") -> dict:
    """also orientates!"""
    print(f"Calculating orientation of Subhalo {sub_id}")
    h0 = Sim.halos(subhalos=True)[sub_id]
    x_cen = analysis.halo.center(h0, retcen=True, cen_size=cen_size, with_velocity=False)
    with transformation.inverse_translate(Sim, x_cen):
        h0 = Sim.halos(subhalos=True)[sub_id]
        v_cen = analysis.halo.vel_center(h0, cen_size=cen_size, retcen=True)
        with transformation.inverse_v_translate(Sim, v_cen):
            h0 = Sim.halos(subhalos=True)[sub_id]
            # Why Gas?
            # Use gas from inner 5kpc to calculate angular momentum vector
            gas_central = h0.gas[filt.Sphere(disk_size)]
            z_vec = analysis.angmom.ang_mom_vec(gas_central)
            z_vec /= np.linalg.norm(z_vec)
            z_Rot = analysis.angmom.calc_faceon_matrix(z_vec)
    orientation = {"x_cen": x_cen, "v_cen": v_cen, "z_Rot": z_Rot}
    return orientation


def save_pynbody_orientation(fname, orientation) -> None:
    print("Saving orientation...")
    with h5py.File(fname, "w") as hf:
        for p, x in orientation.items():
            hf.create_dataset(p, data=x)
    print("Saved orientation!")


def save_pynbody_orientation_sim(Sim) -> None:
    fname = f"{Sim.analysis_folder}pynbody_orientation_{Sim.orientation_name}.hdf5"
    orientation = Sim.orientation
    save_pynbody_orientation(fname, orientation)
