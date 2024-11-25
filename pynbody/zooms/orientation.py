import os

# from .zoom import ZoomSnap
from .. import analysis, filt, transformation
import numpy as np
import h5py


def load_orientation(sim_snap, extra: str | None = None) -> dict[str, np.ndarray]:
    folder = sim_snap.analysis_folder
    fname = f"{folder}pynbody_orientation.hdf5" if extra is None else f"{folder}pynbody_orientation_{extra}.hdf5"
    if os.path.isfile(fname):
        with h5py.File(fname, "r") as hf:
            orientation_dic = {p: np.asarray(hf[p][:]) for p in hf.keys()}  # type: ignore
    else:
        print("creating orienation")
        orientation_dic = calc_apply_pynbody_orientation(sim_snap)
        print("Saving orientaiton")
        save_pynbody_orientation(folder, orientation_dic)

    return orientation_dic


def check_z_Rot(matrix: np.ndarray) -> None:
    ortho_tol = 1.0e-8
    resid = np.dot(matrix, np.asarray(matrix).T) - np.eye(3)
    resid = (resid**2).sum()
    if resid > ortho_tol:
        raise ValueError("Transformation matrix is not orthogonal")


def calc_apply_pynbody_orientation(sim_snap, cen_size: str = "1 kpc", disk_size: str = "5 kpc") -> dict:
    """also orientates!"""
    h0 = sim_snap.halos()[0]
    print("Assuming main halo on group_id=0!")
    x_cen = analysis.halo.center(h0, retcen=True, cen_size=cen_size)
    tx = transformation.inverse_translate(sim_snap, x_cen)
    v_cen = analysis.halo.vel_center(h0, cen_size=cen_size, retcen=True)
    tx = transformation.inverse_v_translate(tx, v_cen)
    # Why Gas?
    # Use gas from inner 5kpc to calculate angular momentum vector
    gas_central = h0.gas[filt.Sphere(disk_size)]
    z_vec = analysis.angmom.ang_mom_vec(gas_central)
    z_vec /= np.linalg.norm(z_vec)
    z_Rot = analysis.angmom.calc_faceon_matrix(z_vec)
    orientation = {"x_cen": x_cen, "v_cen": v_cen, "z_Rot": z_Rot}
    return orientation


def save_pynbody_orientation(folder, orientation) -> None:
    fname = folder + "pynbody_orientation.hdf5"
    with h5py.File(fname, "w") as hf:
        for p, x in orientation.items():
            hf.create_dataset(p, data=x)
    print("saved orientation")


def single_rotation(matrix, array):
    """Taken from pynbodies own Rotation translation"""
    assert array.shape[1] == 3
    array = np.dot(matrix, array.transpose()).transpose()
    return array
