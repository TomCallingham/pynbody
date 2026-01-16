import os
from .. import analysis, filt, transformation
import numpy as np
import h5py


def load_orientation_sub_id(Sim, sub_id: int = 0) -> dict[str, np.ndarray]:
    fname = f"{Sim.analysis_folder}pynbody_orientation_{Sim.orientation_name}.hdf5"
    if os.path.isfile(fname):
        with h5py.File(fname) as hf:
            orientation_dic = {p: np.asarray(hf[p][:]) for p in hf.keys()}
    else:
        orientation_dic = calc_apply_pynbody_orientation(Sim, sub_id)
        save_pynbody_orientation(fname, orientation_dic)

    return orientation_dic


def load_orientation_name(Sim) -> dict[str, np.ndarray]:
    fname = f"{Sim.analysis_folder}pynbody_orientation_{Sim.orientation_name}.hdf5"
    if os.path.isfile(fname):
        with h5py.File(fname) as hf:
            orientation_dic = {
                p: np.asarray(hf[p][:]) for p in ["x_cen", "v_cen", "z_Rot"]
            }
            orientation_dic["name"] = Sim.orientation_name
    else:
        raise FileExistsError(f"orientation {Sim.orientation_name} not found")
    return orientation_dic


def calc_apply_pynbody_orientation(
    Sim, sub_id: int = 0, cen_size: str = "1 kpc", disk_size: str = "5 kpc"
) -> dict:
    """also orientates!"""
    print(f"Calculating orientation of Subhalo {sub_id}")
    print("New analysis")
    h0 = Sim.halos(subhalos=True)[sub_id]
    x_cen = analysis.halo.center(
        h0, return_cen=True, cen_size=cen_size, with_velocity=False
    )
    print("xcen calculated")

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
            # z_Rot = analysis.angmom.calc_faceon_matrix(z_vec)
            z_Rot = rot_z_to_vec(z_vec)
    orientation = {"x_cen": x_cen, "v_cen": v_cen, "z_Rot": z_Rot}
    return orientation


def save_pynbody_orientation(fname, orientation) -> None:
    print("Saving orientation...")
    print(fname)
    with h5py.File(fname, "w") as hf:
        for p, x in orientation.items():
            hf.create_dataset(p, data=x)
    print("Saved orientation!")


def save_pynbody_orientation_sim(Sim) -> None:
    fname = f"{Sim.analysis_folder}pynbody_orientation_{Sim.orientation_name}.hdf5"
    orientation = Sim.orientation
    save_pynbody_orientation(fname, orientation)


import numpy as np


def rot_z_to_vec(z_new, z0=(0.0, 0.0, 1.0), eps=1e-12):
    print("New Rodrigues rot calc!")
    """
    Finds rotation to make z0 the z_new, but without rotation around z_new. 
    pynbody.analysis.angmom.calc_faceon_matrix(z_vec) can give some funny rotation...
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    z0 /= np.linalg.norm(z0)
    z_new /= np.linalg.norm(z_new)

    v = np.cross(z0, z_new)  # rotation axis * sin(theta)
    s = np.linalg.norm(v)  # sin(theta)
    c = np.dot(z0, z_new)  # cos(theta)

    if s < eps:
        # z0 and z1 are parallel or anti-parallel
        if c > 0:
            return np.eye(3)  # already aligned
        # 180° rotation: axis is not unique; choose a deterministic perpendicular axis
        # pick something perpendicular to z0
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, z0)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        axis = a - np.dot(a, z0) * z0
        axis /= np.linalg.norm(axis)
        # Rodrigues with theta = pi
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + 2 * (K @ K)  # since sin(pi)=0, (1-cos(pi))=2

    axis = v / s
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2, with sinθ=s, cosθ=c
    return np.eye(3) + s * K + (1 - c) * (K @ K)
