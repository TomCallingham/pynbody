from collections.abc import Callable
import os
from .. import analysis, filt, transformation
import numpy as np
import h5py


def orientate_snap(self) -> None:
    folder = self.analysis_folder
    fname = folder + "pynbody_orientation.hdf5"
    if os.path.isfile(fname):
        with h5py.File(fname, "r") as hf:
            orientation = {p: hf[p][:] for p in hf.keys()}  # type: ignore
        apply_pynbody_orientation(self, orientation)
    else:
        print("creating orienation")
        orientation = calc_apply_pynbody_orientation(self)
        print("Saving orientaiton")
        save_pynbody_orientation(folder, orientation)


def calc_apply_pynbody_orientation(sim_snap, cen_size="1 kpc", disk_size="5 kpc") -> dict:
    """also orientates!"""
    h0 = sim_snap.halos()[0]
    x_cen = analysis.halo.center(h0, retcen=True, cen_size=cen_size)
    tx = transformation.inverse_translate(sim_snap, x_cen)
    v_cen = analysis.halo.vel_center(h0, cen_size=cen_size, retcen=True)
    tx = transformation.inverse_v_translate(tx, v_cen)
    #Why Gas?
    # Use gas from inner 5kpc to calculate angular momentum vector
    gas_central = h0.gas[filt.Sphere(disk_size)]
    z_vec = analysis.angmom.ang_mom_vec(gas_central)
    z_vec /= np.linalg.norm(z_vec)
    z_Rot = analysis.angmom.calc_faceon_matrix(z_vec)
    tx = transformation.transform(tx, z_Rot)
    orientation = {"x_cen": x_cen, "v_cen": v_cen, "z_Rot": z_Rot}
    return orientation


def apply_pynbody_orientation(sim_snap, orientation) -> None:
    """also orientates!"""
    # Could make all pynbody orientations lazy
    x_cen, v_cen, z_Rot = orientation["x_cen"], orientation["v_cen"], orientation["z_Rot"]
    tx = transformation.inverse_xv_translate(sim_snap, x_shift=x_cen, v_shift=v_cen)
    tx = transformation.transform(tx, z_Rot)


def lazy_orientate_snap(self) -> None:
    fname = self.analysis_folder + "pynbody_orientation.hdf5"
    if os.path.isfile(fname):
        try:
            with h5py.File(fname, "r") as hf:
                orientation = {p: hf[p][:] for p in hf.keys()}  # type: ignore
            self.lazy_orient = lazy_apply_pynbody_orientation(orientation)
            return
        except Exception as e:
            print(e)
            print("Can't load existing orientation, remaking?")

    try:
        print("creating and applying orienation")
        orientation = calc_apply_pynbody_orientation(self)
        print("Saving orientaiton")
        save_pynbody_orientation(self.analysis_folder, orientation)
        self.lazy_orient = orientation
    except Exception as e:
        print(e)
        print(f"Orient failed, {self}")


def lazy_apply_pynbody_orientation(orientation) -> Callable:
    """also orientates!"""
    x_cen, v_cen, z_Rot = orientation["x_cen"], orientation["v_cen"], orientation["z_Rot"]

    def translate_snap(sim_snap):
        tx = transformation.inverse_xv_translate(sim_snap, x_shift=x_cen, v_shift=v_cen)
        tx = transformation.transform(tx, z_Rot)

    return translate_snap


def save_pynbody_orientation(folder, orientation) -> None:
    fname = folder + "pynbody_orientation.hdf5"
    with h5py.File(fname, "w") as hf:
        for p, x in orientation.items():
            hf.create_dataset(p, data=x)
    print("saved")
