import numpy as np
from collections.abc import Callable
import contextlib
import os
from functools import wraps
import h5py
from pynbody.array import SimArray
from pynbody import family
from pynbody.snapshot import FamilySubSnap

from .agama_potential import action_angles_props, calc_action_angles


def get_fam_str(sim) -> str:
    fams = sim.families()
    if len(fams) > 1:
        raise AttributeError("Only one family at a time for this property")
    return str(fams[0])


def cache_prop(func) -> Callable:
    @wraps(func)
    def wrapper(sim):
        base = sim.ancestor if hasattr(sim, "ancestor") else sim
        print(f"Use cache: {base.use_cache}")
        if not base.use_cache:
            return func(sim)
        fam = get_fam_str(sim)
        func_str = func.__name__
        if func_str in base.cached_props[fam]:
            print("loading from cache")
            return load_cached(sim, fam, func_str)
        result = func(sim)
        save_cached(sim, func_str, fam, result)
        return result
    return wrapper


def load_cached(sim, fam, func_str) -> SimArray:
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    file = f"{base.analysis_folder}{fam}_cached_properties.hdf5"
    with h5py.File(file, "r") as hf:
        result = hf[func_str]
        result_units = str(result.attrs["units"])
        result = SimArray(result[:])
    result.sim = sim
    if result_units != "NoUnit()":
        result.units = result_units
    return result


def load_cached_props(sim) -> dict:
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    fams = [str(f) for f in list(sim.families())]
    props = {}
    for fam in fams:
        file = f"{base.analysis_folder}{fam}_cached_properties.hdf5"
        if not os.path.isfile(file):
            props[fam] = []
        else:
            with h5py.File(file, "r") as hf:
                props[fam] = list(hf.keys())
    return props


def save_cached(sim, func_str, fam, result) -> SimArray:
    result_units = result.units
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    file = f"{base.analysis_folder}{fam}_cached_properties.hdf5"
    print(f"saving {func_str}...")
    with h5py.File(file, "a") as hf:
        if func_str in list(hf.keys()):
            del hf[func_str]
        dset = hf.create_dataset(f"{func_str}", data=result.view(np.ndarray))
        dset.attrs["units"] = str(result_units)
    base.cached_props[fam].append(func_str)
    print("saved!")
    return result


def del_cached_props(sim, fam, del_props) -> None:
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    file = f"{base.analysis_folder}{fam}_cached_properties.hdf5"
    print(f"deleting {del_props} for {fam}")
    with h5py.File(file, "a") as hf:
        for p in del_props:
            print(f"deleting {p}")
            with contextlib.suppress(KeyError):
                del hf[p]


family_dict = {'dm': family.dm, 'gas': family.gas, "star": family.star}


def multiple_read(sim, data_dict, read_key, save=False) -> SimArray:
    '''adds multiple properties to the sim at once, returning the chosen value'''
    # TODO: This is currently for one family only
    if save:
        save_multiple_cached(sim, data_dict)
    fam_str = get_fam_str(sim)
    fam = family_dict[fam_str]
    base = sim.base if isinstance(sim, FamilySubSnap) else sim
    props = [p for p in list(data_dict.keys()) if p != read_key]
    for p in props:
        if p in base._family_arrays:
            base._family_arrays[p][fam] = data_dict[p]
        else:
            base._family_arrays[p] = {fam: data_dict[p]}
    return data_dict[read_key]


def save_multiple_cached(sim, data_dict) -> None:
    base = sim.ancestor if hasattr(sim, "ancestor") else sim
    if not base.use_cache:
        return
    fam = get_fam_str(sim)
    for func_str, result in data_dict.items():
        save_cached(sim, func_str, fam, result)
