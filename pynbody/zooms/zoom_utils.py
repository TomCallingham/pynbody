import numpy as np
from collections.abc import Callable
from functools import wraps
from ..array import SimArray
from ..snapshot.subsnap import FamilySubSnap, HierarchyIndexedSubSnap, SubSnapBase

def top_hierarchy_family(func) -> Callable:
    ''' If the function must first be loaded data to the top of the hierarchy, like pynbody group_ids,
    this func ensures it is first loaded at the top before filtering down'''
    @wraps(func)
    def wrapper(sim):
        if not issubclass(type(sim),SubSnapBase):
            return func(sim)
        elif issubclass(type(sim),HierarchyIndexedSubSnap):
            anc = sim.ancestor
            if hasattr(sim, "_unifamily") and sim._unifamily is not None:
                fam = sim._unifamily
                index = sim.ancestors_index[anc]-anc._get_family_slice(fam).start
                anc = sim.ancestor[fam]
            else:
                index = [sim.ancestors_index[anc]]
            result = func(anc)[index]
            return result
        elif hasattr(sim, "_unifamily") and sim._unifamily is not None:
            print("Watchout, FamilySubSnap _unifamily using top array...")
            fam = sim._unifamily
            anc = sim.ancestor[fam]
            result = func(anc)
            if hasattr(sim, "_family_indices"):
                result = result[sim._family_indices[fam]]
            return result
        else:
            snap_type = type(sim)
            raise AttributeError(f"Unrecognised type: {snap_type}. Not a Main Snapshot, FamilySubSnap, or HierarchyIndexedSubSnap")
    return wrapper

def match_saved_sorted(sim_snap, saved_data,load_keys=None,p_id_key="iord") -> dict:

    sim_families = sim_snap.families()
    fam0 = sim_families[0]
    str_fam0 = str(fam0)
    #star or stars!
    # load_keys = [key for key in list(saved_data[str_fam0].keys()) if key!=p_id_key] if load_keys is None else load_keys
    load_keys = list(saved_data[str_fam0].keys()) if load_keys is None else load_keys
    n_part = len(sim_snap)
    data = {}
    for xkey in load_keys:
        x = saved_data[str_fam0][xkey]
        ndim,dtype = len(x.shape),x.dtype
        dims = (n_part) if ndim ==1 else  (n_part,ndim)
        if "int" in str(dtype):
            data[xkey] = -np.ones(dims, dtype=int)
        else:
            data[xkey] = np.full(dims, np.nan, dtype=float)
    all_p_ids = sim_snap["iord"].v
    for fam in sim_snap.families():
        fam_slice = sim_snap._get_family_slice(fam)
        fam_str = str(fam)
        saved_ids = saved_data[fam_str][p_id_key]
        p_ids = all_p_ids[fam_slice]

        p_match = np.isin(p_ids, saved_ids)
        z_match = np.isin(saved_ids, p_ids[p_match])
        p_argsort = np.argsort(p_ids[p_match])
        p_indexes = np.where(p_match)[0][p_argsort]
        for xkey in load_keys:
            data[xkey][p_indexes] = saved_data[fam_str][xkey][z_match]

    return data

def multiple_read(sim, data_dict: dict[str,SimArray], read_key:str) -> SimArray:
    """adds multiple properties to the sim at once, returning the chosen value"""
    props = [p for p in list(data_dict.keys()) if p != read_key]
    if not isinstance(sim,FamilySubSnap):
        for p in props:
            sim._arrays[p] = data_dict[p]
        return data_dict[read_key]

    fam = sim._unifamily
    base = sim.ancestor 
    for p in props:
        if p in base._family_arrays:
            base._family_arrays[p][fam] = data_dict[p]
        else:
            base._family_arrays[p] = {fam: data_dict[p]}

    return data_dict[read_key]
