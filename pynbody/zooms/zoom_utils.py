import numpy as np
from collections.abc import Callable
from functools import wraps
from ..array import SimArray
from ..snapshot.subsnap import FamilySubSnap, HierarchyIndexedSubSnap, SubSnapBase


def get_fill_val(dtype):
    if np.issubdtype(dtype, np.floating):
        fill_val = np.nan
    elif np.issubdtype(dtype, np.signedinteger):
        fill_val = -1
    elif np.issubdtype(dtype, np.unsignedinteger):
        fill_val = 0
    elif np.issubdtype(dtype, np.bool):
        fill_val = False
    else:
        raise TypeError(f"Unknown dtype value {dtype}?")
    return fill_val


def top_hierarchy_family(func) -> Callable:
    """If the function must first be loaded data to the top of the hierarchy, like pynbody group_ids,
    this func ensures it is first loaded at the top before filtering down"""

    @wraps(func)
    def wrapper(sim):
        if not issubclass(type(sim), SubSnapBase):
            return func(sim)
        elif issubclass(type(sim), HierarchyIndexedSubSnap):
            anc = sim.ancestor
            if hasattr(sim, "_unifamily") and sim._unifamily is not None:
                fam = sim._unifamily
                index = sim.ancestors_index[anc] - anc._get_family_slice(fam).start
                anc = sim.ancestor[fam]
            else:
                index = [sim.ancestors_index[anc]]
            result = func(anc)[index]
            return result
        elif hasattr(sim, "_unifamily") and sim._unifamily is not None:
            fam = sim._unifamily
            anc = sim.ancestor[fam]
            result = func(anc)
            if hasattr(sim, "_family_indices"):
                result = result[sim._family_indices[fam]]
            return result
        else:
            snap_type = type(sim)
            raise AttributeError(
                f"Unrecognised type: {snap_type}. Not a Main Snapshot, FamilySubSnap, or HierarchyIndexedSubSnap"
            )

    return wrapper


def match_saved_sorted(sim, saved_data, load_keys=None, p_id_key="iord") -> dict:
    sim_families = sim.families()
    str_fam0 = str(sim_families[0])
    # star or stars!
    load_keys = load_keys if load_keys is not None else list(saved_data[str_fam0].keys())
    n_part = len(sim)
    data = {}
    for xkey in load_keys:
        x = saved_data[str_fam0][xkey]
        ndim, dtype = len(x.shape), x.dtype
        dims = (n_part) if ndim == 1 else (n_part, ndim)
        data[xkey] = np.full(dims, get_fill_val(dtype))
    all_p_ids = sim["iord"].v
    for fam in sim_families:
        fam_slice = sim._get_family_slice(fam)
        fam_str = str(fam)
        saved_ids = saved_data[fam_str][p_id_key]
        p_ids = all_p_ids[fam_slice]

        pos = np.searchsorted(saved_ids, p_ids)  # A,B
        pos[pos >= len(saved_ids)] = 0
        valid = saved_ids[pos] == p_ids

        # pos = np.searchsorted(saved_ids, p_ids)
        # mask = pos < len(saved_ids)
        # valid = np.zeros(p_ids.shape, dtype=bool)
        # valid[mask] = saved_ids[pos[mask]] == p_ids[mask]

        filt_B = np.nonzero(valid)[0]
        filt_A = pos[valid]

        for xkey in load_keys:
            data[xkey][filt_B] = saved_data[fam_str][xkey][filt_A]

    for xkey in load_keys:
        data[xkey] = data[xkey].view(SimArray)
        data[xkey].sim = SimArray

    return data


def multiple_read(sim, data_dict: dict[str, SimArray], read_key: str) -> SimArray:
    """adds multiple properties to the sim at once, returning the chosen value"""
    props = [p for p in list(data_dict.keys()) if p != read_key]
    if not isinstance(sim, FamilySubSnap):
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


def add_dic_data(Sim, data_dic, load_keys, p_id_key):
    matched_stars_peri_data = match_saved_sorted(Sim, data_dic, load_keys, p_id_key)
    for key, x in matched_stars_peri_data.items():
        Sim[key] = x
    return Sim
