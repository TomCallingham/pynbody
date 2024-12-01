import numpy as np

import pynbody.util.indexing_tricks
from pynbody import filt, util
from pynbody.snapshot import SimSnap

from .. import array, family

import inspect


class ExposedBaseSnapshotMixin:
    # The following will be objects common to a SimSnap and all its SubSnaps
    _inherited = [
        "_immediate_cache_lock",
        "lazy_off",
        "lazy_derive_off",
        "lazy_load_off",
        "auto_propagate_off",
        "properties",
        "_derived_array_names",
        "_family_derived_array_names",
        "_dependency_tracker",
        "immediate_mode",
        "delay_promotion",
        # ME
        "hierarchy",
    ]

    def __init__(self, base: SimSnap, *args, **kwargs):
        self.base = base
        super().__init__(base, *args, **kwargs)

    def _inherit(self):
        self._file_units_system = self.base._file_units_system
        self._unifamily = self.base._unifamily
        self._get_array_lock = self.base._get_array_lock

        for x in self._inherited:
            if hasattr(self.base, x):
                setattr(self, x, getattr(self.base, x))

        # ME. Get auriga inherit down
        if hasattr(self.base, "_subfunc_inherit"):
            setattr(self, "_subfunc_inherit", getattr(self.base, "_subfunc_inherit"))
            for x in self.base._subfunc_inherit:
                if not hasattr(self.ancestor, x):
                    continue
                subfunc = getattr(self.ancestor, x)

                def my_subfunc(*args, **kwargs):
                    return subfunc(*args, **kwargs, subsnap=self)

                my_subfunc.__doc__ = subfunc.__doc__
                my_subfunc.__signature__ = inspect.signature(subfunc)

                setattr(self, x, my_subfunc)


class SubSnapBase(SimSnap):
    def __init__(self, base):
        super().__init__()
        self._subsnap_base = base
        self._transformations = base._transformations

    def _get_array(self, name, index=None, always_writable=False):
        if self.immediate_mode:
            return self._get_from_immediate_cache(
                name, lambda: self._subsnap_base._get_array(name, None, always_writable)[self._slice]
            )

        else:
            ret = self._subsnap_base._get_array(
                name, pynbody.util.indexing_tricks.concatenate_indexing(self._slice, index), always_writable
            )
            ret.family = self._unifamily
            return ret

    def _set_array(self, name, value, index=None):
        self._subsnap_base._set_array(
            name, value, pynbody.util.indexing_tricks.concatenate_indexing(self._slice, index)
        )

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        base_family_slice = self._subsnap_base._get_family_slice(fam)
        sl = pynbody.util.indexing_tricks.relative_slice(
            base_family_slice,
            pynbody.util.indexing_tricks.intersect_slices(self._slice, base_family_slice, len(self._subsnap_base)),
        )
        sl = pynbody.util.indexing_tricks.concatenate_indexing(sl, index)
        if self.immediate_mode:
            return self._get_from_immediate_cache(
                (name, fam), lambda: self._subsnap_base._get_family_array(name, fam, None, always_writable)[sl]
            )
        else:
            return self._subsnap_base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        fslice = self._get_family_slice(family)
        self._subsnap_base._set_family_array(
            name, family, value, pynbody.util.indexing_tricks.concatenate_indexing(fslice, index)
        )

    def _promote_family_array(self, *args, **kwargs):
        self._subsnap_base._promote_family_array(*args, **kwargs)

    def __delitem__(self, name):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    def _del_family_array(self, name, family):
        # is this the right behaviour?
        raise RuntimeError("Arrays can only be deleted from the base snapshot")

    @property
    def _filename(self):
        return str(self._subsnap_base._filename) + ":" + self._descriptor

    def keys(self):
        return list(self._subsnap_base.keys())

    def loadable_keys(self, fam=None):
        if self._unifamily:
            return self._subsnap_base.loadable_keys(self._unifamily)
        else:
            return self._subsnap_base.loadable_keys(fam)

    def derivable_keys(self):
        return self._subsnap_base.derivable_keys()

    def infer_original_units(self, *args):
        """Return the units on disk for a quantity with the specified dimensions"""
        return self._subsnap_base.infer_original_units(*args)

    def _get_family_slice(self, fam):
        sl = pynbody.util.indexing_tricks.relative_slice(
            self._slice,
            pynbody.util.indexing_tricks.intersect_slices(
                self._slice, self._subsnap_base._get_family_slice(fam), len(self._subsnap_base)
            ),
        )
        return sl

    def _load_array(self, array_name, fam=None):  # , **kwargs):
        self._subsnap_base._load_array(array_name, fam)

    def write_array(self, array_name, fam=None, **kwargs):
        fam = fam or self._unifamily
        if not fam or self._get_family_slice(fam) != slice(0, len(self)):
            raise OSError(
                "Array writing is available for entire simulation arrays or family-level arrays, but not for arbitrary subarrays"
            )

        self._subsnap_base.write_array(array_name, fam=fam, **kwargs)

    def _derive_array(self, array_name, fam=None):
        self._subsnap_base._derive_array(array_name, fam)

    def family_keys(self, fam=None):
        return self._subsnap_base.family_keys(fam)

    def _create_array(self, *args, **kwargs):
        self._subsnap_base._create_array(*args, **kwargs)

    def _create_family_array(self, *args, **kwargs):
        self._subsnap_base._create_family_array(*args, **kwargs)

    def physical_units(self, *args, **kwargs):
        self._subsnap_base.physical_units(*args, **kwargs)

    def is_derived_array(self, array_name, fam=None):
        fam = fam or self._unifamily
        return self._subsnap_base.is_derived_array(array_name, fam)

    def find_deriving_function(self, name):
        return self._subsnap_base.find_deriving_function(name)

    def unlink_array(self, name):
        self._subsnap_base.unlink_array(name)

    def get_index_list(self, relative_to, of_particles=None):
        if of_particles is None:
            of_particles = np.arange(len(self))

        if relative_to is self:
            return of_particles

        return self._subsnap_base.get_index_list(
            relative_to, pynbody.util.indexing_tricks.concatenate_indexing(self._slice, of_particles)
        )


class SubSnap(ExposedBaseSnapshotMixin, SubSnapBase):
    """Represent a sub-view of a SimSnap, initialized by specifying a
    slice.  Arrays accessed through __getitem__ are automatically
    sub-viewed using the given slice."""

    def __init__(self, base, _slice):
        super().__init__(base)
        self._inherit()

        if isinstance(_slice, slice):
            # Various slice logic later (in particular taking
            # subsnaps-of-subsnaps) requires having positive
            # (i.e. start-relative) slices, so if we have been passed a
            # negative (end-relative) index, fix that now.
            if _slice.start is None:
                _slice = slice(0, _slice.stop, _slice.step)
            if _slice.start < 0:
                _slice = slice(len(base) + _slice.start, _slice.stop, _slice.step)
            if _slice.stop is None or _slice.stop > len(base):
                _slice = slice(_slice.start, len(base), _slice.step)
            if _slice.stop < 0:
                _slice = slice(_slice.start, len(base) + _slice.stop, _slice.step)

            self._slice = _slice

            descriptor = "[" + str(_slice.start) + ":" + str(_slice.stop)
            if _slice.step is not None:
                descriptor += ":" + str(_slice.step)
            descriptor += "]"

        else:
            raise TypeError("Unknown SubSnap slice type")

        self._num_particles = pynbody.util.indexing_tricks.indexing_length(_slice)

        self._descriptor = descriptor


class IndexingViewMixin:
    def __init__(self, *args, **kwargs):
        index_array = kwargs.pop("index_array", None)
        iord_array = kwargs.pop("iord_array", None)
        allow_family_sort = kwargs.pop("allow_family_sort", False)

        super().__init__(*args, **kwargs)
        self._descriptor = "indexed"

        self._unifamily = self._subsnap_base._unifamily
        self._file_units_system = self._subsnap_base._file_units_system

        if index_array is None and iord_array is None:
            raise ValueError("Cannot define a subsnap without an index_array or iord_array.")
        if index_array is not None and iord_array is not None:
            # typo, without -> with
            raise ValueError("Cannot define a subsnap with both and index_array and iord_array.")
        if iord_array is not None:
            index_array = self._iord_to_index(iord_array)

        if isinstance(index_array, filt.Filter):
            self._descriptor = "filtered"
            index_array = index_array.where(self._subsnap_base)[0]

        elif isinstance(index_array, tuple):
            if isinstance(index_array[0], np.ndarray):
                index_array = index_array[0]
            else:
                index_array = np.array(index_array)
        else:
            index_array = np.asarray(index_array)

        findex = self._subsnap_base._family_index()[index_array]

        if allow_family_sort:
            sort_ar = np.argsort(findex)
            index_array = index_array[sort_ar]
            findex = findex[sort_ar]
        elif not all(np.diff(findex) >= 0):
            # Check the family index array is monotonically increasing
            # If not, the family slices cannot be implemented
            # TODO: This seems expnsive?
            raise ValueError("Families must retain the same ordering in the SubSnap")

        self._slice = index_array
        self._family_slice = {}
        self._family_indices = {}
        self._num_particles = len(index_array)

        # Find the locations of the family slices
        for i, fam in enumerate(self._subsnap_base.ancestor.families()):
            ids = np.where(findex == i)[0]
            if len(ids) > 0:
                new_slice = slice(ids.min(), ids.max() + 1)
                self._family_slice[fam] = new_slice
                self._family_indices[fam] = (
                    np.asarray(index_array[new_slice]) - self._subsnap_base._get_family_slice(fam).start
                )

    def _iord_to_index(self, iord):
        # Maps iord to indices. Note that this requires to perform an argsort (O(N log N) operations)
        # and a binary search (O(M log N) operations) with M = len(iord) and N = len(self._subsnap_base).

        # Find index of particles using a search sort
        iord_base = self._subsnap_base["iord"].v
        iord_base_argsort = self._subsnap_base["iord_argsort"].v

        dtype = np.uint64
        iord = np.ascontiguousarray(iord, dtype)
        iord_base = np.ascontiguousarray(iord_base, dtype)
        iord_base_argsort = np.ascontiguousarray(iord_base_argsort, dtype)

        if not util.is_sorted(iord) == 1:
            raise Exception("Expected iord to be sorted in increasing order.")

        index_array = util.binary_search(a=iord, b=iord_base, sorter=iord_base_argsort)

        # Check that the iord match
        if np.any(index_array == len(iord_base)):
            raise Exception("Some of the requested ids cannot be found in the dataset.")

        return index_array


class IndexedSubSnap(IndexingViewMixin, ExposedBaseSnapshotMixin, SubSnapBase):
    """Represents a subset of the simulation particles according
    to an index array.

    Parameters
    ----------
    base : SimSnap object
        The base snapshot
    index_array : integer array or None
        The indices of the elements that define the sub snapshot. Set to None to use iord-based instead.
    iord_array : integer array or None
        The iord of the elements that define the sub snapshot. Set to None to use index-based instead.
        This may be computationally expensive. See note below.

    Notes
    -----
    `index_array` and `iord_array` arguments are mutually exclusive.
    In the case of `iord_array`, an sorting operation is required that may take
    a significant time and require O(N) memory.
    """

    def __init__(self, base, index_array=None, iord_array=None, *args, **kwargs):
        super().__init__(base, index_array=index_array, iord_array=iord_array, *args, **kwargs)
        self._inherit()

    def _get_family_slice(self, fam):
        # A bit messy: jump out the SubSnap inheritance chain
        # and call SimSnap method directly...
        return SimSnap._get_family_slice(self, fam)

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        sl = self._family_indices.get(fam, slice(0, 0))
        sl = pynbody.util.indexing_tricks.concatenate_indexing(sl, index)

        return self._subsnap_base._get_family_array(name, fam, sl, always_writable)

    def _set_family_array(self, name, family, value, index=None):
        self._subsnap_base._set_family_array(
            name, family, value, pynbody.util.indexing_tricks.concatenate_indexing(self._family_indices[family], index)
        )

    def _create_array(self, *args, **kwargs):
        self._subsnap_base._create_array(*args, **kwargs)


class FamilySubSnap(SubSnap):
    """Represents a one-family portion of a parent snap object"""

    def __init__(self, base, fam):
        super().__init__(base, base._get_family_slice(fam))

        self._unifamily = fam
        self._descriptor = ":" + fam.name

    def __delitem__(self, name):
        if name in list(self._subsnap_base.keys()):
            raise ValueError("Cannot delete global simulation property from sub-view")
        elif name in self._subsnap_base.family_keys(self._unifamily):
            self._subsnap_base._del_family_array(name, self._unifamily)

    def keys(self):
        global_keys = list(self._subsnap_base.keys())
        family_keys = self._subsnap_base.family_keys(self._unifamily)
        return list(set(global_keys).union(family_keys))

    def family_keys(self, fam=None):
        # We now define there to be no family-specific subproperties,
        # because all properties can be accessed through standard
        # __setitem__, __getitem__ methods
        return []

    def _get_family_slice(self, fam):
        if fam is self._unifamily:
            return slice(0, len(self))
        else:
            return slice(0, 0)

    def _get_array(self, name, index=None, always_writable=False):
        try:
            return SubSnap._get_array(self, name, index, always_writable)
        except KeyError:
            return self._subsnap_base._get_family_array(name, self._unifamily, index, always_writable)

    def _create_array(self, array_name, ndim=1, dtype=None, zeros=False, derived=False, shared=None):
        # Array creation now maps into family-array creation in the parent
        self._subsnap_base._create_family_array(array_name, self._unifamily, ndim, dtype, zeros, derived, shared)

    def _set_array(self, name, value, index=None):
        if name in list(self._subsnap_base.keys()):
            self._subsnap_base._set_array(
                name, value, pynbody.util.indexing_tricks.concatenate_indexing(self._slice, index)
            )
        else:
            self._subsnap_base._set_family_array(name, self._unifamily, value, index)

    def _create_family_array(self, array_name, family, ndim, dtype, derived, shared):
        self._subsnap_base._create_family_array(array_name, family, ndim, dtype, derived, shared)

    def _promote_family_array(self, *args, **kwargs):
        pass

    def _load_array(self, array_name, fam=None):  # , **kwargs):
        if fam is self._unifamily or fam is None:
            self._subsnap_base._load_array(array_name, self._unifamily)

    def _derive_array(self, array_name, fam=None):
        if fam is self._unifamily or fam is None:
            self._subsnap_base._derive_array(array_name, self._unifamily)


class HierarchyIndexedSubSnap(IndexingViewMixin, ExposedBaseSnapshotMixin, SubSnapBase):
    """Represents a subset of the simulation particles according
    to an index array.

    Parameters
    ----------
    base : SimSnap object
        The base snapshot
    index_array : integer array or None
        The indices of the elements that define the sub snapshot. Set to None to use iord-based instead.
    iord_array : integer array or None
        The iord of the elements that define the sub snapshot. Set to None to use index-based instead.
        This may be computationally expensive. See note below.

    Notes
    -----
    `index_array` and `iord_array` arguments are mutually exclusive.
    In the case of `iord_array`, an sorting operation is required that may take
    a significant time and require O(N) memory.
    """

    def __init__(self, base, index_array=None, iord_array=None, *args, **kwargs):
        super().__init__(base, index_array=index_array, iord_array=iord_array, *args, **kwargs)
        self._inherit()

        self._arrays = {}
        # self._family_arrays = {}  # NOTE: Needed, else orientation fails...
        # self.ancestor_family=None
        #
        self._filt_load = True
        self._filt_load_helper = None
        # self._init_master()
        self._init_ancestors_arrays()
        self._init_ancestors_index()

    # def _init_master(self):
    #     if isinstance(self._subsnap_base, HierarchyIndexedSubSnap):
    #         if self._subsnap_base.master:
    #             self.master_subsnap = self._subsnap_base
    #         elif self._subsnap_base.master_subsnap:
    #             self.master_subsnap = self._subsnap_base.master_subsnap
    #     else:
    #         self.master_subsnap = None
    #     self.master = False

    def _init_ancestors_arrays(self):
        # _ancestor_of_arrays  {array_key:ancestors}
        # TODO: Make lazy, don't need to calculate every link to every one
        # TODO: Use weakrefs, so intemediates are deletable
        if isinstance(self._subsnap_base, HierarchyIndexedSubSnap):
            old_dic = dict(self._subsnap_base._ancestors_of_arrays)
            new_dic = {key: self._subsnap_base for key in self._subsnap_base._arrays}
            self._ancestors_of_arrays = new_dic | old_dic
            # TODO: check this is a shallow copy!
        elif isinstance(self._subsnap_base, FamilySubSnap):
            self._ancestors_of_arrays = {key: self._subsnap_base.ancestor for key in self._subsnap_base.ancestor.keys()}
        else:
            self._ancestors_of_arrays = {key: self._subsnap_base for key in self._subsnap_base.keys()}

    def _init_ancestors_index(self):
        # ancestors_index|  {ancestors:slice }
        if isinstance(self._subsnap_base, HierarchyIndexedSubSnap):
            self.ancestors_index = {self._subsnap_base: self._slice}
            for ancestor in self._subsnap_base.ancestors_index:  # .keys():
                self.ancestors_index[ancestor] = self._subsnap_base.ancestors_index[ancestor][self._slice]
        elif isinstance(self._subsnap_base, FamilySubSnap):
            start = self.ancestor._get_family_slice(self._subsnap_base._unifamily).start
            stop = self.ancestor._get_family_slice(self._subsnap_base._unifamily).stop
            index = self._slice + start
            if np.min(self._slice) < 0:
                index[self._slice < 0] += stop - start
            self.ancestors_index = {self._subsnap_base.ancestor: index}
        else:
            self.ancestors_index = {self._subsnap_base: self._slice}
            pass
            # TODO: Fails if IndexedSubSnap that is not Hierarcical. Which should never happen, but...
            #

    def _load_array(self, array_name, fam=None, **kwargs):
        """If implemented, load filtered from files. Good for many hdf files.
        Alternatively, load from ancestor and filter
        """
        if hasattr(self.ancestor, "_load_array_filtered") and self._filt_load is True:
            self.ancestor._load_array_filtered(array_name, target=self, fam=fam)
            return
        self.ancestor._load_array(array_name, fam)

        if fam is None:
            self._arrays[array_name] = self.ancestor._arrays[array_name][self.ancestors_index[self.ancestor]]
            del self.ancestor._arrays[array_name]
        else:
            index = self.ancestors_index[self.ancestor] - self.ancestor._get_family_slice(fam).start
            self._arrays[array_name] = self.ancestor._family_arrays[array_name][fam][index]
            del self.ancestor._family_arrays[array_name][fam]
            if len(self.ancestor._family_arrays[array_name]) == 0:
                del self.ancestor._family_arrays[array_name]

    def _get_array(self, name, index=None, always_writable=False):
        """Retreves arrays that are already loaded in above hierarchy"""
        if name not in self.keys():
            # If not in keys, check that ancestors havent been updated!
            self._init_ancestors_arrays()

        if name in self._arrays:
            x = self._arrays[name]
        # elif self.master_subsnap:
        #     x = self.master_subsnap[name][self.ancestors_index[self.master_subsnap]]
        elif name in self._ancestors_of_arrays:
            ancestor_snap = self._ancestors_of_arrays[name]
            x = ancestor_snap[name][self.ancestors_index[ancestor_snap]]
            # if self.master:
            #     self._arrays[name] = x
        # Now Failing!
        elif name not in self.all_keys():
            raise KeyError(f"No known array {name} for any family")
        elif self._unifamily:
            raise KeyError(f"No array {name} for family {self._unifamily.name}")
        else:
            raise KeyError("Can't get array")
        if index is None:
            return x
        else:
            print("Getting Array in HierarchyIndexedSubSnap with an index?")
            return x[index]

    # def set_master(self):
    #     """Make selection the master collection of arrays. Useful for key subsets"""
    #     # TODO: Be able to select master level
    #     print("setting master!")
    #     self.master = True

    def keys(self):
        return list(self._arrays.keys()) + list(self._ancestors_of_arrays.keys())

    def _derive_array(self, array_name, fam=None):
        SimSnap._derive_array(self, array_name, fam)

    def _create_array(self, *args, **kwargs):
        SimSnap._create_array(self, *args, **kwargs)

    def _set_array(self, name, value, index=None):
        assert len(value) == len(self)
        self._arrays[name] = value

    def __delitem__(self, name):
        if name in self._arrays:
            del self._arrays[name]
            if name in self._derived_array_names:
                del self._derived_array_names[self._derived_array_names.index(name)]
        elif name in self._ancestors_of_arrays:
            del self._ancestors_of_arrays[name]
        # else:
        #     print(f"No array of name {name} found to delete?")

    def _get_family_slice(self, fam):
        fam_slice = self._family_slice.get(fam, slice(0, 0))
        return fam_slice

    def _get_family_array(self, name, fam, index=None, always_writable=False):
        # TODO: update
        # This should never be called, as Fams of this array array are HierarchyIndexedSubSnap?
        print("Calling get_family_array from Hierarcical  array, check!")
        sl = self._get_family_slice(fam)
        return self._get_array(name)[sl]

    def family_keys(self, fam=None):
        # Indexd SubSnaps don't do family arrays!
        # TODO: Check this - CAN have mixed family subsnaps.
        # print("In Hierarchy get fam keys")
        # We now define there to be no family-specific subproperties,
        # because all properties can be accessed through standard
        # __setitem__, __getitem__ methods
        return []

    def _set_family_array(self, name, family, value, index=None):
        raise AttributeError("Should not be setting family array from HierarchyIndexedSubSnap")

    def __getitem__(self, i) -> array.SimArray | SubSnapBase:
        if isinstance(i, family.Family):
            fam = i
            fam_filt = np.zeros(len(self), dtype=bool)
            fam_filt[self._family_slice[fam]] = True
            fam_snap = self[fam_filt]
            fam_snap._unifamily = fam
            fam_snap._descriptor = ":" + fam.name
            return fam_snap
        # elif self.master and isinstance(i, str) and i not in self._arrays:
        #     print("Hierarchy master get item")
        #     x = super().__getitem__(i)
        #     self._arrays[i] = x
        #     print("master force add!")
        #     print(self)
        #     return self._arrays[i]

        return super().__getitem__(i)
