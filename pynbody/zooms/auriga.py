from pynbody.halo.subfindhdf import ArepoSubfindHDFCatalogue, SubFindHDFSubHalo, SubFindHDFSubhaloCatalogue
from pynbody.snapshot.gadgethdf import GadgetHDFSnap, GadgetHdfMultiFileManager
from pynbody.halo import Halo
from pynbody.snapshot import IndexedSubSnap
from pynbody.zooms.zoom import ZoomSnap

import pynbody
import numpy as np

import h5py


class AurigaStarsWind:

    def __init__(self) -> None:
        self._stars = None
        self._winds = None

    @property
    def stars(self) -> IndexedSubSnap:
        if self._stars is None:
            stars = self.s  # pyright: ignore
            self._stars = stars[stars["GFM_StellarFormationTime"] > 0]
        return self._stars

    @property
    def winds(self) -> IndexedSubSnap:
        if self._winds is None:
            stars = self.s  # pyright: ignore
            self._winds = stars[stars["GFM_StellarFormationTime"] < 0]
        return self._winds


class AurigaSubfindHdfMultiFileManager(GadgetHdfMultiFileManager):
    _nfiles_attrname = "NumFiles"

    def __init__(self, filename, mode='r'):
        super().__init__(filename, mode)
        to_del = []
        for i, file in enumerate(self._filenames):
            with h5py.File(file) as hf:
                if len(hf["Group"].keys()) == 0:  # type: ignore
                    to_del.append(i)
        for i in to_del[::-1]:
            del self._filenames[i]
        self._numfiles -= len(to_del)


class AurigaSubfindHDFCatalogue(ArepoSubfindHDFCatalogue):

    def __init__(self, sim, subs=False, grp_array=False):
        self.halo_file = sim.halo_file
        super().__init__(sim, subs=subs, grp_array=grp_array)
        self.physical_units()

    def _get_catalogue_multifile(self, _):
        return AurigaSubfindHdfMultiFileManager(self.halo_file)

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self) - 1:
            description = "Subhalo" if self._sub_mode else "Group"
            raise ValueError(f"{description} {i} does not exist")

        type_map = self.base._family_to_group_map

        if self._sub_mode:
            lengths = self._subfind_halo_lengths
            offsets = self._subfind_halo_offsets
        else:
            lengths = self._fof_group_lengths
            offsets = self._fof_group_offsets

        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()):
            for g_ptype in g_ptypes:
                tot_len += lengths[g_ptype][i]

        plist = np.zeros(tot_len, dtype='int64')

        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap
            sl = self.base._family_slice[ptype]

            for g_ptype in type_map[ptype]:
                # add the particle indices to the particle list
                offset = offsets[g_ptype][i]
                length = lengths[g_ptype][i]
                ind = np.arange(sl.start + offset, sl.start + offset + length)
                plist[npart:npart + length] = ind
                npart += length

        if self._sub_mode:
            # SubFindHDFSubHalo wants to know our parent group
            group, halo = self._group_and_halo_from_halo_index(i)
            return AurigaSubFindHDFSubHalo(halo, group, self, self, self.base, plist)
        else:
            return AurigaSubFindFOFGroup(i, self, self.base, plist)


class AurigaSubFindHDFSubHalo(SubFindHDFSubHalo, AurigaStarsWind):
    def __init__(self, halo_id, group_id, subfind_data_object, *args):
        SubFindHDFSubHalo.__init__(self, halo_id, group_id, subfind_data_object, *args)
        AurigaStarsWind.__init__(self)
        # self.physical_units()


class AurigaSubFindHDFSubhaloCatalogue(SubFindHDFSubhaloCatalogue, AurigaStarsWind):
    """
    Gadget's SubFind HDF Subhalo catalogue.

    Initialized with the parent FOF group catalogue and created
    automatically when an fof group is created
    """

    def __init__(self, group_id, group_catalogue):
        SubFindHDFSubhaloCatalogue.__init__(self, group_id, group_catalogue)
        AurigaStarsWind.__init__(self)
        # self.physical_units()

    def _get_halo(self, i):
        if self.base is None:
            raise RuntimeError("Parent SimSnap has been deleted")

        if i > len(self) - 1:
            raise ValueError("FOF group %d does not have subhalo %d" % (self._group_id, i))

        # need this to index the global offset and length arrays
        absolute_id = self._group_catalogue._fof_group_first_subhalo[self._group_id] + i

        # now form the particle IDs needed for this subhalo
        type_map = self.base._family_to_group_map

        halo_lengths = self._group_catalogue._subfind_halo_lengths
        halo_offsets = self._group_catalogue._subfind_halo_offsets

        # create the particle lists
        tot_len = 0
        for g_ptypes in list(type_map.values()):
            for g_ptype in g_ptypes:
                tot_len += halo_lengths[g_ptype][absolute_id]

        plist = np.zeros(tot_len, dtype='int64')

        npart = 0
        for ptype in self.base._families_ordered():
            # family slice in the SubFindHDFSnap
            sl = self.base._family_slice[ptype]

            for g_ptype in type_map[ptype]:
                # add the particle indices to the particle list
                offset = halo_offsets[g_ptype][absolute_id]
                length = halo_lengths[g_ptype][absolute_id]
                ind = np.arange(sl.start + offset, sl.start + offset + length)
                plist[npart:npart + length] = ind
                npart += length

        return AurigaSubFindHDFSubHalo(
            i, self._group_id, self._group_catalogue, self, self.base, plist)


class AurigaSubFindFOFGroup(Halo, AurigaStarsWind):
    """
    SubFind FOF group class, modified for Auriga
    """

    def __init__(self, group_id, *args):
        """Construct a special halo representing subfind's FOF group"""
        Halo.__init__(self, group_id, *args)

        self._subhalo_catalogue = AurigaSubFindHDFSubhaloCatalogue(
            group_id, self._halo_catalogue)

        self._descriptor = "fof_group_" + str(group_id)

        self.properties.update(self._halo_catalogue.get_halo_properties(group_id, subs=False))

        AurigaStarsWind.__init__(self)
        # self.physical_units()

    def __getattr__(self, name):
        if name == 'sub':
            return self._subhalo_catalogue
        else:
            return Halo.__getattr__(self, name)


auriga_eps = {4: 369 * pynbody.units.pc,
              3: 184 * pynbody.units.pc}


class AurigaLikeHDFSnap(GadgetHDFSnap, AurigaStarsWind, ZoomSnap):
    """Reads AurigaHDF"""
    _readable_hdf5_test_key = "PartType1/SubGroupNumber"

    def __init__(self, particle_filename, halo_filename,
                 analysis_folder, level=4, orientate=True):
        GadgetHDFSnap.__init__(self, particle_filename)
        self.halo_file = halo_filename
        self.analysis_folder = analysis_folder
        self.properties["eps"] = auriga_eps.get(level, None)
        AurigaStarsWind.__init__(self)
        # self.physical_units()
        ZoomSnap.__init__(self, orientate)

    def halos(self, subs=False) -> AurigaSubfindHDFCatalogue:
        """Load the Auriga FOF halos.
            Access halo with halo[0]
            Access Subhalo with halo[0].sub[0]
            Or, if subs=True, halo[0] is the first subhalo"""
        return AurigaSubfindHDFCatalogue(self, subs=subs)

    def derivable_keys(self) -> list:
        """Returns a list of arrays which can be lazy-evaluated."""
        keys = GadgetHDFSnap.derivable_keys(self)
        res = [key for key in keys if key not in auriga_bad_keys]
        return res


auriga_bad_keys = ['V_lum_den', 'dm', 'OI', 'i_mag', 'r_mag', 'rho_ne', 'nxh',
                   'NV', 'R_lum_den', 'HeIII', 'ne', 'R_mag', 'OVI', 'mgxh', 'j_mag',
                   'caxh', 'U_mag', 'hydrogen', 'J_lum_den', 'c_n_sq', 'nefe',
                   'MGII', 'OII', 'zeldovich_offset', 'sife',
                   'H_mag', 'feh', 'redshift', 'CIV', 'k_mag', 'ofe',
                   'I_mag', 'K_lum_den', 'aform', 'r_lum_den', 'HeII',
                   'sixh', 'em', 'HeI', 'j_lum_den', 'J_mag', 'U_lum_den',
                   'cxh', 'i_lum_den', 'u_mag', 'B_lum_den', 'v_lum_den', 'p',
                   'cosmodm', 'h_lum_den', 'SIV', 'B_mag', 'HID12', 'I_lum_den',
                   'b_lum_den', 'u_lum_den', 'hetot', 'HIeos', 'HII', 'sxh',
                   'V_mag', 'v_mag', 'k_lum_den', 'h_mag', 'b_mag', 'H_lum_den', 'HI',
                   'nexh', 'mgfe', 'doppler_redshift', 'oxh', 'halpha', 'K_mag']
from . import auriga_attributes
