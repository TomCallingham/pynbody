from pynbody.halo.subfindhdf import ArepoSubfindHDFCatalogue, SubFindHDFHaloCatalogue
from pynbody.snapshot.gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager
from pynbody.halo import Halo

# from pynbody.snapshot import IndexedSubSnap
from pynbody.zooms.zoom import ZoomSnap

import pynbody
import numpy as np

import h5py


class AurigaStarsWind:
    def __init__(self) -> None:
        self._stars = None
        self._winds = None

    @property
    def stars(self):  # -> IndexedSubSnap:
        if self._stars is None:
            stars = self.s  # pyright: ignore
            self._stars = stars[stars["aform"] > 0]
        return self._stars

    @property
    def winds(self):  # -> IndexedSubSnap:
        if self._winds is None:
            stars = self.s  # pyright: ignore
            self._winds = stars[stars["aform"] < 0]
        return self._winds


class AurigaSubFindHdfMultiFileManager(_GadgetHdfMultiFileManager):
    _nfiles_attrname = "NumFiles"

    def __init__(self, filename, mode="r"):
        """Need to account for empty files in Auriga!"""
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
    # def __init__(self, sim, subs=False, subhalos=False, user_provided_filename=None):
    def __init__(self, sim, filename=None, subs=None, subhalos=False, _inherit_data_from=None):
        super().__init__(sim, filename=filename, subs=subs, subhalos=subhalos, _inherit_data_from=_inherit_data_from)
        self.physical_units()

    @classmethod
    def _get_catalogue_multifile(cls, _, user_provided_filename):
        if user_provided_filename is None:
            raise AssertionError("AurigaSubfind neeeds filename provided!")
        return AurigaSubFindHdfMultiFileManager(user_provided_filename)

    def _get_halo(self, halo_number) -> Halo:
        halo_index = self.number_mapper.number_to_index(halo_number)
        return AurigaHalo(
            halo_number,
            self._get_properties_one_halo_using_cache_if_available(halo_number, halo_index),
            self,
            self.base,
            self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index),
        )


class AurigaHalo(Halo, AurigaStarsWind):
    def __init__(self, halo_number, properties, halo_catalogue, *args, **kwa):
        Halo.__init__(self, halo_number, properties, halo_catalogue, *args, **kwa)
        AurigaStarsWind.__init__(self)
        # self.physical_units()


auriga_eps = {4: 369 * pynbody.units.pc, 3: 184 * pynbody.units.pc}


class AurigaLikeHDFSnap(GadgetHDFSnap, AurigaStarsWind, ZoomSnap):
    """Reads AurigaHDF"""

    _readable_hdf5_test_key = "PartType1/SubGroupNumber"

    def __init__(self, particle_filename, halo_filename, analysis_folder, level=4, orientate=True, use_cache=False):
        GadgetHDFSnap.__init__(self, particle_filename)
        self.halo_file = halo_filename
        self.properties["eps"] = auriga_eps.get(level)
        AurigaStarsWind.__init__(self)
        # self.physical_units()
        ZoomSnap.__init__(self, orientate, use_cache=use_cache, analysis_folder=analysis_folder)

    def halos(self, subhalos=False) -> AurigaSubfindHDFCatalogue:
        """Load the Auriga FOF halos.
        Access halo with halo[0]
        Access Subhalo with halo[0].sub[0]
        Or, if subs=True, halo[0] is the first subhalo"""
        return AurigaSubfindHDFCatalogue(self, filename=self.halo_file, subhalos=subhalos)

    def derivable_keys(self) -> list:
        """Returns a list of arrays which can be lazy-evaluated."""
        keys = GadgetHDFSnap.derivable_keys(self)
        res = [key for key in keys if key not in auriga_bad_keys]
        return res


auriga_bad_keys = [
    "V_lum_den",
    "dm",
    "OI",
    "i_mag",
    "r_mag",
    "rho_ne",
    "nxh",
    "NV",
    "R_lum_den",
    "HeIII",
    "ne",
    "R_mag",
    "OVI",
    "mgxh",
    "j_mag",
    "caxh",
    "U_mag",
    "hydrogen",
    "J_lum_den",
    "c_n_sq",
    "nefe",
    "MGII",
    "OII",
    "zeldovich_offset",
    "sife",
    "H_mag",
    "feh",
    "redshift",
    "CIV",
    "k_mag",
    "ofe",
    "I_mag",
    "K_lum_den",
    "aform",
    "r_lum_den",
    "HeII",
    "sixh",
    "em",
    "HeI",
    "j_lum_den",
    "J_mag",
    "U_lum_den",
    "cxh",
    "i_lum_den",
    "u_mag",
    "B_lum_den",
    "v_lum_den",
    "p",
    "cosmodm",
    "h_lum_den",
    "SIV",
    "B_mag",
    "HID12",
    "I_lum_den",
    "b_lum_den",
    "u_lum_den",
    "hetot",
    "HIeos",
    "HII",
    "sxh",
    "V_mag",
    "v_mag",
    "k_lum_den",
    "h_mag",
    "b_mag",
    "H_lum_den",
    "HI",
    "nexh",
    "mgfe",
    "doppler_redshift",
    "oxh",
    "halpha",
    "K_mag",
]
from . import auriga_attributes
