from ..halo.subfindhdf import ArepoSubfindHDFCatalogue
from ..snapshot.gadgethdf import GadgetHDFSnap, _GadgetHdfMultiFileManager
from ..halo import Halo, HierarchicalHalo
from ..snapshot import IndexedSubSnap
from ..snapshot.subsnap import HierarchyIndexedSubSnap
from .. import units
import numpy as np
from typing import Literal

from .zoom import ZoomSnap

import h5py

from .. import family


class AurigaStarsWind:
    def __init__(self) -> None:
        self._special_gettr_keys = ["stars", "star", "wind", "winds"]

    def _special_getattr__(self, name, base):
        """This function overrides the behaviour of f.X where f is a SimSnap object.

        It serves two purposes; first, it provides the family-handling behaviour
        which makes f.dm equivalent to f[pynbody.family.dm]. Second, it implements
        persistent objects -- properties which are shared between two equivalent SubSnaps."""
        if name in ["stars", "star"]:
            sim = base[family.get_family("star")]
            return sim[sim["aform"] > 0]
        elif name in ["wind", "winds"]:
            sim = base[family.get_family("star")]
            return sim[sim["aform"] < 0]
        print("Name not in special get=", name)
        return


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
        self._numfiles -= len(to_del)  # type: ignore


class AurigaSubfindHDFCatalogue(ArepoSubfindHDFCatalogue):
    def __init__(self, sim, filename=None, subs=None, subhalos=False, _inherit_data_from=None):
        super().__init__(sim, filename=filename, subs=subs, subhalos=subhalos, _inherit_data_from=_inherit_data_from)
        self.physical_units()

    @classmethod
    def _get_catalogue_multifile(cls, sim, user_provided_filename) -> AurigaSubFindHdfMultiFileManager:
        if user_provided_filename is None:
            raise AssertionError("AurigaSubfind neeeds filename provided!")
        return AurigaSubFindHdfMultiFileManager(user_provided_filename)

    def _get_halo(self, halo_number) -> Halo | HierarchicalHalo:
        halo_index = self.number_mapper.number_to_index(halo_number)
        if hasattr(self._base(), "hierarchy") and self._base().hierarchy:
            return AurigaHierarchicalHalo(
                halo_number,
                self._get_properties_one_halo_using_cache_if_available(halo_number, halo_index),
                self,
                self.base,
                self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index),
            )
        return AurigaHalo(
            halo_number,
            self._get_properties_one_halo_using_cache_if_available(halo_number, halo_index),
            self,
            self.base,
            self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index),
        )


class AurigaHalo(Halo, AurigaStarsWind):
    def __init__(self, halo_number, properties, halo_catalogue, *args, **kwa):
        AurigaStarsWind.__init__(self)
        Halo.__init__(self, halo_number, properties, halo_catalogue, *args, **kwa)


class AurigaHierarchicalHalo(HierarchicalHalo, AurigaStarsWind):
    def __init__(self, halo_number, properties, halo_catalogue, *args, **kwa):
        AurigaStarsWind.__init__(self)
        HierarchicalHalo.__init__(self, halo_number, properties, halo_catalogue, *args, **kwa)


auriga_eps = {4: 369 * units.pc, 3: 184 * units.pc}


class AurigaLikeHDFSnap(
    ZoomSnap,
    GadgetHDFSnap,
    AurigaStarsWind,
):  # , AurigaStarsWind):
    """Reads AurigaHDF"""

    _readable_hdf5_test_key = "PartType1/SubGroupNumber"
    _namemapper_config_section = "auriga-name-mapping"

    def __init__(
        self,
        particle_filename: str,
        halo_filename: str,
        analysis_folder: str | None = None,
        orientate_center: None | int | dict = 0,
        pot_symmetry: Literal["axi", "spherical"] = "axi",
        level=4,
    ):
        # self._set_default_gadget_units()
        self._file_units_system_default = True
        GadgetHDFSnap.__init__(self, particle_filename)
        self.halo_file = halo_filename
        self.properties["eps"] = auriga_eps.get(level)
        AurigaStarsWind.__init__(self)

        self.physical_units()

        print("flipping zyx!")
        self.zyx_order()  # Needs to be before orientation goes!
        ZoomSnap.__init__(self, analysis_folder, orientate_center, pot_symmetry)

        self.forcefloat64 = True
        self._mass_dtype = np.float64

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
