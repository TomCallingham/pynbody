from ..snapshot.simsnap import SimSnap

from functools import cached_property
from .orientation import load_orientation
from .property_cache import del_cached_props, load_cached_props
import os


class ZoomSnap:
    def __init__(
        self,
        orientate: bool = True,
        use_cache: bool = False,
        analysis_folder: str | None = None,
        orientation: None | dict = None,
    ) -> None:
        self.analysis_folder = analysis_folder
        self._check_analysis_folder(analysis_folder)
        self.use_cache = use_cache if analysis_folder is not None else False
        self.orientate = orientate
        self._orientation = orientation
        self._cache = None
        # Some unit bugs if config option is used? Unclear...  :(
        self.physical_units()

        self.hierarchy = False
        self._init_lazy_orientation()

    @cached_property
    def potential(self):  # -> agama.Potential:
        from .agama_potential import agama_pynbody_load

        return agama_pynbody_load(self, symm="axi")

    @property
    def orientation(self) -> dict:
        if self._orientation is None:
            self._orientation = load_orientation(self)
        return self._orientation

    @property
    def cached_props(self) -> dict:
        if self._cache is None:
            self._cache = load_cached_props(self)
        return self._cache

    def del_cached_keys(self, fam_str: str, del_keys) -> None:
        del_cached_props(self, fam_str, del_keys)

    @classmethod
    def derived_array(cls, fn):
        if cls not in SimSnap._derived_array_registry:
            SimSnap._derived_array_registry[cls] = {}
        SimSnap._derived_array_registry[cls][fn.__name__] = fn
        fn.__stable__ = False
        return fn

    def _check_analysis_folder(self, analysis_folder: str | None):
        if analysis_folder is None:
            return
        if os.path.isdir(analysis_folder):
            return
        print("Creating analysis_folder!")
        os.makedirs(analysis_folder)

    def _init_lazy_orientation(self):
        if not self.orientate:
            return
        if "x_cen" in self.orientation:
            self.translate(-self.orientation["x_cen"])
        if "v_cen" in self.orientation:
            self.offset_velocity(-self.orientation["v_cen"])
        if "z_Rot" in self.orientation:
            self.rotate(self.orientation["z_Rot"])


# Needed to load zoom attributes
from . import zoom_attributes
