# import agama
from ..snapshot import SimSnap

from .orientation import lazy_orientate_snap
from .property_cache import del_cached_props, load_cached_props


class ZoomSnap:
    def __init__(self, orientate=True, use_cache=False, analysis_folder=None) -> None:
        self.analysis_folder = analysis_folder
        self.use_cache = use_cache if analysis_folder is not None else False
        if orientate:
            lazy_orientate_snap(self)
        self._pot = None
        self._cache = None
        # Some unit bugs if config option is used? Unclear...  :(
        self.physical_units()

    @property
    def potential(self): #-> agama.Potential:
        if self._pot is None:
            from .agama_potential import agama_pynbody_load
            self._pot = agama_pynbody_load(self, symm="axi")
        return self._pot

    @property
    def cached_props(self) -> dict:
        if self._cache is None:
            self._cache = load_cached_props(self)
        return self._cache

    def del_cached_keys(self, fam_str, del_keys) -> None:
        del_cached_props(self, fam_str, del_keys)


    @classmethod
    def derived_array(cls, fn):
        if cls not in SimSnap._derived_array_registry:
            SimSnap._derived_array_registry[cls] = {}
        SimSnap._derived_array_registry[cls][fn.__name__] = fn
        fn.__stable__ = False
        return fn


#Needed to load zoom attributes
from . import zoom_attributes
