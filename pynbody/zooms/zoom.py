import agama
from .orientation import lazy_orientate_snap
from .agama_potential import agama_pynbody_load
from .property_cache import del_cached_props, load_cached_props
from pynbody.snapshot import SimSnap


class ZoomSnap:
    def __init__(self, orientate=True) -> None:
        # self.analysis_folder = ""
        if orientate:
            lazy_orientate_snap(self)
        self._pot = None
        self._cache = None
        self.physical_units()

    @property
    def potential(self) -> agama.Potential:
        if self._pot is None:
            self._pot = agama_pynbody_load(self, symm="axi")
        return self._pot

    @property
    def cached_props(self) -> dict:
        if self._cache is None:
            self._cache = load_cached_props(self)
        return self._cache

    def del_cached_keys(self, del_keys) -> None:
        del_cached_props(self, del_keys)

    @classmethod
    def derived_quantity(cls, fn):
        if cls not in SimSnap._derived_quantity_registry:
            SimSnap._derived_quantity_registry[cls] = {}
        SimSnap._derived_quantity_registry[cls][fn.__name__] = fn
        fn.__stable__ = False
        return fn

    # @property
    # def host(self) -> HostData:
    #     if self._host is None:
    #         h0 = self.halos()[0]
    #         h0.physical_units()
    #         print("Currently host on 0")
    #         self._host = HostData(h0)
    #     return self._host
    #
from . import zoom_attributes
