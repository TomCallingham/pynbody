import agama
from .orientation import lazy_orientate_snap
from .agama_potential import agama_pynbody_load
from .property_cache import del_cached_props, load_cached_props
from pynbody.snapshot import SimSnap


class ZoomSnap:
    def __init__(self, orientate=True, use_cache=False, analysis_folder=None) -> None:
        self.analysis_folder = analysis_folder
        self.use_cache = use_cache if analysis_folder is not None else False
        if orientate:
            lazy_orientate_snap(self)
        self._pot = None
        self._cache = None
        # As we ALWAYS want physical units? Can this be turned on in config?
        # Nothing but bugs :(
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

    def del_cached_keys(self, fam_str, del_keys) -> None:
        del_cached_props(self, fam_str, del_keys)


    @classmethod
    def derived_array(cls, fn):
        if cls not in SimSnap._derived_array_registry:
            SimSnap._derived_array_registry[cls] = {}
        SimSnap._derived_array_registry[cls][fn.__name__] = fn
        fn.__stable__ = False
        return fn

    # TODO HOST
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
