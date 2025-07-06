from typing import Literal
from ..snapshot.simsnap import SimSnap

from functools import cached_property
from .orientation import load_orientation_name, load_orientation_sub_id
import os
import numpy as np


class ZoomSnap:
    def __init__(
        self,
        analysis_folder: str | None = None,
        orientate_center: None | int | dict = 0,
        pot_symmetry: Literal["axi", "spherical"] = "axi",
    ) -> None:
        """orientate_center: Where to center. Nothing, subhalo int, or own orientation,"""
        if not hasattr(self, "hierarchy"):
            self.hierarchy = True

        self.analysis_folder = analysis_folder
        self._check_analysis_folder(analysis_folder)
        self.orientate_center = orientate_center
        self._init_orientation()
        self.pot_symm = pot_symmetry
        # Some unit bugs if config option is used? Unclear...  :(
        self.physical_units()
        self._subfunc_inherit = ["sample", "sort_by"]

    @cached_property
    def potential(self):  # -> agama.Potential:
        from .agama_potential import agama_pynbody_load

        return agama_pynbody_load(self)

    def _init_orientation(self) -> None:
        if self.orientate_center is None:
            self.orientation = None
            self.orientation_name = None
            return
        elif isinstance(self.orientate_center, dict):
            self.orientation = self.orientate_center
            self.orientation_name = self.orientate_center.get("name", "own_orient")
        elif isinstance(self.orientate_center, int):
            self.orientation_name = f"subhalo_{self.orientate_center}"
            self.orientation = load_orientation_sub_id(self, sub_id=self.orientate_center)
        elif isinstance(self.orientate_center, str):
            self.orientation_name = f"{self.orientate_center}"
            self.orientation = load_orientation_name(self)
            self.orientation["name"] = self.orientation_name
        else:
            raise TypeError(f"Orientation Center not recognised {type(self.orientate_center)}")
        # self.description+=f":{self.orientation_name}"

        if "x_cen" in self.orientation:
            self.translate(-self.orientation["x_cen"])
        if "v_cen" in self.orientation:
            self.offset_velocity(-self.orientation["v_cen"])
        if "z_Rot" in self.orientation:
            self.rotate(self.orientation["z_Rot"])

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
        print(f"Creating analysis_folder: {analysis_folder}")
        os.makedirs(analysis_folder)

    def sample(
        self, fraction: float | None = None, n_select: int | None = None, seed: None | int = 42, subsnap: None = None
    ):
        """
        Randomly downsample current.

        Parameters:
            fraction (float): Reduce to fraction of current.
            n_select (int): Select a specific number.
            seed (int): Seed for the local random number generator.
        Returns:
        randomly sampled
        """
        sim = self if subsnap is None else subsnap
        size = len(sim)
        if n_select is not None:
            pass
        elif n_select is None and fraction is not None:
            n_select = int(round(fraction * size))
        else:
            raise AssertionError("Must select a Fraction or number to select")

        assert n_select < size, "Selected number must be less than current size"

        # Create a local RNG with the provided seed so that the global state is unaffected
        filter_array = np.zeros(size, dtype=bool)
        if seed is None:
            true_indices = np.random.choice(size, size=n_select, replace=False)
        else:
            true_indices = np.random.default_rng(seed).choice(size, size=n_select, replace=False)
        filter_array[true_indices] = True

        return sim[filter_array]

    def sort_by(self, sort_key, subsnap: None = None):
        """
        sort selection by key
        """
        sim = self if subsnap is None else subsnap
        indexes = np.argsort(sim[sort_key].v)

        return sim[indexes]


# Needed to load zoom attributes
from . import zoom_attributes
