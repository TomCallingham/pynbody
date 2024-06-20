# class AurigaSubFindHDFHaloCatalogue(SubFindHDFHaloCatalogue, AurigaStarsWind):
#     """
#     Gadget's SubFind HDF Subhalo catalogue.

#     Initialized with the parent FOF group catalogue and created
#     automatically when an fof group is created
#     """

#     # def __init__(self, group_id, group_catalogue):
#     def __init__(self, sim, filename=None, subs=None, subhalos=False, _inherit_data_from=None):
#         SubFindHDFHaloCatalogue.__init__(sim, filename, subs, subhalos, _inherit_data_from)
#         AurigaStarsWind.__init__(self)
#         # self.physical_units()

#     def _get_halo(self, i):
#         if self.base is None:
#             raise RuntimeError("Parent SimSnap has been deleted")

#         if i > len(self) - 1:
#             raise ValueError("FOF group %d does not have subhalo %d" % (self._group_id, i))

#         # need this to index the global offset and length arrays
#         absolute_id = self._group_catalogue._fof_group_first_subhalo[self._group_id] + i

#         # now form the particle IDs needed for this subhalo
#         type_map = self.base._family_to_group_map

#         halo_lengths = self._group_catalogue._subfind_halo_lengths
#         halo_offsets = self._group_catalogue._subfind_halo_offsets

#         # create the particle lists
#         tot_len = 0
#         for g_ptypes in list(type_map.values()):
#             for g_ptype in g_ptypes:
#                 tot_len += halo_lengths[g_ptype][absolute_id]

#         plist = np.zeros(tot_len, dtype="int64")

#         npart = 0
#         for ptype in self.base._families_ordered():
#             # family slice in the SubFindHDFSnap
#             sl = self.base._family_slice[ptype]

#             for g_ptype in type_map[ptype]:
#                 # add the particle indices to the particle list
#                 offset = halo_offsets[g_ptype][absolute_id]
#                 length = halo_lengths[g_ptype][absolute_id]
#                 ind = np.arange(sl.start + offset, sl.start + offset + length)
#                 plist[npart : npart + length] = ind
#                 npart += length

#         return AurigaSubFindHDFSubHalo(i, self._group_id, self._group_catalogue, self, self.base, plist)

# def _get_halo(self, halo_number) -> Halo:
#     halo_index = self.number_mapper.number_to_index(halo_number)
#     return Halo(
#         halo_number,
#         self._get_properties_one_halo_using_cache_if_available(halo_number, halo_index),
#         self,
#         self.base,
#         self._get_particle_indices_one_halo_using_list_if_available(halo_number, halo_index),
#     )


# class AurigaSubFindFOFGroup(Halo, AurigaStarsWind):
#     """
#     SubFind FOF group class, modified for Auriga
#     """

#     def __init__(self, group_id, *args):
#         """Construct a special halo representing subfind's FOF group"""
#         Halo.__init__(self, group_id, *args)

#         self._subhalo_catalogue = AurigaSubFindHDFHaloCatalogue(group_id, self._halo_catalogue)

#         self._descriptor = "fof_group_" + str(group_id)

#         self.properties.update(self._halo_catalogue.get_halo_properties(group_id, subhalos=False))

#         AurigaStarsWind.__init__(self)
#         # self.physical_units()

#     def __getattr__(self, name):
#         if name == "sub":
#             return self._subhalo_catalogue
#         else:
#             return Halo.__getattr__(self, name)

