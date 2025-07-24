from dataclasses import dataclass, field
from ctypes import c_ushort

from adder.type_checker import *
from adder.data import atomic_mass, zam

@dataclass
class Isotope:
    """This class contains relevant information about an isotope.

    Parameters
    ----------
    name : str
        The isotope name in GND format, e.g., "U235"
    xs_library : str
        Cross section library reference, i.e. "80c" if using MCNP.
    is_depleting : bool, optional
        Whether or not the isotope should be treated as depleting;
        defaults to True.

    Attributes
    ----------
    name : str
        The isotope name in GND format, e.g., "U235"
    Z : int
        Proton number; for example U-235's Z is 92.
    A : int
        Mass number; for example U-235's A is 235.
    M : int
        Metastable state; for example the ground state is 0.
    xs_library : str
        cross section library reference, i.e. "80c" if using MCNP.
    is_depleting : bool
        Whether or not the isotope should be treated as depleting

    """
    name: str
    xs_library: str
    is_depleting: bool = True
    _Z: c_ushort = field(init=False)
    _A: c_ushort = field(init=False)
    _M: c_ushort = field(init=False)

    def __post_init__(self):
        # Do the relevant checks of values
        check_type("name", self.name, str)
        check_type("xs_library", self.xs_library, str)
        check_type("is_depleting", self.is_depleting, bool)
        # TODO: These may be not necessary given reasonableness of names, but
        # are left as this should have no major runtime impact and should
        # provide an easy point to add in pseudo-nuclide control

        Z, A, M = zam(self.name)
        check_type("Z", Z, int)
        check_greater_than("Z", Z, 0, equality=False)
        check_less_than("Z", Z, 118, equality=True)
        check_type("A", A, int)
        check_greater_than("A", A, 0, equality=True)
        check_less_than("A", A, 300, equality=True)
        check_less_than("M", M, 10, equality=True)
        if A == 0 and M != 0:
            raise ValueError("Cannot provide M for elemental data!")

        self._Z, self._A, self._M = c_ushort(Z), c_ushort(A), c_ushort(M)

    @property
    def Z(self):
        return self._Z.value

    @property
    def A(self):
        return self._A.value

    @property
    def M(self):
        return self._M.value

    @property
    def atomic_mass(self):
        return atomic_mass(self.name)

    def __repr__(self):
        msg = "<Isotope {}, xs_lib: {}".format(self.name, self.xs_library) + \
            ", is_depleting: {}>".format(self.is_depleting)
        return msg

    def __hash__(self):
        # Control the hash so it is cheaper to create (its executed alot by the
        # isotope registry), and to guarantee reproducibility even across
        # threads
        return hash((self.name, self.xs_library, self.is_depleting))


class IsotopeRegistry:
    """This class is a pre-allocated registry of all isotopes that may be
    present in the model.

    This registry is necessary for multiple reasons:

    1. Materials must know which isotopes they contain, however, storing
        Isotope instances for each Material will be very costly for even
        moderately-sized models.  Therefore, we instead keep all the isotopes in
        one place (this registry) and instead the Materials need only to keep
        track of the index of the isotope in this registry.

    2. This class provides a simple location of isotopes that can be placed in
        shared memory by downstream parallelization techniques so that every
        material will have access to the same isotope registry from the same
        location in memory. This limits the amount of data duplication necessary
        to initialize parallel execution blocks.

    This registry will be pre-allocated with all possible isotopes, including
    the combinations present in the neutronics library and perturbations of
    is-depleting or not. This will use additional space, however, it will only
    be several thousand isotopes and therefore will be significantly more
    compact than storing on the Materials themselves.

    Parameters
    ----------
    neutronics_library_isos : dict
        The keys are the isotope names in GND format and the values are an
        Iterable of associated library names available in the neutronics
        library.
    """

    def __init__(self, neutronics_library_isos):
        iso_names = []
        iso_xs_libs = []
        # First we set up the set of isotopes based on the neutronics lib
        for iso_name, set_of_iso_libs in neutronics_library_isos.items():
            # This is really just pulling values from the input directly as no
            # further processing is needed
            iso_names.append(iso_name)
            iso_xs_libs.append(set_of_iso_libs)

        # For robustness, make sure the iso names and xs_libs are the same size
        if len(iso_names) != len(iso_xs_libs):
            msg = 'iso_names and iso_xs_libs must be same length!'
            raise ValueError(msg)

        # Now we create depleting and non-depleting versions of each isotope
        self._isos = []
        self._iso_names = []
        self._idxs_from_name = {}
        for i in range(len(iso_names)):
            iso_name = iso_names[i]
            for xs_lib in iso_xs_libs[i]:
                for is_depleting in (True, False):
                    self.register_isotope(iso_name, xs_lib, is_depleting)

    def clear(self):
        """Clears the registry for unit tests"""

        self._isos = []
        self._iso_names = []
        self._idxs_from_name = {}

    def register_depletion_lib_isos(self, depletion_libs, materials):
        """Registers isotopes that exist only in the depletion library and not
        the neutronics library.

        This must be done separate from the neutronics isotopes (in __init__)
        because we need the neutronics isotopes to parse the neutronics file,
        and we need to parse the neutronics file before we can get materials
        which we need here.

        Parameters
        ----------
        depletion_libs : OrderedDict of DepletionLibrary
            The depletion libraries used by the materials
        materials: List of adder.Material
            The List of Materials for which we will need the default xs lib
            from.

        """

        iso_names_and_libs = {}

        # Get all the isotopes in a depleting materials inventory that
        # need to register/check if registered
        for mat in materials:
            if mat.is_depleting:
                dflt_xs_lib = mat.default_xs_library
                for iso in depletion_libs[mat.depl_lib_name].isotopes.keys():
                    if iso in iso_names_and_libs:
                        iso_names_and_libs[iso].add(dflt_xs_lib)
                    else:
                        iso_names_and_libs[iso] = set([dflt_xs_lib])

        # Now we can add it
        for iso_name, iso_xs_libs in iso_names_and_libs.items():
            for xs_lib in iso_xs_libs:
                self.register_isotope(iso_name, xs_lib, is_depleting=True)

    @property
    def num_isos(self):
        return len(self._isos)

    def register_isotope(self, iso_name, xs_lib, is_depleting=True):
        """Creates a new isotope if not present. Returns the index of the
        isotope, whether newly created or not.

        Parameters
        ----------
        iso_name : str
            The isotope name in GND format, e.g., "U235"
        xs_lib : str
            Cross section library reference, e.g., "80c" if using MCNP.
        is_depleting : bool
            Whether or not the isotope should be treated as depleting

        Returns
        -------
        idx : int
            The new isotope index
        """

        # First check to see if this isotope exists
        if iso_name in self._idxs_from_name:
            # The isotope does exist but let's see if we have a xs_lib
            # and depleting match
            idxs = self._idxs_from_name[iso_name]
            for idx in idxs:
                iso = self._isos[idx]
                if iso.name != iso_name:
                    # Add an error if this is the case bc then _idxs_from_name
                    # is out of sync with _isos and we need a dev to figure
                    # out why.
                    msg = "IsotopeRegistry._idxs_from_name may be corrupted!"
                    raise ValueError(msg)
                if iso.xs_library == xs_lib and iso.is_depleting == is_depleting:
                    # Then we have a match, just return our idx
                    return idx
                # Otherwise, try the rest in idxs and if, when we do all of
                # them, we find still make it this far, then it's time for a
                # new isotope

        # If we get here then we need a new isotope. Make it
        idx = self.num_isos
        iso = Isotope(iso_name, xs_lib, is_depleting)
        self._isos.append(iso)
        self._iso_names.append(iso_name)
        if iso_name in self._idxs_from_name:
            self._idxs_from_name[iso_name].append(idx)
        else:
            self._idxs_from_name[iso_name] = [idx]

        return idx

    def switch_iso_depleting_status(self, old_idx, new_is_depleting):
        """Find the idx of the same isotope with a different depleting status.

        Used when assigning isotopes a new status.

        Parameters
        ----------
        old_idx : int
            The index of the isotope to change
        new_is_depleting : bool
            Whether or not the isotope should be treated as depleting

        Returns
        -------
        new_idx : int
            The index of the isotope with the new status
        """

        if old_idx >= self.num_isos:
            msg = "Original Isotope not found!"
            raise ValueError(msg)

        old_iso = self._isos[old_idx]

        # We can short-circuit the register isotope process by checking to see
        # if this isotope already has the new is_depleting status.
        if new_is_depleting == old_iso.is_depleting:
            return old_idx

        # Otherwise, we need a new index so lets just go get it
        new_idx = self.register_isotope(old_iso.name, old_iso.xs_library,
            new_is_depleting)
        return new_idx

    def get_isotope(self, iso_name, xs_lib, is_depleting):
        """Gets the isotope index from the registry.

        This is somewhat similar to the register_isotope method but we know it
        is read-only because we only access results, therefore making it more
        applicable in parallel sections.

        Parameters
        ----------
        iso_name : str
            The isotope name in GND format, e.g., "U235"
        xs_lib : str
            Cross section library reference, i.e. "80c" if using MCNP.
        is_depleting : bool
            Whether or not the isotope should be treated as depleting

        Returns
        -------
        idx : int
            The isotope index
        """

        # First check to see if this isotope exists
        if iso_name in self._idxs_from_name:
            # The isotope does but lets see if we have a xs_lib
            # and depleting match
            idxs = self._idxs_from_name[iso_name]
            for idx in idxs:
                iso = self._isos[idx]
                if iso.name != iso_name:
                    # Add an error if this is the case bc then _idxs_from_name
                    # is out of sync with _isos and we need a dev to figure
                    # out why.
                    msg = "IsotopeRegistry._idxs_from_name may be corrupted!"
                    raise ValueError(msg)
                if iso.xs_library == xs_lib and iso.is_depleting == is_depleting:
                    # Then we have a match, just return our idx
                    return idx

        # Just in case, to help with debugging (should never get here!)
        raise ValueError(f'Invalid Isotope: {iso_name}, {xs_lib}, {is_depleting}')

ISO_REGISTRY = IsotopeRegistry({})
