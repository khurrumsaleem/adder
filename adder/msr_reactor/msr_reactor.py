import multiprocessing as mp
import traceback
import weakref

from adder.reactor import Reactor, parallel_worker, parallel_init
from adder.cram import CRAMDepletion
from adder.constants import SUPPLY, BASE_LIB, CHUNKSIZE_FACTOR
from adder.isotope import ISO_REGISTRY


class MSRReactor(Reactor):
    """Class containing the description of a flowing-fluid reactor

    Parameters
    ----------
    name : str
        The case name
    neutronics_solver : str
        The neutronics solver to use
    depletion_solver : str
        The depletion solver intended to be used
    mpi_command : str
        The command to run MPI
    neutronics_exec : str
        The command to run the neutronics solver
        commands
    depletion_exec : str
        The command to run the depletion solver
        commands
    base_neutronics_input_filename : str
        Path and filename of the initial neutronics solver input file
    h5_filename : str
        Path and filename of the HDF5 file containing results
    num_neut_threads : int
        Number of shared memory threads to utilize for neutronics
    num_depl_threads : int
        Number of shared memory threads to utilize for depletion
    num_mpi_procs : int
        Number of distributed memory instances to utilize
    depletion_chunksize : int
        Number of depleting materials to load at a time per thread
    use_depletion_library_xs : bool
        Whether or not to use the depletion library's cross sections
    neutronics_reactivity_threshold : float
        The reactivity threshold to use when excluding isotopes from
        the neutronics model
    neutronics_reactivity_threshold_initial : bool
        Whether or not to apply the neutronics_reactivity_threshold to the
        initial inventory.

    Attributes
    ----------
    neutronics : adder.Neutronics
        The neutronics solver interface
    depletion : adder.Depletion
        The depletion solver interface
    material : Iterable of adder.Material
        The materials in the reactor model
    storage : OrderedDict of adder.Material
        A dictionary containing the adder.Material objects that are not
        currently in the model but instead are located in storage. The
        dictionary entries are accessed via the name attribute of
        adder.Material.
    h5_filename : str
        Path and filename of the HDF5 file containing results.
    h5_file : h5py.File
        The HDF5 file containing results.
    begin_time : datetime.datetime
        The date and time of the initialization of this object
    end_time : datetime.datetime or None
        The date and time at the time of closing the HDF5 file (i.e.,
        the end of execution), or None, if not yet done.
    case_label : str
        The label of the current case, from the user input
    operation_label : str
        The label of the current operation being executed
    case_idx : int
        The index of the current case block
    operation_idx : int
        The index of the current operation being executed within the case block
    step_idx : int
        The index of the step within the operation block
    time : float
        The time of the current state point
    power : float
        The operating power for the current state point
    flux_level : float
        The operating flux for the current state point, integrated over
        all depleting regions and energy groups
    Q_recoverable : float
        The recoverable energy per fission, in units of MeV/fission, for
        the current state point
    keff : float
        k-eigenvalue for the current state point
    keff_stddev : float
        The standard deviation of the calculated k-eigenvalue for the
        current state point; value is 0.0 if a deterministic solver is
        used.
    control_groups : dict
        The set of control group information from the input file, including
        the current perturbation status. The key to the dict is the name and
        the value is an instance of a ControlGroup object.
    fast_forward : bool
        Whether this is a fast forward run (True) or not (False).
    fixed_fuel_depletion : adder.Depletion
        The depletion solver interface for the fixed fuel

    """

    def __init__(self, name, neutronics_solver, depletion_solver,
                 mpi_command, neutronics_exec, depletion_exec,
                 base_neutronics_input_filename, h5_filename, num_neut_threads,
                 num_depl_threads, num_mpi_procs, depletion_chunksize,
                 use_depletion_library_xs, neutronics_reactivity_threshold,
                 neutronics_reactivity_threshold_initial):
        super().__init__(name, neutronics_solver, depletion_solver,
                         mpi_command, neutronics_exec, depletion_exec,
                         base_neutronics_input_filename, h5_filename,
                         num_neut_threads, num_depl_threads, num_mpi_procs,
                         depletion_chunksize, use_depletion_library_xs,
                         neutronics_reactivity_threshold,
                         neutronics_reactivity_threshold_initial)

        if depletion_solver.startswith("msr"):
            # This is either just "msr", and thus the cram order is 16,
            # or the order is specified
            if depletion_solver == "msr":
                order = 16
            else:
                # Strip off the order after "cram" and convert to integer
                order = int(depletion_solver.partition("msr")[2])
            self.fixed_fuel_depletion = \
                CRAMDepletion(depletion_exec, num_depl_threads, num_mpi_procs,
                              depletion_chunksize, order)
            self.update_logs(self.fixed_fuel_depletion.logs)
            self.fixed_fuel_depletion.clear_logs()
        else:
            msg = "If a MSR reactor is defined, the MSR depletion solver " + \
                "must be used!"
            self.log("error", msg)

    def init_materials_and_input(self, neutronics_lib_file, depletion_lib_file,
                                 depletion_lib_name, user_mats_info,
                                 user_univ_info, shuffled_mats, shuffled_univs,
                                 flux_smoothing_method, solve_method,
                                 system_data):
        """Reads the base neutronics input file, stores the parsed
        relevant information in :param:`neutronics_inputs` and creates
        the material information. The material information is stored
        in the :param:`materials` parameter. This method also
        initializes the set of nuclides that are possible in the
        neutronics solve and loads the depletion library

        Parameters
        ----------
        neutronics_liby_file : str
            Path to the neutronics solver's nuclear data library files.
            For MCNP this would be the xsdir file.
        depletion_lib_file : str
            The path to the depletion solver's library HDF5 file
        depletion_lib_name : str
            The name of the specific library to use within
            :param:`depletion_lib_file`
        user_mats_info : OrderedDict
            The keys are the ids in the neutronics solver and
            the value is an OrderedDict of the name, depleting boolean,
            volume, density, non_depleting_isotopes list, 
            apply_reactivity_threshold_to_initial_inventory flag, and 
            use_default_depletion_library flag.
        user_univ_info : OrderedDict
            The keys are the universe ids in the neutronics solver and
            the value is an OrderedDict of the name.
        shuffled_mats : set
            The set of material names that are shuffled
        shuffled_univs : set
            The set of universe names that are shuffled
        flux_smoothing_method : {"histogram", "average"}
            The flux smoothing methodology to apply when the fuel fluid
            flows from one region to another in the core.
        solve_method : {"brute", "tmatrix", "tmatrix_expm", "rxn_rate_avg"}
            The methodology to use when performing the depletion
        system_data : List of OrderedDict
            The data for each system
        """

        super().init_materials_and_input(neutronics_lib_file,
                                         depletion_lib_file,
                                         depletion_lib_name, user_mats_info,
                                         user_univ_info, shuffled_mats,
                                         shuffled_univs)

        self.depletion_libs[BASE_LIB].set_atomic_mass_vector()

        # Initialize the MSR systems
        self.depletion.set_msr_params(flux_smoothing_method, solve_method,
                                      system_data, self.materials,
                                      self.depletion_libs)

        # Find the materials that are stationary
        # We will need to update this every time as the stationary
        # components can be shuffled
        fluid_mats = self.depletion.fluid_mats
        for mat in self.materials:
            if mat.name not in fluid_mats:
                if not self.neutronics.use_depletion_library_xs:
                    if not mat.is_default_depletion_library:
                        # Then we need a new library
                        new_lib = self.depletion_libs[BASE_LIB].clone(
                            new_name=mat.name)
                        self.depletion_libs[new_lib.name] = new_lib
                        mat.depl_lib_name = new_lib.name

        # Get the new isotopes
        new_isos = {}
        for system in self.depletion.systems:
            sys_lib = system.library
            sys_isos = [
                iso_name for iso_name, index in
                    sys_lib.isotope_indices.items()
                    if index >= system.num_original_isotopes]
            for iso in sys_isos:
                for mat in fluid_mats:
                    xs_lib = mat.default_xs_library
                    if iso in new_isos:
                        new_isos[iso].add(xs_lib)
                    else:
                        new_isos[iso] = set([xs_lib])
        # Now we can add it
        for iso_name, iso_xs_libs in new_isos.items():
            for xs_lib in iso_xs_libs:
                for is_depleting in (True, False):
                    ISO_REGISTRY.register_isotope(iso_name, xs_lib,
                    is_depleting)

    def _parallel_depletion_manager(self, dt, depletion_step, num_substeps):
        """Executes the depletion in parallel"""

        if dt <= 0.:
            return

        # Determine the number of depleting materials
        num_depl_mats = len(list((i for i, mat in enumerate(self._materials) if
                     (mat.name not in self.depletion.fluid_mats and
                      mat.is_depleting and mat.status != SUPPLY))))

        # Deplete the flowing fluid materials
        self.log("info", "Depleting Fluid System Materials", 8)
        self.depletion.execute(self.materials, self.depletion_libs, dt,
            depletion_step, num_substeps, self.neutronics.reactivity_threshold)

        # And now deplete the remaining materials
        # Update the materials that are not fluids
        mat_idxs = (i for i, mat in enumerate(self._materials) if
                     (mat.name not in self.depletion.fluid_mats and
                      mat.is_depleting and mat.status != SUPPLY))

        msg = "Depleting Remaining Materials with {} Thread".format(
            self.fixed_fuel_depletion.num_threads)
        if self.fixed_fuel_depletion.num_threads > 1:
            # Make it plural
            msg += "s"
        self.log("info", msg, 8)

        self.fixed_fuel_depletion.compute_decay(self.depletion_libs[BASE_LIB])

        # Perform the depletion
        self.log("info", "Executing Depletion", 10)
        weak_deplete = weakref.proxy(self.fixed_fuel_depletion)
        if self.depletion.num_threads == 1:
            for i_mat in mat_idxs:
                mat = self._materials[i_mat]
                lib = self.depletion_libs[mat.depl_lib_name]
                matrix = weak_deplete.compute_library(lib, mat._flux)

                try:
                    new_isos, new_fracs, new_density = \
                        weak_deplete.execute(mat, matrix,
                                             lib.isotope_indices,
                                             lib.inverse_isotope_indices, dt,
                                             depletion_step, num_substeps,
                                             True)
                except Exception:
                    error_msg = traceback.format_exc()
                    print(error_msg)
                self._materials[i_mat].apply_new_composition(
                    new_isos, new_fracs, new_density)
        else:

            # Calculate chunksize
            if weak_deplete.chunksize == 0:
                chunksize, extra = divmod(num_depl_mats,
                                    weak_deplete.num_threads * CHUNKSIZE_FACTOR)
                if extra:
                    chunksize += 1

                # Check to make sure chunksize is an integer greater than zero
                chunksize = int(chunksize)
                if chunksize < 1:
                    chunksize = 1

            else:
                chunksize = weak_deplete.chunksize

            # Do the parallel processing
            errors = []

            with mp.Pool(
                    processes=weak_deplete.num_threads,
                    initializer=parallel_init,
                    initargs=(weak_deplete, dt, depletion_step, num_substeps)) as pool:
                for i_mat, new_isos, new_fracs, new_density \
                    in pool.imap_unordered(parallel_worker, mat_idxs,
                                           chunksize=chunksize):
                    if isinstance(new_isos, str):
                        errors.append((i_mat, new_isos))
                    else:
                        self._materials[i_mat].apply_new_composition(
                            new_isos, new_fracs, new_density)

            # Now we handle the errors. They *may* be due to parallel system
            # issues (file I/O, etc), so lets just run the few cases in serial
            if len(errors) > 0:
                # Now raise the error messages
                msg = "The Following Materials Encountered Errors While " + \
                    "Processed in Parallel:"
                self.log("info_file", msg)
                for i_mat, error in errors:
                    self.log("info_file", error)
                self.log("info", "Rerunning Failed Depletions", 8)
                # And re-run in serial
                for i_mat, _ in errors:
                    mat = self._materials[i_mat]
                    lib = self.depletion_libs[mat.depl_lib_name]
                    mtx = self.depletion.compute_library(lib, mat.flux),
                    idxs = lib.isotope_indices
                    inv_idxs = lib.inverse_isotope_indices
                    try:
                        new_isos, new_fracs, new_density = \
                            self.depletion.execute(
                                mat, mtx, idxs, inv_idxs,
                                dt, depletion_step, num_substeps, True)
                    except Exception:
                        error_msg = traceback.format_exc()
                        self.log("error", error_msg)
                    mat.apply_new_composition(new_isos, new_fracs, new_density)
