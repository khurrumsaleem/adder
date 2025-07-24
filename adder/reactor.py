from collections import OrderedDict
import datetime
import os
import traceback
import multiprocessing as mp
import weakref

import h5py
import numpy as np

from adder.loggedclass import LoggedClass
from adder.constants import *
import adder.mcnp as mcnp
import adder.origen22 as origen22
import adder.cram as cram
from adder.depletionlibrary import DepletionLibrary
from adder.msr import MSRDepletion
from adder.material import Material
from adder.mcnp.tally import Tally
from adder.neutronics import Neutronics
from adder.depletion import Depletion
from adder.type_checker import *
from adder.control_group import ControlGroup
from adder.utils import get_transform_args
import adder.isotope


class Reactor(LoggedClass):
    """Class containing the description of the reactor

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
    materials : Iterable of adder.Material
        The materials in the reactor model
    depletion_libs : OrderedDict of DepletionLibrary
        The depletion libraries used by the materials
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
        The operating 1-group flux for the current state point, averaged over
        all depleting regions
    Q_recoverable : float
        The recoverable energy per fission, in units of MeV/fission, for
        the current state point
    keff : float
        k-eigenvalue for the current state point
    keff_stddev : float
        The standard deviation of the calculated k-eigenvalue for the
        current state point; value is 0.0 if a deterministic solver is
        used.
    nu : float
        Average number of neutrons per fission (if power > 0)
    flux_normalization : float
        Flux normalization factor if power set for depletion step
        (otherwise = 1.0)
    power_renormalization : float
        Power re-normalization factor if renormalize_power_density flag is
        set to True (otherwise = 1.0)
    control_groups : dict
        The set of control group information from the input file, including
        the current perturbation status. The key to the dict is the name and
        the value is an instance of a ControlGroup object.
    user_tallies_adder_i : dict
        The dict containing the information read from the ADDER input for user tallies
    user_tally_res : dict
        The dict containing the information read from the mctal output file for user tallies
    fast_forward : bool
        Whether this is a fast forward run (True) or not (False).
    """

    def __init__(self, name, neutronics_solver, depletion_solver,
                 mpi_command, neutronics_exec, depletion_exec,
                 base_neutronics_input_filename, h5_filename, num_neut_threads,
                 num_depl_threads, num_mpi_procs, depletion_chunksize,
                 use_depletion_library_xs, neutronics_reactivity_threshold,
                 neutronics_reactivity_threshold_initial):
        super().__init__(2, __name__)
        self.name = name
        self.neutronics = get_neutronics(
            neutronics_solver, mpi_command, neutronics_exec,
            base_neutronics_input_filename, num_neut_threads, num_mpi_procs,
            use_depletion_library_xs, neutronics_reactivity_threshold,
            neutronics_reactivity_threshold_initial)
        self.depletion = get_depletion(
            depletion_solver, depletion_exec, num_depl_threads, num_mpi_procs,
            depletion_chunksize)
        self.update_logs(self.depletion.logs)
        self.depletion.clear_logs()
        self.depletion_libs = OrderedDict()
        self.h5_filename = h5_filename
        self.h5_file = None
        self.h5_initialized = False

        # Initialize data for the current statepoint
        self.step_idx = 0
        self.case_label = "Initial"
        self.operation_label = "Initial"
        self.case_idx = 0
        self.operation_idx = 0
        self.time = 0.0
        self.power = 0.0
        self.flux_level = 0.0
        self.Q_recoverable = 200.0
        self.keff = 0.0
        self.keff_stddev = 0.0
        self.nu = 0.0
        self.flux_normalization = 1.0
        self.power_renormalization = 1.0
        self.materials = None
        self.control_groups = {}
        self.user_tallies_adder_i = {}
        self.user_tally_res = {}
        self.fast_forward = False

        self.end_time = None
        self.begin_time = datetime.datetime.now()

        # Set up the logger and log that we initialized our reactor
        self.log("info", "Initialized Reactor {}".format(self.name))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def neutronics(self):
        return self._neutronics

    @neutronics.setter
    def neutronics(self, neutronics):
        check_type("neutronics", neutronics, Neutronics)
        self._neutronics = neutronics

    @property
    def depletion(self):
        return self._depletion

    @depletion.setter
    def depletion(self, depletion):
        check_type("depletion", depletion, Depletion)
        self._depletion = depletion

    @property
    def depletion_libs(self):
        return self._depletion_libs

    @depletion_libs.setter
    def depletion_libs(self, libs):
        check_type("depletion_libs", libs, dict)
        self._depletion_libs = libs
        # For our parallel region to be efficient, we do not want to have
        # pickling of our massive data for each parallel region. Instead, if
        # we make the libraries be globally accessible, then the Linux OS will
        # use a copy-on-write technique so that no actual copies are done unless
        # we write to the objects. We don't plan on writing these in the
        # parallel region so that is fine. The items to pass to the library are
        # the material and depletion libs.
        global global_libs
        global_libs = self._depletion_libs


    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, materials):
        if materials is not None:
            check_iterable_type("materials", materials, Material,
                                min_depth=1, max_depth=1)
        self._materials = materials
        # For our parallel region to be efficient, we do not want to have
        # pickling of our massive data for each parallel region. Instead, if
        # we make the libraries be globally accessible, then the Linux OS will
        # use a copy-on-write technique so that no actual copies are done unless
        # we write to the objects. We don't plan on writing these in the
        # parallel region so that is fine. The items to pass to the library are
        # the material and depletion libs.
        global global_mats
        global_mats = self._materials

    @property
    def h5_filename(self):
        return self._h5_filename

    @h5_filename.setter
    def h5_filename(self, h5_filename):
        check_type("h5_filename", h5_filename, str)
        self._h5_filename = h5_filename

    @property
    def case_label(self):
        return self._case_label

    @case_label.setter
    def case_label(self, case_label):
        check_type("case_label", case_label, str)
        self._case_label = case_label

    @property
    def operation_label(self):
        return self._operation_label

    @operation_label.setter
    def operation_label(self, op_label):
        check_type("operation_label", op_label, str)
        self._operation_label = op_label

    @property
    def case_idx(self):
        return self._case_idx

    @case_idx.setter
    def case_idx(self, case_idx):
        check_type("case_idx", case_idx, int)
        self._case_idx = case_idx

    @property
    def operation_idx(self):
        return self._operation_idx

    @operation_idx.setter
    def operation_idx(self, op_idx):
        check_type("operation_idx", op_idx, int)
        self._operation_idx = op_idx

    @property
    def step_idx(self):
        return self._step_idx

    @step_idx.setter
    def step_idx(self, step_idx):
        check_type("step_idx", step_idx, int)
        self._step_idx = step_idx

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        check_type("time", time, float)
        self._time = time

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        check_type("power", power, float)
        self._power = power

    @property
    def flux_level(self):
        return self._flux_level

    @flux_level.setter
    def flux_level(self, flux_level):
        check_type("flux_level", flux_level, float)
        self._flux_level = flux_level

    @property
    def Q_recoverable(self):
        return self._Q_recoverable

    @Q_recoverable.setter
    def Q_recoverable(self, Q_recoverable):
        check_type("Q_recoverable", Q_recoverable, float)
        self._Q_recoverable = Q_recoverable

    @property
    def keff(self):
        return self._keff

    @keff.setter
    def keff(self, keff):
        check_type("keff", keff, float)
        self._keff = keff

    @property
    def keff_stddev(self):
        return self._keff_stddev

    @keff_stddev.setter
    def keff_stddev(self, keff_stddev):
        check_type("keff_stddev", keff_stddev, float)
        self._keff_stddev = keff_stddev

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, nu):
        check_type("nu", nu, float)
        self._nu = nu

    @property
    def flux_normalization(self):
        return self._flux_normalization

    @flux_normalization.setter
    def flux_normalization(self, flux_normalization):
        check_type("flux_normalization", flux_normalization, float)
        self._flux_normalization = flux_normalization

    @property
    def power_renormalization(self):
        return self._power_renormalization

    @power_renormalization.setter
    def power_renormalization(self, power_renormalization):
        check_type("power_renormalization", power_renormalization, float)
        self._power_renormalization = power_renormalization

    @property
    def control_groups(self):
        return self._control_groups

    @control_groups.setter
    def control_groups(self, control_groups):
        check_type("control_groups", control_groups, dict)
        self._control_groups = control_groups

    @property
    def user_tallies_adder_i(self):
        return self._user_tallies_adder_i

    @user_tallies_adder_i.setter
    def user_tallies_adder_i(self, user_tallies_adder_i):
        check_type("user_tallies_adder_i", user_tallies_adder_i, dict)
        self._user_tallies_adder_i = user_tallies_adder_i

    @property
    def user_tally_res(self):
        return self._user_tally_res

    @user_tally_res.setter
    def user_tally_res(self, user_tally_res):
        check_type("user_tally_res", user_tally_res, dict)
        self._user_tally_res = user_tally_res

    @property
    def total_volume(self):
        total_volume = 0.
        for material in self.materials:
            if material.is_depleting and material.status == IN_CORE:
                total_volume += material.volume
        return total_volume

    def __repr__(self):
        return "<Reactor {}>".format(self.name)

    def _init_h5_file(self):
        """Initializes the HDF5 file so subsequent steps can be written
        as they are completed.
        """
        self.h5_file = h5py.File(self.h5_filename, "w", libver="latest")

        # Set the filetype and version
        self.h5_file.attrs['filetype'] = np.bytes_(FILETYPE_REACTOR_H5)
        self.h5_file.attrs['version'] = [VERSION_REACTOR_H5, 0]

        # Store the runtime options as root attributes
        self.h5_file.attrs["name"] = np.bytes_(self.name)
        self.neutronics.init_h5_file(self.h5_file)
        self.depletion.init_h5_file(self.h5_file)

        self.h5_file.attrs["begin_time"] = \
            np.bytes_(self.begin_time.strftime(TIME_STRINGF))
        self.h5_initialized = True

        # TODO: Should add Isotope Registry to file before implementing a full
        # restart capability

    def _open_h5(self):
        if self.h5_initialized:
            self.h5_file = h5py.File(self.h5_filename, "r+", libver="latest")
        else:
            self._init_h5_file()

    def _close_h5(self):
        self.h5_file.close()
        self.h5_file = None

    def update_hdf5(self):
        """Writes the latest to the HDF5 file."""

        # Create the group for this step_idx. Note if it is a case without
        # a step, then the step_idx will be 1.
        group_name = HDF_GROUPNAME.format(self.case_idx, self.operation_idx,
                                          self.step_idx)
        self.log("info", "Writing {} to HDF5 File".format(group_name), 8)

        # Open the H5 File
        self._open_h5()

        # Create the group and assign attributes
        group = self.h5_file.create_group(group_name)
        group.attrs["case_label"] = np.bytes_(self.case_label)
        group.attrs["operation_label"] = np.bytes_(self.operation_label)
        group.attrs["step_idx"] = self.step_idx
        group.attrs["time"] = self.time
        group.attrs["power"] = self.power
        group.attrs["flux_level"] = self.flux_level
        group.attrs["Q_recoverable"] = self.Q_recoverable
        group.attrs["keff"] = self.keff
        group.attrs["keff_stddev"] = self.keff_stddev
        group.attrs["nu"] = self.nu
        group.attrs["flux_normalization"] = self.flux_normalization 
        group.attrs["power_renormalization"] = self.power_renormalization
        end_time = datetime.datetime.now()
        group.attrs["step_end_time"] = \
            np.bytes_(end_time.strftime(TIME_STRINGF))

        # Materials info
        mats_group = group.create_group("materials")
        # write dataset material
        for material in self.materials:
            material.to_hdf5(mats_group)

        # Tallies info
        # write dataset tallies
        tallies_group = group.create_group("user_tallies")
        if "user_tallies" in self.neutronics.__dir__():
            for tally_id in self.neutronics.user_tallies.keys():
                if tally_id not in self.user_tally_res.keys():
                    # assignment of empty values to tally not simulated in MCNP
                    self.neutronics.user_tallies[tally_id].tally_block = []
                    self.neutronics.user_tallies[tally_id].facet_ids = []
                    self.neutronics.user_tallies[tally_id].tally_matrix = np.array([])
                    self.neutronics.user_tallies[tally_id].tally_matrix_err = np.array([])
                # each tally, simulated or not, is saved into HDF5 file
                self.neutronics.user_tallies[tally_id].to_hdf5(tallies_group)
            self.user_tally_res = {}


        # Control group info
        ctrl_group = group.create_group("control_groups")
        for ctrl_object in self.control_groups.values():
            ctrl_object.to_hdf5(ctrl_group)

        # And close the file
        self._close_h5()

    def finalize(self):
        """Finalizes the computation """
        self.end_time = datetime.datetime.now()

        # Finalize H5
        self._open_h5()
        self.h5_file.attrs["end_time"] = \
            np.bytes_(self.end_time.strftime(TIME_STRINGF))
        self.h5_file.close()

    def _flux_scaling_constant(self, nu, keff, use_power):
        """Scales the flux from the normalized value in self.flux to
        the value corresponding to the requested power level.

        Parameters
        ----------
        nu : float
            Average number of neutrons produced per fission.
        keff : float
            Effective multiplication factor from the latest neutronics
            solve.
        use_power : bool
            Whether or not to use power (True) or flux (False).

        Returns
        -------
        scale_constant : float
            The factor to scale the flux by for the requested power
            level
        """

        if use_power:
            # Convert the Q value to MJ / fission instead of MeV/fission
            # (note that the 10^6 factors from J to MJ and eV to MeV cancel
            # each other out)
            Q = JOULE_PER_EV * self.Q_recoverable

            try:
                C = nu * self.power / (keff * Q)
            except ZeroDivisionError:
                msg = "Either a power history is used on a fissile-free system" \
                    " (keff = 0) or the recoverable energy from fission " \
                    "contains a floating-point error.\nIf the former, use " \
                    "a flux history for this model; if the latter, consult a " \
                    "developer"
                self.log("error", msg)
        else:
            if not use_power:
                # Get the total volume * flux for normalization
                tot_vol_flux = 0.
                for material in self.materials:
                    if material.is_depleting and material.status == IN_CORE:
                        tot_vol_flux += material.flux_1g * material.volume
            C = self.flux_level / tot_vol_flux * self.total_volume

        return C

    def _update_material_fluxes(self, flux, nu, keff, use_power, is_zero_power):
        """Given a fluxes array, this method sets the fluxes within
        each current material object

        Parameters
        ----------
        flux : OrderedDict of np.ndarray
            Dictionay where the key is the cell id and the value is the
            calculates group-wise flux.
        nu : float
            Average number of neutrons produced per fission
        keff : float
            Effective multiplication factor from the latest neutronics
            solve.
        use_power : bool
            Whether or not to use power (True) or flux (False).
        is_zero_power : bool
            Whether this is a zero power step (True) or not (False).
        """

        if use_power:
            # Reset the flux level attribute so we can accumulate it
            # here
            self.flux_level = 0.

        if not is_zero_power:
            flux_scaling = self._flux_scaling_constant(nu, keff, use_power)
        else:
            flux_scaling = 0.0

        for material in self.materials:
            if material.is_depleting and material.status == IN_CORE:
                if not is_zero_power:
                    material.flux = flux[material.id] * flux_scaling
                else:
                    material.flux = np.zeros(material.num_groups)
                if use_power and not is_zero_power:
                    # Then we need to update the total flux level
                    self.flux_level += material.flux_1g * material.volume
            else:
                material.flux = np.zeros(material.num_groups)

        # The last step is to normalize flux level by the total volume
        if use_power and not is_zero_power:
            self.flux_level /= self.total_volume

        # Update constants and normalizations
        self.nu = nu
        self.flux_normalization = flux_scaling

    def _update_user_tallies(self, user_tally_res):
        """Given the results read and processed from mctal, this method sets for tally objects
        material_names, universe_names, facet_ids, tally_matrix and tally_matrix_err
        Parameters
        ----------
        user_tally_res: dict
        The results for user tallies, where the key is the tally id and nested dictionaries contain the results.
        Tally results are normalized according to the flux normalization.
        """

        for tally_id in user_tally_res.keys():
            self.neutronics.user_tallies[tally_id].tally_matrix = user_tally_res[tally_id]["tally_matrix"] * \
                                                                  self.flux_normalization
            self.neutronics.user_tallies[tally_id].tally_matrix_err = user_tally_res[tally_id]["tally_matrix_err"]
            self.neutronics.user_tallies[tally_id].facet_ids = user_tally_res[tally_id]["facet_ids"]
            self.neutronics.user_tallies[tally_id].material_names = user_tally_res[tally_id]["material_names"]
            self.neutronics.user_tallies[tally_id].universe_names = user_tally_res[tally_id]["universe_names"]




    def _update_material_fission_quantities(self, dt, 
            renormalize_power_density=False):
        """Given a depletion step duration, this method calculates the
        power density (in W/cm3), fission density (in fissions/cm3) and
        burnup (in MWd) for every material.

        Parameters
        ----------
        dt : float
            Duration (in days) of the depletion step, used to calculated
            fission density and burnup
        renormalize_power_density : bool, optional
            If True, power density is renormalized to total input power
        """
        
        reactor_power_check = 0.0
        step_fiss_density = np.zeros(len(self.materials))

        # Loop through materials and calculate fission-Q and fission rate 
        for imat, material in enumerate(self.materials):
            material_power = 0.0
            material.power_density = 0.0
            if material.is_depleting and material.status == IN_CORE:
                # The flux has already been updated, so it only checks total
                if self.flux_level > 0.0:
                    # Calculate power density and fission density
                    lib = self.depletion_libs[material.depl_lib_name]
                    tot_Q, tot_fiss_rate = material.compute_Q(lib) 
                    # Get power in MeV/s, convert to MW, divide by volume
                    # to get power density in W/cm3
                    material_power = tot_Q * JOULE_PER_EV
                    reactor_power_check += material_power
                    material.power_density = material_power * EV_PER_MEV \
                                           / material.volume 
                    # Calculate material fission density
                    step_fiss_density[imat] = tot_fiss_rate * \
                        dt * SECONDS_PER_DAY / material.volume
                else:
                    # Power density is 0
                    # Fission density and burnup are unchanged
                    material.power_density = 0.0
        
        # Calculate rel. diff. to input reactor power (if >0)
        power_rel_diff = abs(reactor_power_check/self.power-1.) \
                         if self.power else 0.0

        # Now get re-normalization factor
        norm_factor = 1.
        if renormalize_power_density and self.power > 0.0:
            norm_factor = self.power / reactor_power_check

        # Provide user warnings
        msgs = []
        if power_rel_diff > POWER_PRECISION:
            # If the cross-sections were *NOT* calculated notify user
            if self.neutronics.use_depletion_library_xs:  
                msgs.append("XS not updated at this step")
            if renormalize_power_density:
                # Warn user of the re-normalization
                msg = "Power re-normalized from" + \
                     f" {reactor_power_check:.2f} MW to" + \
                     f" {self.power:.2f} MW"
                msgs.append(msg)
                msgs.append(f"  ({power_rel_diff*100.0:.2f}% diff.)")
            else:
                # Warn user of the power difference
                msg = "Calculated power sums to" + \
                     f" {reactor_power_check:.2f} MW," + \
                     f" input power is {self.power:.2f} MW" 
                msgs.append(msg)
                msgs.append(f"  ({power_rel_diff*100.0:.2f}% diff.)")
            for msg in msgs:
                self.log("warning", msg, 10)

        # Loop through materials to update power density and calculate burnup
        reactor_power_check = 0
        reactor_burnup_check = 0
        for imat, material in enumerate(self.materials):
            if step_fiss_density[imat]:
                material.power_density *= norm_factor
                material_power = \
                    material.power_density * material.volume / EV_PER_MEV
                material.fission_density += step_fiss_density[imat]
                material.burnup += material_power * dt
                reactor_power_check += material_power
                reactor_burnup_check += material.burnup

        # Record power renormalization factor
        self.power_renormalization = norm_factor

    def _update_Q_recoverable(self):
        """Updates the recoverable energy from fission
        """

        tot_Q = 0.
        tot_fiss_rate = 0.

        for material in self.materials:
            if material.status == IN_CORE:
                if material.volume is None:
                    msg = "Material {}'s volume is not present; if trying to " + \
                        "run a zero-power depletion first as the first step, " + \
                        "then any volumes analytically obtained in the neutronics " + \
                        "solver are not available and must be manually entered"
                    self.log("error", msg.format(material.name))
                lib = self.depletion_libs[material.depl_lib_name]
                mat_Q, mat_fiss_rate = material.compute_Q(lib)
                self.update_logs(material.logs)
                material.clear_logs()
                tot_Q += mat_Q
                tot_fiss_rate += mat_fiss_rate

        # Its possible for a tot fission rate of 0, if so, set Q_rec = 0
        if tot_fiss_rate > 0.:
            self.Q_recoverable = tot_Q / tot_fiss_rate
        else:
            self.Q_recoverable = 0.


    def _update_depletion_constants(self, flux, nu, keff, use_power, volumes, user_tally_res,
                          is_zero_power, dt, renormalize_power_density=False):
        """This method calculates Q_recoverable, scales and sets the
        fluxes for each material object.

        Parameters
        ----------
        flux : OrderedDict of np.ndarray
            Dictionary where the key is the material id and the value is
            the calculated group-wise flux for each material instance.
        nu : float
            Average number of neutrons produced per fission.
        keff : float
            Effective multiplication factor from the latest neutronics
            solve.
        use_power : bool
            Whether or not to use power (True) or flux (False).
        volumes : OrderedDict
            The material id is the key, the value is the volume in cm^3
        user_tally_res : OrderedDict
            The results for user tallies, listed for tally ids and for each
            tally id: - multi-dimensional array with results,
                      - cell ids, material ids, material names, related to the components
                        "followed" by the tally
        is_zero_power : bool
            Whether this is a zero power step (True) or not (False).
        dt : float
            Duration of the depletion time step, used to calculated
            fission densities and burnup values
        renormalize_power_density : bool, optional
            If True, power density is renormalized to total input power
        """

        # The fluxes are to be set using the scaling factor, and the
        # scaling factor requires Q. Q needs the flux.
        # Luckily the dependency is broken since Q only needs normalized
        # and not scaled fluxes, so we can do that first.
        # Therefore we will calculate as follows:
        # 1) global Q, 2) determine global scaling constant,
        # 3) scale and set fluxes

        # Set the unscaled material flux so we can compute Q
        for material in self.materials:
            if material.is_depleting and material.status == IN_CORE:
                material.flux = flux[material.id]
            else:
                material.flux = np.zeros(material.num_groups)

            # Update the volume, even if not depletable
            if material.status == IN_CORE:
                # Do it for every instance of the material in the model
                material.volume = volumes[material.id]

        # Now compute Q if not a zero power step or simply set to zero otherwise
        if is_zero_power:
            self.Q_recoverable = 0.
        else:
            self._update_Q_recoverable()

        # Now we can determine the scaling factor and set the flux
        self._update_material_fluxes(flux, nu, keff, use_power, is_zero_power)
        self._update_material_fission_quantities(dt,
            renormalize_power_density=renormalize_power_density)

        # Now update tallies
        if "user_tallies" in self.neutronics.__dir__():
            self._update_user_tallies(user_tally_res)

    def init_library(self, filename, library_name):
        """Initializes the depletion library

        Parameters
        ----------
        decay_data : str
            The path to the depletion solver's decay data file
        xs_data : str
            The path to the depletion solver's cross section data file
        default_type : str or None
            The default one-group library to use with the depletion
            solver, if needed.

        """

        msg = "Loading Depletion Data {} From {}".format(library_name,
                                                         filename)
        self.log("info", msg, 4)
        try:
            self.depletion_libs[BASE_LIB] = \
                DepletionLibrary.from_hdf5(filename, library_name)
        except (OSError, FileNotFoundError):
            # FileNotFoundError is necessary for h5py >= 3.2.0
            msg = "Depletion Library {} Was Not Found!".format(filename)
            self.log("error", msg, None)

        # Now check the library contents
        msgs = self.depletion_libs[BASE_LIB].check_library()
        for msg in msgs:
            self.log("info_file", msg, None)

    def   init_materials_and_input(self, neutronics_lib_file, depletion_lib_file,
                                 depletion_lib_name, user_mats_info,
                                 user_univ_info, shuffled_mats, shuffled_univs):
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
        """

        if neutronics_lib_file is not None:
            check_type("neutronics_lib_file", neutronics_lib_file, str)
        check_type("depletion_lib_file", depletion_lib_file, str)
        check_type("depletion_lib_name", depletion_lib_name, str)
        check_type("user_mats_info", user_mats_info, OrderedDict)
        check_type("user_univ_info", user_univ_info, OrderedDict)

        # Initialize the depletion library information
        self.init_library(depletion_lib_file, depletion_lib_name)
        default_depl_lib = self.depletion_libs[BASE_LIB]

        # Define the materials and other information from the neutronics
        # model definitinon
        self.log("info", "Processing Neutronics Input", 4)
        neutronics_library_isos = self.neutronics.parse_library(
            neutronics_lib_file)

        # Set up the isotope registry [we do this after printing
        # Processing Neutronics Input bc self.neutronics.parse_library does the
        # input parsing and that can take a while, but we need to do before
        # self.neutronics.read_input too]
        adder.isotope.ISO_REGISTRY = \
            adder.isotope.IsotopeRegistry(neutronics_library_isos)

        materials = self.neutronics.read_input(
            default_depl_lib.num_neutron_groups, user_mats_info,
            user_univ_info, shuffled_mats, shuffled_univs, self.depletion_libs)

        # Now add the materials to the Reactor class
        self.materials = materials

        # Now add the depletion library to the materials data
        self.log("info", "Assigning Depletion Libraries to Materials", 4)
        for mat in self.materials:
            if mat.is_depleting:
                if mat.status != SUPPLY:
                    if not self.neutronics.use_depletion_library_xs:
                        if not mat.is_default_depletion_library:
                            new_lib = default_depl_lib.clone(new_name=mat.name)
                            self.depletion_libs[new_lib.name] = new_lib
                            mat.depl_lib_name = new_lib.name
                else:
                    # We dont want to clone the library yet as that will happen
                    # when the material is moved in to the core anyways
                    pass
        self.log_materials()

        # Finally ensure the isotope registry has the depletion library info
        # for each and every type of isotope we expect
        adder.isotope.ISO_REGISTRY.register_depletion_lib_isos(
            self.depletion_libs, self.materials)

    def validate_storage_materials(self):
        """Cycle through the materials list and check whether there are any 
        storage materials that:
        - have no density assigned: ADDER should exit with an error
        - have no unique depletion library assigned: this is all the supply
            materials that are re-defined as storage in the ADDER input file 
        Materials are modified in place.
        """
        # Cycle through materials
        for mat in self.materials:
            if mat.status == STORAGE:
                # Check if density was assigned to material
                if not mat.density:
                    # Error: storage element without a density value assigned
                    msg = f"Storage material {mat.name} was not assigned a "\
                           "density in the ADDER input file"
                    self.log("error", msg)
                # Initialize depletion libraries for storage materials
                elif (not self.neutronics.use_depletion_library_xs and
                    not mat.is_default_depletion_library):
                    default_depl_lib = self.depletion_libs[BASE_LIB]
                    new_lib = default_depl_lib.clone(new_name=mat.name)
                    self.depletion_libs[new_lib.name] = new_lib
                    mat.depl_lib_name = new_lib.name

    def _parallel_depletion_manager(self, dt, depletion_step, num_substeps):
        """Executes the depletion in parallel"""

        if dt <= 0.:
            return

        # Determine the number of depleting materials
        num_depl_mats = len(list((i for i, mat in enumerate(self._materials) if
                                  (mat.is_depleting and
                                   mat.status != adder.constants.SUPPLY))))

        msg = "Depleting {:,} Materials with {} Thread".format(
            num_depl_mats, self.depletion.num_threads)
        if self.depletion.num_threads > 1:
            # Make it plural
            msg += "s"
        self.log("info", msg, 8)

        # Precompute the decay data
        # TODO: This can be moved to somewhere else and not done every
        # timestep, however, it has zero runtime cost so who cares if we do it
        # every time
        self.depletion.compute_decay(self.depletion_libs[BASE_LIB])

        # Perform the depletion
        self.log("info", "Executing Depletion", 10)
        weak_deplete = weakref.proxy(self.depletion)
        mat_idxs = (i for i, mat in enumerate(self._materials) if
                    (mat.is_depleting and mat.status != adder.constants.SUPPLY))
        if self.depletion.num_threads == 1:
            for i_mat in mat_idxs:
                mat = self._materials[i_mat]
                flux = mat.flux
                lib = self.depletion_libs[mat.depl_lib_name]
                matrix = weak_deplete.compute_library(lib, flux)

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
                    mat = self.materials[i_mat]
                    lib = self.depletion_libs[mat.depl_lib_name]
                    mtx = self.depletion.compute_library(lib, mat.flux),
                    idxs = lib.isotope_indices
                    inv_idxs = lib.inverse_isotope_indices
                    try:
                        new_isos, new_fracs, new_density = \
                            self.depletion.execute(mat, mtx, idxs, inv_idxs,
                                dt, depletion_step, num_substeps, True)
                    except Exception:
                        error_msg = traceback.format_exc()
                        self.log("error", error_msg)
                    mat.apply_new_composition(new_isos, new_fracs, new_density)

    def _deplete_step(self, log_msg, dt, depletion_step, use_power,
                      num_substeps, is_zero_power, user_tallies_adder_i,
                      include_user_tallies, is_endpoint=False, is_corrector_step=False,
                      is_cecm=False, label=None,
                      renormalize_power_density=False):
        """Advances one step, factoring in if the neutronics computation
        is necessary or not."""

        self.log("info", log_msg, 8)
        user_tally_res={}

        if label is None:
            step_label = "{}; {}: {}".format(self.case_label,
                                             self.operation_label, dt)
        else:
            step_label = label

        if not is_zero_power:
            # Set the defaults, applicable primarily to predictor step
            store_input = True
            user_output = True
            deplete_tallies = True
            fname = \
                STEP_FNAME.format(self.case_idx, self.operation_idx,
                                  self.step_idx) + "p"
            if is_endpoint or dt == 0.:
                # then we are just trying to get the user tally info
                deplete_tallies = False
                fname = STEP_FNAME.format(self.case_idx, self.operation_idx,
                                          self.step_idx) + "e"
            elif is_corrector_step:
                store_input = False
                user_tallies_adder_i = {}
                include_user_tallies = False
                user_output = False
                fname = STEP_FNAME.format(self.case_idx, self.operation_idx,
                                          self.step_idx) + "c"

            # Execute the neutronics solver and get the results
            keff, keff_stddev, flux, nu, volumes, user_tally_res = \
                self.neutronics.execute(fname, step_label, self.materials,
                                        self.depletion_libs, store_input,
                                        deplete_tallies, user_tallies_adder_i,
                                        include_user_tallies, user_output,
                                        self.fast_forward)
            self.user_tally_res = user_tally_res
        else:
            # Set the dummy values we need to so the rest of the
            # processing can proceed
            keff, keff_stddev, nu = 0.0, 0.0, 0.0

            # Create a flux result that has 0s, and get the volumes
            # from the last time
            mat_ids = []
            volumes = OrderedDict()
            for material in self.materials:
                if material.is_depleting:
                    mat_ids.append(material.id)

                volumes[material.id] = material.volume
            # Create zero flux vectors, using material from the last
            # time through the above loop just to get the num of groups
            flux = OrderedDict.fromkeys(mat_ids,
                                        value=np.zeros(material.num_groups))
            # user tally res to zero
            user_tally_res = {}
            fname = "none"

        # Store the output filename
        output_name = fname

        # Print the HDF5 data after we have keff, but before we deplete
        self.keff = keff
        self.keff_stddev = keff_stddev

        # Update the depletion constants before we print to file
        if not is_endpoint:
            if is_cecm:
                dt_fission = 0. if is_corrector_step else dt*2.
            else:
                dt_fission = dt
            self._update_depletion_constants(flux, nu, keff, use_power,volumes,
                                              user_tally_res, is_zero_power, dt_fission,
                                             renormalize_power_density=renormalize_power_density)


        if not is_corrector_step:
            # Store the results to the HDF5 file
            self.update_hdf5()

            if dt > 0.:
                # Execute depletion through the wrapper which handles
                # parallel execution
                self._parallel_depletion_manager(dt, depletion_step,
                                                 num_substeps)

        return output_name

    def _deplete_cecm(self, dt, depletion_step, use_power, num_substeps,
                      user_tallies_adder_i, include_user_tallies, renormalize_power_density=False):
        """Performs the CE/CM method (constant extrap on predictor and
        constant midpoint on corrector, i.e., predictor-corrector."""

        # Store the material inventories we will need, keyed by mat id
        mat_isotopes = {}
        mat_atom_fractions = {}
        mat_densities = {}
        for mat in self.materials:
            if mat.is_depleting and mat.status != SUPPLY:
                mat_isotopes[mat.id] = mat.isotopes[:]
                mat_atom_fractions[mat.id] = mat.atom_fractions.copy()
                mat_densities[mat.id] = mat.density

        log_msg = "Executing Predictor"
        step_label = "{}; {}: {} [predictor]".format(self.case_label,
                                                     self.operation_label, dt)
        fname = self._deplete_step(log_msg, 0.5 * dt, depletion_step,
                                   use_power, num_substeps, False,
                                   user_tallies_adder_i, include_user_tallies,
                                   is_cecm = True, label = step_label,
                                   renormalize_power_density = renormalize_power_density)

        # Now with the inventories from the half-time step
        # (from predictor), we re-compute the fluxes in the next
        # call to _deplete_step (no depletion performed in this next
        # function)
        log_msg = "Executing Corrector"
        step_label = "{}; {}: {} [corrector]".format(self.case_label,
                                                     self.operation_label, dt)
        fname = self._deplete_step(log_msg, dt, depletion_step, use_power,
                                   num_substeps, False,  user_tallies_adder_i,
                                   include_user_tallies, is_corrector_step=True,
                                   is_cecm = True, label = step_label,
                                   renormalize_power_density = renormalize_power_density)

        # Now put the original material inventories back in so we can
        # do the corrector step depletion
        for mat in self.materials:
            if mat.id in mat_isotopes:
                mat.isotopes = mat_isotopes[mat.id]
                mat.atom_fractions = mat_atom_fractions[mat.id]
                mat.density = mat_densities[mat.id]
        # Now do the final depletion to the end point
        self._parallel_depletion_manager(dt, depletion_step, num_substeps)

        # Reset to using previously computed xs
        self.neutronics.use_depletion_library_xs = True

        return fname

    def deplete(self, delta_ts, cumulative_time_steps, powers,
                fluxes,  user_tallies_adder_i, include_user_tallies,
                num_substeps=0, solution_method="cecm", execute_endpoint=True,
                renormalize_power_density=False):

        """Deplete the model according to a given power history.

        Parameters
        ----------
        delta_ts : Iterable of float
            Time duration for each entry in the power history in units
            of days.
        cumulative_time_steps: int
            The total number of depletion time steps that have occured up
            to this deplete block.
        powers : Iterable of float or None
            Total recoverable fission system power for each time bin
            in units of MegaWatts [MW]. If None, then fluxes are
            required
        fluxes : Iterable of float or None
            Neutron Flux level, integrated over all energy and all of
            the depleting spatial domains. If None, then power is
            required
        num_substeps : int, optional
            The number of substeps to use within ORIGEN when performing
            the depletion; defaults to no substeps.
        solution_method : "predictor" or "cecm"
            The depletion algorithm to use; defaults to the "cecm"
            option, the classical predictor-corrector approach.
        execute_endpoint : bool, optional
            Whether the last neutronics computation should be computed or not.
        user_tallies: dict
            Dictionary of tally objects with keys corresponding to their identifiers.
        renormalize_power_density : bool, optional
            If True, power density is renormalized to total input power.
        """

        # Handle selection of power or flux as an input
        if powers is None and fluxes is None:
            self.log("error", "Either Powers or Fluxes must be provided")

        if powers is not None:
            use_power = True
        elif fluxes is not None:
            use_power = False
        else:
            self.log("error", "One of Powers or Fluxes must be provided")

        if use_power:
            if len(powers) != len(delta_ts):
                self.log("error", "delta_ts and powers must be same length")
        else:
            if len(fluxes) != len(delta_ts):
                self.log("error", "delta_ts and fluxes must be same length")

        check_value("solution_method", solution_method,
                    VALID_DEPLETION_METHODS)
        start_time = self.time

        # Turn on that we want to create new depletion library xs for
        # the first pass
        self.neutronics.use_depletion_library_xs = \
            self.neutronics.constant_use_depletion_library_xs

        # For every time step:
        for i in range(len(delta_ts)):
            # depletion_step must be included to keep track of the what
            # cumulative depletion time step we are at by adding
            # cumulative_time_steps to the local deplete block's time step
            depletion_step = cumulative_time_steps + i
            end_time = start_time + delta_ts[i]
            msg = "Evaluating {} time-step {}, [{:.2f}-{:.2f}] days".format(
                self.operation_label, i + 1, start_time, end_time)
            self.log("info", msg, 6)

            # Update the results for the start time
            zero_power = False
            if use_power:
                self.power = powers[i]
                self.flux_level = 0.
                if self.power == 0.:
                    zero_power = True
            else:
                self.flux_level = fluxes[i]
                self.power = 0.
                if self.flux_level == 0.:
                    zero_power = True
            self.time = start_time

            if zero_power:
                # Then we just need to decay and dont need predictor or
                # predictor-corrector
                log_msg = "Executing Zero-Power Depletion"
                output_file = \
                    self._deplete_step(log_msg, delta_ts[i], depletion_step,
                                       use_power, num_substeps, True,
                                       user_tallies_adder_i, include_user_tallies,
                                       renormalize_power_density = renormalize_power_density)
            else:

                if delta_ts[i] == 0.:
                    log_msg = "Evaluating Neutronics Only"
                    output_file = \
                        self._deplete_step(log_msg, delta_ts[i],
                                           depletion_step, use_power,
                                           num_substeps, False,
                                           user_tallies_adder_i, include_user_tallies,
                                           renormalize_power_density = renormalize_power_density)
                else:
                    if solution_method == "predictor":
                        log_msg = "Executing Predictor"
                        output_file = \
                            self._deplete_step(log_msg, delta_ts[i],
                                               depletion_step, use_power,
                                               num_substeps, False,
                                               user_tallies_adder_i, include_user_tallies,
                                               renormalize_power_density = renormalize_power_density)

                    elif solution_method == "cecm":
                        output_file = \
                            self._deplete_cecm(delta_ts[i], depletion_step,
                                               use_power, num_substeps,
                                               user_tallies_adder_i, include_user_tallies,
                                               renormalize_power_density = renormalize_power_density)

            # Advance our time and update states accordingly
            start_time = end_time
            self.step_idx += 1

            # Now set that we want to use the depletion library xs for
            # the remaining steps
            self.neutronics.use_depletion_library_xs = True

        # And give the user some neutronics results at the final time if they
        # asked for it
        # First update the logged results before going to predictor
        # as it will write these to the H5 file
        self.time = end_time
        if execute_endpoint:
            if use_power:
                self.power = powers[-1]
                # self.flux_level was updated  with the last set of material
                # fluxes
            else:
                self.flux_level = fluxes[-1]

            # We will do this for code conciseness with the predictor
            # method but with no duration
            log_msg = "Evaluating {} End Point".format(self.operation_label)
            output_file = self._deplete_step(log_msg, 0., depletion_step,
                                            use_power, num_substeps,
                                            zero_power, user_tallies_adder_i,
                                            include_user_tallies, is_endpoint=True,
                                            renormalize_power_density = renormalize_power_density)
            self.step_idx += 1

    def shuffle(self, start_name, to_names, shuffle_type):
        """Shuffles the object named start_name to the locations listed
        in to_names. This will include handling the cases where the
        start_name object is in storage, supply, or in-core.

        Parameters
        ----------
        start_name : str
            The name of the item (of type shuffle_type) to move
        to_names : Iterable of str
            The names of the objects (of type shuffle_type) to be
            displaced. The 0th entry is the location for start_name to
            be moved to. The 1st entry is for the item displaced by that
            first move, so on and so forth. The last item displaced will
            head to storage.
        shuffle_type : str
            The type of shuffle to perform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        """

        if len(to_names) > 1:
            msg = "Moving {} {}".format(shuffle_type, start_name) + \
                " to the Locations of {}s: (".format(shuffle_type)
            for i in range(len(to_names)):
                if i != 0:
                    msg += " "
                msg += "{}".format(to_names[i])
                if i < len(to_names) - 1:
                    msg += ","
            msg += ")"
        else:
            msg = "Moving {} {}".format(shuffle_type, start_name) + \
                " to the Location of {} {}".format(shuffle_type, to_names[0])
        self.log("info", msg, 6)

        self.neutronics.shuffle(start_name, to_names, shuffle_type,
                                self.materials, self.depletion_libs)

    def transform(self, names, yaw, pitch, roll, angle_units, matrix,
                  displacement, transform_type, matrix_notation, reset):
        """Transforms the universe named `name` according to the angles in
        yaw, pitch, and roll and the translation in displacement.

        Parameters
        ----------
        names : List of str
            The names of the items to transform or the group name itself
        yaw : None or float
            The angle to rotate about the z-axis. If None, then the matrix is
            used instead.
        pitch : None or float
            The angle to rotate about the y-axis. If None, then the matrix is
            used instead.
        roll : None or float
            The angle to rotate about the x-axis. If None, then the matrix is
            used instead.
        angle_units : {"degrees", "radians"}
            The units of the above angles; ignored if a matrix is provided
        matrix : None or np.ndarray
            The rotation matrix, provided in a 3x3 matrix. If None, then yaw,
            pitch, and roll are used instead.
        displacement : Iterable of float
            The displacement vector to translate the object
        transform_type : str
            The type of object to transform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        matrix_notation : {"common", "mcnp"}
            The notation used to define the rotation matrix. The matrix with
            the MCNP notation corresponds to the transpose of the matrix defined
            w/ the common mathematical notation ("common") to perform geometrical 
            rotation.
        reset : bool
            If True, the position of the entity is reset to the initial one 
            provided in the MCNP base input file.
        """

        if transform_type == "group":
            msg = "Transforming group: {}".format(names)
        elif len(names) > 1:
            msg = "Transforming {}s: (".format(transform_type)
            for i in range(len(names)):
                if i != 0:
                    msg += " "
                msg += "{}".format(names[i])
                if i < len(names) - 1:
                    msg += ","
            msg += ")"
        else:
            msg = "Transforming {} {}".format(transform_type, names[0])
        if reset:
            msg += " (after being reset to initial location)"
        self.log("info", msg, 6)

        # Extract group info, if provided
        if transform_type == "group":
            if names in self.control_groups:
                grp = self.control_groups[names]
                names_ = grp.set
                axis_ = grp.axis
                transform_type_ = grp.type
                angle_units_ = grp.angle_units
                val = displacement[0]

                # Now create the yaw, pitch, roll, ... info we need
                yaw_, pitch_, roll_, displacement_ = \
                    get_transform_args(val, axis_)
                matrix_ = None
                matrix_notation_="mcnp"
            else:
                msg = "Invalid Control Group {}".format(names)
                self.log("error", msg)
        else:
            # Just provide the re-name of variables
            names_, yaw_, pitch_, roll_, angle_units_, matrix_, displacement_, \
                transform_type_, matrix_notation_ = \
                names, yaw, pitch, roll, angle_units, matrix, displacement, \
                transform_type, matrix_notation

        self.neutronics.transform(names_, yaw_, pitch_, roll_, angle_units_,
                                  matrix_, displacement_, transform_type_,
                                  matrix_notation=matrix_notation_,reset=reset)

        # Modify group displacement as needed
        if transform_type == "group":
            self.control_groups[names].displacement = val

    def geom_sweep(self, group_name, values, user_tallies_adder_i):
        """Performs a sweep of geometric transformations and executes the
        neutronics solver on each, storing the results for the user. This is
        useful for performing rod sweeps, for example.

        Parameters
        ----------
        group_name : str
            The control group name to work with
        values : List of float
            The values to perform the perturbation in units of cm for
            translations and degrees for rotations
        """

        self.log("info", "Performing Geometry Sweep", 6)

        # Save the original keff information
        orig_keff = self.keff
        orig_keff_stddev = self.keff_stddev

        # Convert values to differentials so we can add on the delta each time
        deltas = [values[0]]
        for i in range(1, len(values)):
            deltas.append(values[i] - values[i - 1])

        # Get the group values if needed
        matrix = None
        if group_name in self.control_groups:
            grp = self.control_groups[group_name]
            names = grp.set
            axis = grp.axis
            transform_type = grp.type
            angle_units = grp.angle_units
        else:
            msg = "Invalid Control Group {}".format(group_name)
            self.log("error", msg)

        # Set the default values to use for each case
        store_input = False
        include_user_tallies = False
        user_output = True
        deplete_tallies = False
        # Sweep through each of the deltas, setup our transform,
        # call the transform, execute and store results
        for i in range(len(deltas)):
            # Obtain info needed to setup the transform
            yaw, pitch, roll, displacement = get_transform_args(deltas[i], axis)

            # Perform the transform
            self.neutronics.transform(names, yaw, pitch, roll, angle_units,
                                      matrix, displacement, transform_type,
                                      transform_in_place=False, reset=False)

            # Execute the neutronics solver for this case
            fname = STEP_FNAME.format(self.case_idx, self.operation_idx,
                                      self.step_idx)
            step_label = "Op {} Case {} Sweep Position {}".format(
                self.case_idx, self.operation_idx, self.step_idx)
            if i == 0:
                update_iso_status = True
            else:
                update_iso_status = False
            keff, keff_stddev, _, _, _, _ = \
                self.neutronics.execute(fname, step_label, self.materials,
                                        self.depletion_libs, store_input,
                                        deplete_tallies, user_tallies_adder_i,
                                        include_user_tallies, user_output,
                                        self.fast_forward,update_iso_status)

            # And store results
            self.keff = keff
            self.keff_stddev = keff_stddev
            self.update_hdf5()
            self.step_idx += 1

        # Restore the original keff
        self.keff = orig_keff
        self.keff_stddev = orig_keff_stddev

        # And undo the transform so the model is as the user left it
        yaw, pitch, roll, displacement = get_transform_args(-values[-1], axis)
        self.neutronics.transform(names, yaw, pitch, roll, angle_units,
                                  matrix, displacement, transform_type,
                                  transform_in_place=False, reset=False) 

    def geom_search(self, group_name, k_target, bracket_interval, 
                    reference_position, target_interval, 
                    uncertainty_fraction, 
                    initial_guess, min_active_batches, max_iterations):
        """Performs a search of geometric transformations to identify that
        which yields the target k-eigenvalue within the specified interval.

        Parameters
        ----------
        group_name : str
            The control group name to work with
        k_target : float
            The target k-eigenvalue to search for
        bracket_interval : Iterable of float
            The lower and upper end of the ranges to search.
        reference_position : str or float
            The reference point for the bracket_interval values.
            'initial' denotes the initial position of the
            control group in the neutronics input; whereas
            'last' referes to the last known position of the
            control group. If float, the value is the position.
        target_interval : float
            The range around the k_target for which is considered success
        uncertainty_fraction : float
            This parameter sets the stochastic uncertainty to target when the
            initial computation of an iteration reveals that a viable solution
            may exist.
        initial_guess : float or str
            The starting point to search. If "last", use the last position.
        min_active_batches : int
            The starting number of batches to run to ensure enough keff samples
            so that it follows the law of large numbers. This is the number of
            batches run when determining if a case has a chance of being a
            suitable solution. This only applies to Monte Carlo solvers.
        max_iterations : int
            The total number of search iterations to perform before terminating
            the search.

        """
        self.log("info", "Performing Geometry Search", 6)

        # Get the group values if needed
        matrix = None
        if group_name in self.control_groups:
            grp = self.control_groups[group_name]
            names = grp.set
            axis = grp.axis
            transform_type = grp.type
            angle_units = grp.angle_units
            # Define reference position for the bracket_interval 
            # and set initial_guess
            if reference_position == "initial": 
                val = 0.0
                reset = True
                msg =  'Reference position (0.0 of interval) set to "initial".'
                if initial_guess == "last":
                    attempt_guess = grp.displacement
            elif reference_position == "last":
                val = 0.0
                reset = False
                msg = 'Reference position (0.0 of interval) set to "last".'
                if initial_guess == "last":
                    attempt_guess = 0.0
            else:
                val = float(reference_position)
                reset = True
                msg = 'Reference position (0.0 of interval) set to specific value.'
                if initial_guess == "last":
                    attempt_guess = grp.displacement - val
            self.log("info", msg, 6)


            yaw, pitch, roll, displacement = get_transform_args(val, axis)
            self.neutronics.transform(names, yaw, pitch, roll,
                                      angle_units, matrix, displacement,
                                      transform_type,
                                      transform_in_place=False, reset=reset)
        else:
            msg = "Invalid Control Group {}".format(group_name)
            self.log("error", msg)

        # Reset the attempt_guess to the upper limit of the bracket_interval 
        # if the last position is not within it
        if initial_guess == "last":
            if (attempt_guess <= bracket_interval[0] or 
                attempt_guess > bracket_interval[1]):
                attempt_guess = bracket_interval[1]
                msg = 'Last position outside of bracket interval. '
                msg += 'Using upper bound as initial guess.'
                initial_guess = bracket_interval[1]
            else:
                msg = 'Using last position as initial guess.'
                initial_guess = attempt_guess
            self.log("info", msg, 6)
        elif not initial_guess == bracket_interval[1]:
            self.log("info", f'Using {initial_guess:.3f} as initial guess.', 6)
        
        # Perform search
        converged, iterations, value, keff, keff_stddev = \
            self.neutronics.geom_search(transform_type, axis, names,
                angle_units, k_target, bracket_interval, reference_position, 
                target_interval,
                uncertainty_fraction, initial_guess, min_active_batches,
                max_iterations, self.materials, self.case_idx,
                self.operation_idx, self.depletion_libs, self.user_tallies_adder_i)

        # Print the log messages
        if converged and value not in bracket_interval:
            msg = "Search Converged at {:.3f} " + \
                "with a k-effective of {:1.6f}+/-{:1.6f} after {} iterations"
            msg = msg.format(value, keff, keff_stddev, iterations)
            self.log("info", msg, 6)
        elif value in bracket_interval:
            msg = "Searched Value is Outside Provided Interval as a value " + \
                "of {:.3f} yielded a k-effective of {:1.6f}+/-{:1.6f}; " + \
                "Re-Evaluate with wider bracket_interval parameter!"
            msg = msg.format(value, keff, keff_stddev)
            self.log("error", msg, 6)
        elif iterations > max_iterations:
            msg = "Search Not Converged at {:.3f} " + \
                "with a k-effective of {:1.6f}+/-{:1.6f} after {} iterations"
            msg = msg.format(value, keff, keff_stddev, max_iterations)
            self.log("error", msg, 6)

        # Place components at their final position
        yaw, pitch, roll, displacement = get_transform_args(value, axis)
        self.neutronics.transform(names, yaw, pitch, roll, angle_units,
                                  matrix, displacement, transform_type,
                                  transform_in_place=False, reset=False)

        # Update results
        self.keff = keff
        self.keff_stddev = keff_stddev
        self.control_groups[group_name].displacement = value
        self.update_hdf5()

    def calc_volumes(self, vol_data, target_unc, max_hist,
                     only_depleting):
        """Computes the volumes of the materials of interest, storing
        them on the Material objects

        Parameters
        ----------
        vol_data : dict
            Parameters for the sampling volume
        target_unc : float
            The target uncertainty, in percent, for which the stochastic
            volume estimation will meet on every region's volume.
        max_hist : int
            The maximum number of histories to run in the stochastic
            volume estimation.
        only_depleting : bool
            Whether or not to iterate on volumes for only depleting
            materials; note this can significantly increase the runtime
            if set to False.
        """

        volumes = self.neutronics.calc_volumes(self.materials, vol_data,
                                               target_unc, max_hist,
                                               only_depleting)
        for material in self.materials:
            # Update the volumes on the materials
            if material.id in volumes:
                material.volume = volumes[material.id]

        self.log_materials()
        self.update_hdf5()

    def write_depletion_library_to_hdf5(self, file_name, mat_names, lib_names,
                                        mode="w"):
        """Writes the depletion library to an HDF5 file for the
        specified material.

        Parameters
        ----------
        file_name : str
            The HDF5 file to write to
        mat_names : List of str
            The names of the material whose libraries we should dump.
        lib_names : List of str
            The names to give the libraries within the file named
            file_name, in the group named lib_name
        mode : {'r', 'r+', 'w', 'w-', 'x', 'a'}
            Write mode for the H5 file
        """

        if mat_names[0].lower() == DEFAULT_LIBRARY_DUMP_ALL:
            # Then we will print all depleting materials
            msg = "Writing all Modified Depletion Libraries"
            self.log("info", msg, 6)
            root = h5py.File(file_name, mode=mode)
            for name, lib in self.depletion_libs.items():
                if name != BASE_LIB:
                    lib.to_hdf5(root)

        else:
            if len(mat_names) > 1:
                msg = "Writing Depletion Library for Materials: ("
                for i in range(len(mat_names)):
                    if i != 0:
                        msg += " "
                    msg += "{}".format(mat_names[i])
                    if i < len(mat_names) - 1:
                        msg += ","
                msg += ")"
            else:
                msg = "Writing Depletion Library for Material " + \
                    "{}".format(mat_names[0])
            self.log("info", msg, 6)

            root = h5py.File(file_name, mode=mode)
            for (lib_name, mat_name) in zip(lib_names, mat_names):
                # Find the material
                name_found = False
                for mat in self.materials:
                    if mat.name == mat_name:
                        name_found = True
                        break
                if not name_found:
                    msg = "Invalid material name, {}".format(mat_name) + \
                        ", to write!"
                    self.log("error", msg)
                lib = self.depletion_libs[mat.depl_lib_name]
                lib.to_hdf5(root, revised_name=lib_name)

    def log_materials(self):
        # Print the material names to the log file
        lines = ["\n", "         ID  |  Depl  |       Name        |  Vol [cc]"]
        lines.append("-" * len(lines[1]))
        for mat in self.materials:
            d_flag = str(mat.is_depleting)
            if mat.volume is None:
                mat_str = "None"
            else:
                mat_str = "{:1.6E}".format(mat.volume)
            lines.append("{:>11}  | {:^6} | {:^17} | {}".format(mat.id, d_flag,
                                                                mat.name,
                                                                mat_str))
        lines.append("")
        msg = "\n".join(lines)
        self.log("info_file", "Material Naming:" + msg)

    @classmethod
    def from_hdf5(cls, filename):
        """Initializes a Reactor from an HDF5 file; the Reactor object
        will be setup to write back to the same hdf5 file

        Parameters
        ----------
        filename : str
            HDF5 file to read from

        Returns
        -------
        this : Reactor
            An initialized Reactor object from the HDF5 file

        """

        check_type("filename", filename, str)

        h5_in = h5py.File(filename, "r", libver="latest")

        # Check the filetype and version
        check_filetype_version(h5_in, FILETYPE_REACTOR_H5, VERSION_REACTOR_H5)

        # Get the initialization parameters
        name = h5_in.attrs["name"].decode()
        neutronics_solver = h5_in.attrs["neutronics_solver"].decode()
        neutronics_mpi = h5_in.attrs["neutronics_mpi_cmd"].decode()
        neutronics_exec = h5_in.attrs["neutronics_exec"].decode()
        depletion_solver = h5_in.attrs["depletion_solver"].decode()
        depletion_exec = h5_in.attrs["depletion_exec"].decode()
        base_neutronics_input_filename = \
            h5_in.attrs["base_neutronics_input_filename"].decode()
        num_neut_threads = int(h5_in.attrs["num_neutronics_threads"])
        num_depl_threads = int(h5_in.attrs["num_depletion_threads"])
        num_mpi_procs = int(h5_in.attrs["num_mpi_procs"])
        depletion_chunksize = int(h5_in.attrs["depletion_chunksize"])
        use_depletion_library_xs = \
            bool(h5_in.attrs["use_depletion_library_xs"])
        reactivity_threshold = float(h5_in.attrs["reactivity_threshold"])
        reactivity_threshold_init = bool(
            h5_in.attrs["reactivity_threshold_initial"])

        this = cls(name, neutronics_solver, depletion_solver, neutronics_mpi,
                   neutronics_exec, depletion_exec,
                   base_neutronics_input_filename, filename, num_neut_threads,
                   num_depl_threads, num_mpi_procs, depletion_chunksize,
                   use_depletion_library_xs, reactivity_threshold,
                   reactivity_threshold_init)

        # Now we can go through each case set the values
        # First get only the groups we care about in the order they were
        # written
        case_group_names = []
        for group_name in h5_in.keys():
            if group_name.startswith("case_"):
                case_group_names.append(group_name)
        case_group_names = sorted(case_group_names)

        # Now in this group we find the latest operation
        case_group = h5_in[case_group_names[-1]]
        op_group_names = []
        for group_name in case_group.keys():
            if group_name.startswith("operation_"):
                op_group_names.append(group_name)
        op_group_names = sorted(op_group_names)

        # And, the latest step
        op_group = case_group[op_group_names[-1]]
        step_group_names = []
        for group_name in op_group.keys():
            if group_name.startswith("step_"):
                step_group_names.append(group_name)
        step_group_names = sorted(step_group_names)

        # Now get the last step point and process that
        group = op_group[step_group_names[-1]]
        case_label = group.attrs["case_label"].decode()
        op_label = group.attrs["operation_label"].decode()
        case_idx = int(case_group_names[-1][len("case_"):])
        op_idx = int(op_group_names[-1][len("operation_"):])
        step_idx = int(step_group_names[-1][len("step_"):])
        time = float(group.attrs["time"])
        power = float(group.attrs["power"])
        flux_level = float(group.attrs["flux_level"])
        Q_recoverable = float(group.attrs["Q_recoverable"])
        keff = float(group.attrs["keff"])
        keff_stddev = float(group.attrs["keff_stddev"])
        case_materials = []
        mat_group = group["materials"]
        for mat_name in mat_group.keys():
            case_materials.append(Material.from_hdf5(mat_group, mat_name))
        materials = case_materials[:]
        case_controls = {}
        ctrl_group = group["control_groups"]
        for ctrl_subgroup in ctrl_group.values():
            data = ControlGroup.from_hdf5(ctrl_subgroup)
            case_controls[data.name] = data

        # Now store the data we need
        # used in the last case
        this.Q_recoverable = Q_recoverable
        this.h5_file = h5_in
        this.case_label = case_label
        this.operation_label = op_label
        this.case_idx = case_idx
        this.operation_idx = op_idx
        this.step_idx = step_idx
        this.time = time
        this.power = power
        this.flux_level = flux_level
        this.kefs = keff
        this.keff_stddev = keff_stddev
        this.materials = materials
        this.control_groups = case_controls

        return this

    def write_neutronics_input(self, fname, deplete_tallies, user_tallies_adder_i,
                               include_user_tallies, user_output):
        """This method writes the neutronics input to disk

        Parameters
        ----------
        fname : str
            The filename to write
        output_options : dict
            The options relevant to each neutronics solver
        deplete_tallies : bool
            Whether or not to write the tallies needed for depletion
        user_tallies_adder_i : dict
            Whether or not to write the user tallies
        include_user_tallies : dict
            Whether or not to write the user tallies
        user_output : bool
            Whether or not to write the user's output control cards
        """

        self.log("info", "Writing Input to {}".format(fname), 6)
        label = STEP_FNAME.format(self.case_idx, self.operation_idx,
                                  self.step_idx)
        self.neutronics.write_input(fname, label, self.materials,
                                    self.depletion_libs, False,
                                    deplete_tallies, user_tallies_adder_i,
                                    include_user_tallies, user_output)
        self.update_hdf5()

    def process_operations(self, ops, no_deplete, fast_forward):
        """This method processes the operations from the input file

        Parameters
        ----------
        ops : List of tuple
            This is either a 3- or 4-tuple. If it is a 4-tuple, it contains
            the case label (str), operation label (str), the name of the
            operation's method (str), and a tuple of the arguments for that
            method.
        no_deplete : bool
            If True, the user has said they do not want us to actually
            execute the neutronics and depletion steps. This is to interrogate
            fuel management.
        fast_forward : bool
            If True, then existing neutronics output in the working directory
            will be read from and not re-computed. This only applies to
            neutronics calculations done in support of deplete blocks.

        """

        # Set fast_forward to the reactor as a means to pass this to operations
        # methods without changing the method_args tuple already built by
        # input.py
        self.fast_forward = fast_forward

        # Now process the operations
        self.log("info", "Processing Operations", 0)
        self.operation_idx = 0

        current_case = ""
        for op in ops:
            self.step_idx = 1
            if len(op) == 4:
                current_op = ""
                # Then this include the case label too, so lets print and assign it
                case_label, op_label, method_name, method_args = op[:]
                # if statement to avoid multiple increment on the same case subsection
                if current_case != case_label:
                    current_case = case_label
                    self.log("info", "Executing Case Block: {}".format(case_label))
                    # And update the case label
                    self.case_label = case_label
                    self.case_idx += 1
                    self.operation_idx = 1

            else:
                op_label, method_name, method_args = op[:]
                # if to avoid multiple increment on the same operation subsection
                if current_op != op_label:
                    current_op = op_label
                    self.operation_idx += 1

            # Assign the operation label
            self.operation_label = op_label
            self.log("info", "Executing Operation: {}".format(op_label), 4)
            if no_deplete and method_name == "deplete":
                self.log("info", "Skipping Depletion", 4)
            else:
                getattr(self, method_name)(*method_args)
        self.log("info", "Completed Processing Operations\n", 0)


def get_neutronics(solver_type, mpi_cmd, exec_cmd, base_input_filename,
                   num_threads, num_procs, use_depletion_library_xs,
                   reactivity_threshold, reactivity_threshold_init):
    """Return a Neutronics subclass depending on the solver type.

    The parameters are the same as the class' __init__ method.

    Returns
    -------
    Neutronics
        A subclass of the abstract Neutronics class for the different
        solver types.
    """

    check_type("solver_type", solver_type, str)
    solver_type = solver_type.lower()
    check_value("solver_type", solver_type, NEUTRONICS_SOLVER_TYPES)

    # Now pick the correct class/subclass depending on solver type
    if solver_type == "mcnp":
        return mcnp.McnpNeutronics(mpi_cmd, exec_cmd, base_input_filename,
                                   num_threads, num_procs,
                                   use_depletion_library_xs,
                                   reactivity_threshold,
                                   reactivity_threshold_init)
    elif solver_type == "test":
        return Neutronics(mpi_cmd, exec_cmd, base_input_filename, num_threads,
                          num_procs, use_depletion_library_xs,
                          reactivity_threshold, reactivity_threshold_init)


def get_depletion(solver_type, exec_cmd, num_threads, num_procs, chunksize):
    """Return a Depletion subclass depending on the solver type.

    The parameters are the same as the class' __init__ method.

    Returns
    -------
    Depletion
        A subclass of the abstract Depletion class for the different
        solver types.
    """

    check_type("solver_type", solver_type, str)
    solver_type = solver_type.lower()
    check_value("solver_type", solver_type, DEPLETION_SOLVER_TYPES)

    # Now pick the correct class/subclass depending on solver type
    if solver_type == "origen2.2":
        return origen22.Origen22Depletion(exec_cmd, num_threads, num_procs,
                                          chunksize)
    elif solver_type.startswith("cram"):
        # Strip off the order after "cram" and convert to integer
        order = int(solver_type.partition("cram")[2])
        return cram.CRAMDepletion(exec_cmd, num_threads, num_procs, chunksize,
                                  order)
    elif solver_type.startswith("msr"):
        # This is either just "msr", and thus the cram order is 16,
        # or the order is specified
        if solver_type == "msr":
            order = 16
        else:
            # Strip off the order after "cram" and convert to integer
            order = int(solver_type.partition("msr")[2])
        return MSRDepletion(exec_cmd, num_threads, num_procs, chunksize,
                            order)
    elif solver_type == "test":
        return Depletion(exec_cmd, num_threads, num_procs, chunksize)


def parallel_init(my_depletion, my_dt, my_step, my_substeps):
    # Pre-load the variables for the depletion worker, this is one way that
    # is used to pass multiple arguments to a parallelized function (here,
    # that parallelized func is depletion_worker). Separate benchmarking
    # has shown this is faster than packing the arguments manually.
    # This also will significantly help with chunksizes > 1 on Pool
    global depletion
    global dt
    global depletion_step
    global num_substeps

    depletion = my_depletion
    dt = my_dt
    depletion_step = my_step
    num_substeps = my_substeps


def parallel_worker(i_mat):
    mat = global_mats[i_mat]
    lib = global_libs[mat.depl_lib_name]
    matrix = depletion.compute_library(lib, mat._flux)

    try:
        result = depletion.execute(mat, matrix, lib.isotope_indices,
                                   lib.inverse_isotope_indices, dt,
                                   depletion_step, num_substeps, False)
        result = (i_mat, *result)
    except Exception:
        # instead, convert the traceback to an error string
        error = traceback.format_exc()
        result = (i_mat, error, None, None)
    return result
