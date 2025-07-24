import pytest
import re
import h5py
import numpy as np
from adder.constants import IN_CORE, STORAGE, SUPPLY
from tests import mcnp_2x2x2_trcl
from tests.testing_harness import TestHarness


class ShuffleHarness(TestHarness):

    def execute_test(self):
        """Run ADDER with the appropriate arguments and check the outputs."""
        try:
            self._create_test_lib()
            self._run_adder()
            results = self._get_results()
            self._write_results(results)
            self._compare_mcnp_files()
            _check_materials_status("results.h5")
        finally:
            self._cleanup()

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2_trcl)

    def _build_inputs_no_volumes(self):
        mcnp_2x2x2_trcl_no_volumes = re.sub("vol 1000.0 7R 0.0", "c",
                                            mcnp_2x2x2_trcl)
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2_trcl_no_volumes)


def test_mcnp_shuffle_bymat():
    # This tests MCNPs ability to shuffle fuel. The second test is the same as
    # the first, but the volumes are removed from the materials in the MCNP
    # input to check if ADDER raises a warning during the shuffling of materials
    # with no volumes. Additional tests are added to check the rotation via the
    # the transform subsection using the yaw, pitch, roll and matrix attributes
    # w/ the mcnp (default) and common notation. 

    output_text_files = ["state{}.inp".format(i + 1) for i in range(8)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()

    # Warning message given when materials with no volumes are shuffled.
    warning_message = "At least one of the in-core materials being shuffled " \
                      "does not have a volume set!"

    output_text_files = ["state0.inp"]
    test = ShuffleHarness(output_text_files, "test.h5")
    test.log_messages = [('warning', warning_message, 9)]
    test._build_inputs_no_volumes()
    test._create_test_lib()
    test._run_adder()
    test._check_log_messages()
    test._cleanup()


def _check_materials_status(h5_filename):
    """
    This function checks the status of selected materials from the HDF5 file
    against reference values provided via the _base_materials (initial) 
    dictionary and _update_materials_status function. 
    """
    with h5py.File(h5_filename, "r") as h5:
        i_step = 0
        i_checks = 0
        i_tot_references = 0
        reference = {}
        case_ids = np.sort([int(x.split('_')[-1]) for x in h5.keys()])
        for case_name in [f'case_{x}' for x in case_ids]:
            case = h5[case_name]
            op_ids = np.sort([int(x.split('_')[-1]) for x in case.keys()])
            for operation_name in [f'operation_{x}' for x in op_ids]:
                operation = case[operation_name]
                step_ids = np.sort([int(x.split('_')[-1]) 
                                    for x in operation.keys()])
                for step_name in [f'step_{x}' for x in step_ids]:
                    step = operation[step_name]
                    materials = step['materials']
                    i_step += 1
                    reference = _update_materials_status(reference, i_step)
                    i_tot_references += len(reference)
                    if reference:
                        for material_name in materials:
                            data = materials[material_name]
                            material = data[()]
                            status = material['status']
                            material_id = material['id'][0]
                            try:
                                ref_status = reference[str(material_id)]
                            except KeyError as e:
                                # This material is not one of our checks
                                continue

                            # Check status
                            i_checks += 1
                            assert int(status) == int(ref_status)

                    # Check that all the available references were checked
                    assert i_checks == i_tot_references


_base_materials = {
    '11': IN_CORE,
    '12': IN_CORE,
    '21': IN_CORE,
    '22': IN_CORE,
    '31': IN_CORE,
    '32': IN_CORE,
    '41': IN_CORE,
    '42': IN_CORE,
    '51': SUPPLY,
    '61': STORAGE,
    '62': STORAGE,  
}

def _update_materials_status(previous, write_id):
    """ 
    This function reflects 1-to-1 the shuffle and revolve operations in 
    the test.add file 
    """
    materials = previous.copy()
    if write_id == 1:
        materials = _base_materials.copy()
    elif write_id == 2:
        materials['63'] = IN_CORE           #fresh[1]
        materials['42'] = STORAGE
        materials['11'] = previous['21']
        materials['21'] = previous['11']
        materials['12'] = previous['22']
        materials['22'] = previous['12']
    elif write_id == 3:
        materials['61'] = previous['21']
        materials['21'] = previous['61']
        materials['62'] = previous['22']
        materials['22'] = previous['62']
    elif write_id == 4:
        materials['64'] = IN_CORE           #fresh[2]
        materials['63'] = previous['41'] 
        materials['41'] = STORAGE
    elif write_id == 5:
        # Revolve top
        materials['62'] = previous['31']
        materials['31'] = previous['64']
        materials['64'] = previous['11']
        materials['11'] = previous['62']
        # Revolve bottom
        materials['12'] = previous['61']
        materials['61'] = previous['32']
        materials['32'] = previous['63']
        materials['63'] = previous['12']
    elif write_id == 6:
        materials['62'] = previous['12']
        materials['12'] = previous['62']
    else:
        pass

    return materials

