import pytest
import h5py
import numpy as np
from adder.constants import IN_CORE, STORAGE, SUPPLY
from tests import mcnp_lattice
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
            _check_universes_status("results.h5")
        finally:
            self._cleanup()

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_lattice)


def test_mcnp_shuffle_byuniv():
    # This tests MCNPs ability to shuffle fuel using universes.
    # Additional tests are added to check the rotation via the
    # the transform subsection using the yaw, pitch, roll and 
    # matrix attributes w/ the mcnp (default) and common notation.	

    output_text_files = ["state{}.inp".format(i + 1) for i in range(15)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()


def _check_universes_status(h5_filename):
    """
    This function checks the status of selected materials from the HDF5 file
    against reference values provided via the _base_materials (initial) 
    dictionary and _update_materials_status function. Note that materials are
    checked instead of universes as universes information is not retained in 
    the HDF5 file. All the operations in this integration test however 
    (tests/integration/shuffle_byuniv) are performed by universes, so this 
    tests that shuffled universes have their status set correctly, which should
    be reflected in their materials.
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
    # Fuel materials
    '1': IN_CORE,
    '2': IN_CORE,
    '3': IN_CORE,
    '4': SUPPLY,
    '5': IN_CORE,
    '6': IN_CORE,
    '7': IN_CORE,
    '8': IN_CORE,
    '9': IN_CORE,
    '11': STORAGE,
    '12': STORAGE,  # 11 and 12 are part of the same material u=11
    '13': STORAGE,  
    '14': STORAGE,  # 13 and 14 are part of the same material u=12
    # Non-depleting materials, copied from 10.
    # Only the base material (m10) and the copy for universe 1 are tested here
    # Copies for supply universes are also tested when relevant (see below)
    '10': IN_CORE,  # First instance 
    '16': IN_CORE,  # Copied for universe 1
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
        materials['1'] = previous['9']
        materials['16'] = previous['9']
        materials['9'] = previous['1']
    elif write_id == 3:
        materials['3'] = previous['7']
        materials['7'] = previous['3']
    elif write_id == 4:
        materials['11'] = previous['5']
        materials['12'] = previous['5']
        materials['5']  = previous['11']
    elif write_id == 5:
        materials['5'] = previous['8']
        materials['8'] = previous['5']
    elif write_id == 6:
        materials['11'] = previous['5']
        materials['12'] = previous['5']
        materials['5']  = previous['1']
        materials['1']  = previous['6']
        materials['16'] = previous['6']
        materials['6']  = previous['11']
    elif write_id == 7:
        materials['13'] = previous['1']
        materials['14'] = previous['1']
        materials['1']  = previous['13']
        materials['16'] = previous['13']
    elif write_id == 8:
        materials['27'] = IN_CORE
        materials['28'] = IN_CORE   # Non-depleting material, copy of m10
        materials['13'] = STORAGE
        materials['14'] = STORAGE
    elif write_id == 9:
        materials['27'] = previous['3']
        materials['28'] = previous['3']
        materials['3']  = previous['27']
    elif write_id == 15:
        materials['29'] = IN_CORE 
        materials['30'] = IN_CORE   # Non-depleting material, copy of m10 
        materials['7']  = previous['6']
        materials['6']  = previous['3']
        materials['3']  = STORAGE
    else:
        pass

    return materials

