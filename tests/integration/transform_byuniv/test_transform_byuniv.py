import pytest
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
        finally:
            self._cleanup()

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_lattice)

def test_mcnp_transform_byuniv_storage_error():
    # Test that ADDER exits with an error if a transform is requested on a 
    # storage universe
    error_msg  = 'ERROR - Storage Universes can not be transformed. ' 
    error_msg += 'Move universe 12 to in-core first.'
    test = ShuffleHarness([], "test.h5", input_fname = "test_error_storage.add")
    try:
        test._create_test_lib()
        test._build_inputs()
        test._run_adder()
        with open('adder.log', 'r') as log:
            for line in log:
                pass
            last_line = line
            assert error_msg in last_line
    finally:
        test._cleanup()

def test_mcnp_transform_byuniv_supply_error():
    # Test that ADDER exits with an error if a transform is requested on a 
    # supply universe storage universe
    error_msg  = 'ERROR - Supply Universes can not be transformed. '
    error_msg += 'Use an in-core instance instead (e.g., 12[42]).' 
    test = ShuffleHarness([], "test.h5", input_fname = "test_error_supply.add")
    try:
        test._create_test_lib()
        test._build_inputs()
        test._run_adder()
        with open('adder.log', 'r') as log:
            for line in log:
                pass
            last_line = line
            assert error_msg in last_line
    finally:
        test._cleanup()

def test_mcnp_transform_byuniv():
    # This tests MCNPs ability to transform fuel using universes.
    # Additional tests are added to check the rotation via the
    # the transform subsection using the yaw, pitch, roll and 
    # matrix attributes w/ the mcnp (default) and common notation.	

    output_text_files = ["state{}.inp".format(i + 1) for i in range(8)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()

