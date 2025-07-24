import pytest
from tests import mcnp_cubes
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
            mcnp_input_file.write(mcnp_cubes)


def test_mcnp_transform_byuniv_large_id():
    # This tests MCNPs ability to transform fuel using universes.
    # Additional tests are added to check the rotation via the
    # the transform subsection using the yaw, pitch, roll and 
    # matrix attributes w/ the mcnp (default) and common notation.	

    output_text_files = ["state{}.inp".format(i + 1) for i in range(4)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()
