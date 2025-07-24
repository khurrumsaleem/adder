import pytest
import re
from tests import mcnp_2x2x2_trcl
from tests.testing_harness import TestHarness


class TransformHarness(TestHarness):

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
            mcnp_input_file.write(mcnp_2x2x2_trcl)


def test_mcnp_transform():
    # This tests MCNPs ability to transform cells, surfaces, and universes.
    # The yaw, pitch, roll, displacement, value, and matrix attributes
    # are all tested separately or in conbination. 

    output_text_files = ["state{}.inp".format(i + 1) for i in range(8)]
    test = TransformHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()

