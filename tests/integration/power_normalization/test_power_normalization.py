import pytest
import os
import h5py
import numpy as np
from tests import mcnp_2x2x2
from tests import default_config as config
from tests.testing_harness import TestHarness
from adder.depletionlibrary import DepletionLibrary

ADDER_EXEC = config["exe"]

power_input = [1, 2, 1.5, 2.5, 0, 0, .5, .7]
durations = [1, 1, 1, 1, 2, 1, 4, 3]

POWER_TOL = 1e-2
XS_NOT_UPDATED_COUNT = 3
RENORMALIZED_COUNT = 1

class MCNPHarness(TestHarness):
    def execute_test(self):
        """Run ADDER as in TestHarness, get results, run again and make sure
        the results are the same and that MCNP was skipped. Then cleanup"""
        try:
            self._create_test_lib()
            self._run_adder()
            self._compare_power()
        finally:
            self._cleanup()

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2)

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""

        # And perform the rest
        super()._compare_results()

    def _compare_power(self):

        # Initialize HDF5 data and supporting variables
        data = h5py.File('results.h5', 'r')
        iop = 0
        burnup_check, fissions_integral = 0.0, 0.0
        burnup = 0.0
        
        # Collect warnings and calculated percent differences
        percent_differences = self._check_renormalization_warnings()

        # Loop through operations to extract power
        for case_name in data:
            case = data[case_name]
            for operation_name in case:
                operation = case[operation_name]
                for step_name in operation:
                    power, fissions = 0.0, 0.0
                    step = operation[step_name]
                    op_label = step.attrs['operation_label'].decode('UTF-8')
                    if op_label.startswith('deplete'):
                        materials = step['materials']
                        dt = durations[iop]
                        Qrec = step.attrs['Q_recoverable']
                        Pnorm = step.attrs['power_renormalization']
                        for material in materials.values():
                            mat_power = material['power_density'] * 1e-6 * \
                                        material['volume']
                            power += mat_power 
                            burnup += mat_power * dt 
                            fissions += material['fission_density'] * \
                                             material['volume']
                        if Pnorm == 1.:
                            # Power is not renormalized, hence should be equal
                            # to the number of fissions * Qrec
                            power_check = (fissions - fissions_integral) / \
                                          (dt * 86400) * \
                                          Qrec * 1.6021766208e-19 
                        else:   
                            # Power is renormalized
                            power_check = power_input[iop]
                        burnup_check += power_check * dt

                        # Assert power, burnup, and fission values
                        if power_check or power:
                            assert (abs(power_check - power)/power_check) < 1e-12
                        if burnup_check or burnup:
                            assert (abs(burnup_check - burnup)/burnup_check) \
                               < 1e-12

                        # Compare to input values
                        if power_input[iop]:
                            if Pnorm == 1.0:
                                # These were not renormalized, some difference
                                # is expected
                                calc_diff = abs(power_input[iop] - power) /\
                                                power_input[iop]
                                if percent_differences[iop]:
                                    assert np.round(calc_diff*100, 2) == \
                                        percent_differences[iop]
                                else:
                                    assert calc_diff < POWER_TOL
                            else:
                                # These were renormalized
                                # Power should be identical, check Pnorm
                                assert (abs(power_input[iop] - power) /\
                                        power_input[iop]) < 1e-12
                                if percent_differences[iop]:
                                    assert np.round((1-1/Pnorm)*100,2) == \
                                            percent_differences[iop]
                                else:
                                    assert np.round(1-1/Pnorm,2) <\
                                            POWER_TOL

                        # Update variables for next step
                        iop += 1
                        fissions_integral = fissions

    def _check_renormalization_warnings(self):
        # This checks warnings in the adder.log file 
        power_differences = []
        renormalized_count = 0
        xs_not_updated_count = 0
        difference_count = 0
        iop = 0
        is_predictor = True
        with open('adder.log', 'r') as log:
            for line in log:
                if 'Evaluating deplete' in line:
                    power_differences.append(0.0)
                elif 'Executing Predictor' in line:
                    is_predictor = True
                elif 'Executing Corrector' in line:
                    is_predictor = False
                elif 'XS not updated' in line:
                    xs_not_updated_count += 1
                elif 'Power re-normalized' in line:
                    renormalized_count += 1
                elif 'diff.' in line and '%' in line:
                    difference_count += 1
                    if is_predictor: 
                        power_differences[-1] = float( 
                            line.split('%')[0].split('(')[1])

        # Assert presence of expected warnings
        assert renormalized_count == RENORMALIZED_COUNT 
        assert xs_not_updated_count == XS_NOT_UPDATED_COUNT
        assert difference_count >= xs_not_updated_count

        return power_differences

def test_power_normalization():
    # This tests the execution of MCNP

    output_text_files = ["state{}.inp".format(i) for i in range(4)]
    test = MCNPHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()
