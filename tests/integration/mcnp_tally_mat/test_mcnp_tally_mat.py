import pytest
import os
import h5py
import numpy as np
import glob
from tests.testing_harness import TestHarness

inp_file ="""ADDER test tally materials
c 
c CELLS
c
c Fuel Pin #1
c
101    11  -20.0  -51 31 -101  imp:n=1
102    12  -20.0  -51 101 -102  imp:n=1
103    13  -20.0  -51 102 -103  imp:n=1
104    14  -20.0  -51 103 -104  imp:n=1
105    15  -20.0  -51 104 -32 imp:n=1
c
c Fuel Pin #2
c
201    21  -20.0  -52 31 -101  imp:n=1
202    22  -20.0  -52 101 -102  imp:n=1
203    23  -20.0  -52 102 -103  imp:n=1
204    24  -20.0  -52 103 -104  imp:n=1
205    25  -20.0  -52 104 -32 imp:n=1
c
c Fuel Pin #3
c
301    31  -20.0  -53 31 -101  imp:n=1
302    32  -20.0  -53 101 -102  imp:n=1
303    33  -20.0  -53 102 -103  imp:n=1
304    34  -20.0  -53 103 -104  imp:n=1
305    35  -20.0  -53 104 -32 imp:n=1
c
c Fuel Pin #4
c
401    41  -20.0  -54 31 -101  imp:n=1
402    42  -20.0  -54 101 -102  imp:n=1
403    43  -20.0  -54 102 -103  imp:n=1
404    44  -20.0  -54 103 -104  imp:n=1
405    45  -20.0  -54 104 -32 imp:n=1
c
c Fuel Pin #5
c
501    51  -20.0 -101  u=1 imp:n=1
502    52  -20.0  101 -102  u=1 imp:n=1
503    53  -20.0  102 -103  u=1 imp:n=1
504    54  -20.0  103 -104  u=1 imp:n=1
505    55  -20.0  104  u=1 imp:n=1
c
c Central pin
5    3  -2.7   -61  41 -42  imp:n=1
c Water
1001   2  -1.0   11 -12  21 -22  31 -32  51  52  53  54  61  imp:n=1  $ Water in between pins
1002   2  -1.0   11 -12  21 -22 -31  33                  61  imp:n=1  $ Water below fuel pins
1003   2  -1.0   11 -12  21 -22  32 -34                  61  imp:n=1  $ Water above fuel pins
1004   2  -1.0   11 -12  21 -22 -41  33                 -61  imp:n=1  $ Water above control rod
1005   2  -1.0   11 -12  21 -22  42 -34                 -61  imp:n=1  $ Water below control rod
c Outer world
9999 0  -11  12 -21  22 -33  34  imp:n=0

c 
c SURFACES
c
*11  px  -1.3
*12  px   1.3
*21  py  -1.3
*22  py   1.3
31   pz  -50.0
32   pz   50.0
33   pz  -150.0
34   pz   150.0
41   pz  -50.0
42   pz   50.0
51   1 cz 0.5
52   2 cz 0.5
53   3 cz 0.5
54   4 cz 0.5
61   cz   0.4
c
c Fuel pin axial division
c
101   pz  -48.000
102  pz  -16.000
103   pz  16.000
104   pz  48.000
c

c
c MATERIALS
c 
m0 nlib=70c
c UO2 reference fuel
m1 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
c UO2 fuel P1, P2, P3, P4, P5, PS, PS3, PS4
m11 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m12 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m13 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m14 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m15 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m21 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m22 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m23 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m24 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m25 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m31 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m32 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m33 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m34 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m35 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m41 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m42 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m43 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m44 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m45 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m51 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m52 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m53 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m54 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m55 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m61 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m62 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m63 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m64 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m65 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m71 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m72 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m73 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m74 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m75 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m81 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m82 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m83 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m84 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m85 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
m100 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
c Water
m2  1001.70c -0.1111  8016.70c -0.8889
c UO2 central pin fuel
m3 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
c
c TRANSFORMATIONS
c
tr1  0.65  0.65 0.0
tr2 -0.65  0.65 0.0
tr3 -0.65 -0.65 0.0
tr4  0.65 -0.65 0.0
c
c CALCULATION PARAMETERS
c kcode 10000 1.0 60 210
c Calc. for a simple simulation test case
kcode 1000 1.0 50 150
DBCN 23j 0.1
ksrc 
        0.650    0.650  -49.000
        0.650    0.650  -15.000
        0.650    0.650   15.000
        0.650    0.650   49.000
       -0.650    0.650  -49.000
       -0.650    0.650  -15.000
       -0.650    0.650   15.000
       -0.650    0.650   49.000
        0.650   -0.650  -49.000
        0.650   -0.650  -15.000
        0.650   -0.650   15.000
        0.650   -0.650   49.000
       -0.650   -0.650  -49.000
       -0.650   -0.650  -15.000
       -0.650   -0.650   15.000
       -0.650   -0.650   49.000
c User Tallies
fc4 Fn simple form for initial pos. 1
f4:n 101 102 103 104 105
fm4 0.04786 1 1
e4 1E-5 1E-3
cf4 103
fc14 Fn general form for initial pos. 1
f14:n 101 (102 103 104 105)
fc24 Additional tally for initial pos. 2
f24:n 201 202 203 204 205
fc1 test tally F1
f1:n 51 52
fc2 test tally F2
f2:n 51 52
fc6 test tally F6 and p part.
f6:p 101 102
fc7 test tally F7
f7:n 101 102
"""


class TallyMatHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def _get_results(self):

        # Get results from inputs
        outstr = super()._get_results()+"\n"

        # Process results.h5 using adder_extract_tallies
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_34.csv 34")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_44.csv 44")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_54.csv 54")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_64.csv 64")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_16.csv 16")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_17.csv 17")

        tally_l = ["tally_34", "tally_44", "tally_54", "tally_16",
                   "tally_17"]
        # get results string
        for i in range(len(tally_l)):
            if i == 0:
                outstr += tally_l[i]+"\n"
            else:
                outstr += tally_l[i]+"\n"

            with open(tally_l[i]+".csv", "r") as fin:
                outstr += "".join(fin.readlines())

        return outstr

    def _compare_results(self, tol=1e-15):
        """This function compares files with an allowed tolerance for
        relative variation of numbers from data results."""

        with open('results_test.dat', "r") as results_test:
            results_test_lines = results_test.readlines()

        with open(self.results_true_fname) as results_true:
            results_true_lines = results_true.readlines()

        test_words, true_words = [], []

        for test_line, true_line in zip(results_test_lines, results_true_lines):
            test_words.extend(test_line.split())
            true_words.extend(true_line.split())

        # Ensure files have the same number of words
        assert len(results_test_lines) == len(results_true_lines)

        for i, test_word in enumerate(test_words):
            true_word = true_words[i]
            if ('.' and ',') in test_word and ('.' and ',') in true_word:
                test_numbers, true_numbers = test_word.split(","), true_word.split(",")
                for j, test_number in enumerate(test_numbers):
                    true_number = true_numbers[j]
                    if self._is_float(test_number) and self._is_float(true_number):
                        if float(true_number) != 0:
                            # Ensure relative variation match within tolerance
                            assert (abs((float(test_number) - float(true_number)) /
                                           float(true_number)) < tol)
                        else:
                            assert (float(test_number) == 0)
            else:
                # Ensure any other words match exactly
                assert test_word == true_word

        # Check correct assignment of power densities
        self._check_zero_power_densities('results.h5')

    def _cleanup(self):
        """Delete statepoints, tally, and test files."""

        # Do whatever the harness wants
        super()._cleanup()

        # And add our own
        for csv in glob.glob("*.csv"):
            os.remove(csv)

    @staticmethod
    def _check_zero_power_densities(h5_filename):
        # Power densities should be zero for materials that are in storage
        # NOTE: this test does not check whether the power densities are correct
        # just that - for out-of-core materials - fission densities and total 
        # burnup are unchanged and power density is zero
        fission_densities = {}
        burnups = {}
        in_core_checked = 0
        storage_checked = 0
        supply_checked = 0
        non_depleting_checked = 0
        with h5py.File(h5_filename, 'r') as data:
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
                            for material in materials.values():
                                mat_id = material['id'][0]
                                pden = material['power_density'][0]
                                fiss = material['fission_density'][0]
                                bu = material['burnup'][0]
                                status = material['status'][0]
                                is_depleting = material['is_depleting'][0]
                                if is_depleting:
                                    if status == 0:
                                        # In-core material, should have depleted
                                        assert pden > 0.0
                                        assert fiss > 0.0
                                        assert bu > 0.0
                                        in_core_checked += 1
                                    else:
                                        # Storage material, should have 0.0 power
                                        # density, identical fission density and
                                        # burnup
                                        assert pden == 0.0
                                        if mat_id in fission_densities:
                                            assert fiss == fission_densities[mat_id]
                                            assert bu == burnups[mat_id]
                                        if status == 1:
                                            storage_checked += 1
                                        elif status == 2:
                                            supply_checked += 1
                                else:
                                    assert pden == 0.0
                                    assert fiss == 0.0
                                    assert bu == 0.0
                                    non_depleting_checked += 1
                                # Record current values
                                fission_densities[mat_id] = fiss
                                burnups[mat_id] = bu
    
        # Assert that I've checked everything
        assert in_core_checked == 84
        assert storage_checked == 10
        assert supply_checked == 88
        assert non_depleting_checked == 20

def test_tally_mat():
    # This tests the management of tallies
    # "following" materials during depletion.

    # Warning message given when materials with no volumes are shuffled.
    mcnp_text_files  = ["2nd_step_shuffling_p2p1.inp", "4th_step_shuffling_p1_p2.inp",
                        "case_1_op_1_step_1p.inp", "case_1_op_2_step_1p.inp",
                        "case_3_op_2_step_1p.inp", "case_4_op_3_step_1p.inp",
                        "6th_step_shuffling_ps2_p4.inp"]
    test = TallyMatHarness(mcnp_text_files, "test.h5")
    test._build_inputs()
    test.main()






