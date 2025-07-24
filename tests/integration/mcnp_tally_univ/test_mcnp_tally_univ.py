import pytest
import os
import h5py
import numpy as np
import glob
from tests.testing_harness import TestHarness

inp_file = """ADDER MCNP tally with universes
c 
c CELLS
c
c Fuel Pin #1
c
101    1  7.419604E-02 -101  imp:n=1 u=1 vol=1
102    1  -20.0  101 -102  imp:n=1 u=1 vol=1
103    1  -20.0  102 -103  imp:n=1 u=1 vol=1
104    1  -20.0  103 -104  imp:n=1 u=1 vol=1
105    1  -20.0  104  imp:n=1 u=1 vol=1
c
c Fuel Pin #2
c
201    1  -20.0 -101  imp:n=1 u=2
202    1  -20.0  101 -102  imp:n=1 u=2
203    1  -20.0  102 -103  imp:n=1 u=2
204    1  -20.0  103 -104  imp:n=1 u=2
205    1  -20.0  104  imp:n=1 u=2
c
c Fuel Pin #3
c
301    1  -20.0 -101  imp:n=1 u=3
302    1  -20.0  101 -102  imp:n=1 u=3
303    1  -20.0  102 -103  imp:n=1 u=3
304    1  -20.0  103 -104  imp:n=1 u=3
305    1  -20.0  104  imp:n=1 u=3
c
c Fuel Pin #4
c
401    1  -20.0 -101  imp:n=1 u=4
402    1  -20.0  101 -102  imp:n=1 u=4
403    1  -20.0  102 -103  imp:n=1 u=4
404    1  -20.0  103 -104  imp:n=1 u=4
405    1  -20.0  104  imp:n=1 u=4
c
c Fuel Pin #5
c
501    1  -20.0 -101  imp:n=1 u=5
502    1  -20.0  101 -102  imp:n=1 u=5
503    1  -20.0  102 -103  imp:n=1 u=5
504    1  -20.0  103 -104  imp:n=1 u=5
505    1  -20.0  104  imp:n=1 u=5
c
c Fuel pins
1    0  -51  31 -32  imp:n=1 *fill=1 (1) 
2    0  -52  31 -32  imp:n=1 *fill=2 (2)
3    0  -53  31 -32  imp:n=1 *fill=3 (3)
4    0  -54  31 -32  imp:n=1 *fill=4 (4)
c Experimental location, repeated structure
5    0  -61  41 -42  imp:n=1 fill=6
6 0 71 -72  73 -74 lat=1 u=6 imp:n=1 fill=-2:2 -2:2 0:0 &
8 8 8 8 8 8 7 7 7 8 8 7 7 7 8 8 7 7 7 8 8 8 8 8 8
7 3 -11.0 -75 imp:n=1 u=7 vol=1
8 2 -1.0 75 imp:n=1 u=7
9 2 -11.0 11 -12  21 -22 imp:n=1 u=8
10 3  -2.7   11 -12  21 -22  imp:n=1 u=9
c Control rod
c 5    3  -2.7   -61  41 -42  imp:n=1
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
c Surf for rep. structure
71   px  -0.1
72   px  +0.1
73   py  -0.1
74   py  +0.1
75   cz  0.05
c
c Fuel pin axial division
c
101   pz  -48.000
102  pz  -16.000
103   pz  16.000
104   pz  48.000
c
c Fuel pin transversal division P1
510 px 0
511 py 0

c
c MATERIALS
c 
m0 nlib=70c
c UO2 fuel
m1 92235.70c -0.05 92238.70c -0.83  8016.70c -0.12
c Water
m2  1001.70c -0.1111  8016.70c -0.8889
c Aluminum
m3  54135.70c -1
c
c TRANSFORMATIONS
c
tr1  0.65  0.65 0.0
tr2 -0.65  0.65 0.0
tr3 -0.65 -0.65 0.0
tr4  0.65 -0.65 0.0
c
c CALCULATION PARAMETERS
c kcode 10000 1.0 60 10
c Calc. for a simple simulation test case
kcode 1000 1.0 25 100
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
fc4 tally to follow p1, Fn simple form
f4:n 101 102 103 104 105
fm4 0.04786 1 1
e4 1E-5 1E-3
cf4 103
fc14 tally to follow p1, Fn general form
f14:n 101 (102 103 104 105)
fc24 tally to follow p2
f24:n 201 202 203 204 205
fc54 tally to follow p5, supply element
f54:n 501 502 503 504 505
fc64 tally for repeated structure
f64:n (7 < 6 [2 2 0] <5)
fc1 test tally F1
f1:n 51 52
fc2 test tally F2
f2:n 51 52
fc6 test tally F6 and p part.
f6:p 101 102
fc7 test tally F7
f7:n 101 102
"""


class TallyUnivHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def _get_results(self):

        # Get results from inputs
        outstr = super()._get_results() + "\n"

        # Process results.h5 using adder_extract_tallies
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_4.csv 4")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_14.csv 14")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_24.csv 24")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_54.csv 54")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_64.csv 64")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_1.csv 1")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_2.csv 2")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_6.csv 6")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_7.csv 7")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_74.csv 74")
        os.system("../../../scripts/adder_extract_tally.py results.h5 tally_84.csv 84")

        tally_l = ["tally_4", "tally_14", "tally_24", "tally_54", "tally_64",
                   "tally_1", "tally_2", "tally_6", "tally_7", "tally_74", "tally_84"]
        # get results string
        for i in range(len(tally_l)):
            if i == 0:
                outstr += tally_l[i] + "\n"
            else:
                outstr += tally_l[i] + "\n"

            with open(tally_l[i] + ".csv", "r") as fin:
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
        assert in_core_checked == 100
        assert storage_checked == 50
        assert supply_checked == 50
        assert non_depleting_checked == 46

def test_tally_univ():
    # This tests the management of tallies
    # "following" universes during depletion.

    # Warning message given when materials with no volumes are shuffled.
    mcnp_text_files = ["2nd_step_shuffling_p2p1.inp", "4th_step_shuffling_p1_p2.inp", "5th_step_shuffling_ps2nd_p4.inp",
                       "6th_step_shuffling_ps2_p4.inp", "case_1_op_1_step_1p.inp", "case_1_op_2_step_1p.inp",
                       "case_3_op_2_step_1p.inp", "case_4_op_3_step_1p.inp", "case_7_op_3_step_1p.inp"]
    test = TallyUnivHarness(mcnp_text_files, "test.h5")
    test._build_inputs()
    test.main()
