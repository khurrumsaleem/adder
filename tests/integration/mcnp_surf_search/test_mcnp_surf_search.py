import filecmp
import pytest
import os
import numpy as np
import h5py
from tests.testing_harness import TestHarness
from adder.mcnp.constants import CI_95

inp_file = """Cylinder rod test case
1 1 4e-3 -1  10 -11 imp:n=1    $ homogenized fuel
2 0      #1         imp:n=0    $ problem exterior

c Define surfaces
1 cz 4
10 pz 0.    $ rx bottom
11 pz 5.    $ A range of 5 to 25 is suitable. as the crit pos is ~14cm

c define material
m1 92235.70c 1.0 nlib=70c
kcode 30000 1.0 20 100
ksrc 0. 0. 1. 1. 1. 1. 1. -1. 1. -1. 1. 1. -1. -1. 1.
"""


class GeomSweepHarness(TestHarness):
    REF_RANGES = [(8.58, 8.84), (2.14, 2.22), (2.03, 2.35), (-2.12, -1.80), (0, 1), (0, 1)]

    k_targets = [1.0, 0.9, 0.95, 0.925, 0.822, 0.822]
    target_intervals = [0.001, 0.001, 0.001, 0.001, 0.005, 0.015]
    REF_TARGET_RANGES = []

    for target, interval in zip(k_targets, target_intervals):
        REF_TARGET_RANGES.append([target - interval, target + interval])

    @property
    def REF_INITIALS(self):
        return [self.displacements[0], 2.0, 0.0]

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def update_results(self):
        pass

    def _get_results(self):
        """Digest info in the output and return as array of the 2 positions."""
        output, initial_positions, keffs, keff_stddevs = self._get_outputs()
        self.displacements = output
        self.initial_positions = initial_positions
        self.keffs = keffs
        self.keff_stddevs = keff_stddevs

    def _write_results(self, results_string):
        pass

    def _overwrite_results(self):
        pass

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""

        assert len(self.REF_RANGES) == len(self.displacements)
        for i in range(len(self.REF_RANGES)):
            assert self.REF_RANGES[i][0] <= self.displacements[i] <= self.REF_RANGES[i][1]
        assert np.array_equal(np.round(np.array(self.REF_INITIALS), 3), 
                              np.round(np.array(self.initial_positions), 3))

        for i in range(len(self.REF_TARGET_RANGES)):

            keff = self.keffs[i]
            keff_std = self.keff_stddevs[i]
            keff_95lo = keff - (CI_95 * keff_std)
            keff_95hi = keff + (CI_95 * keff_std)

            assert keff_95lo >= self.REF_TARGET_RANGES[i][0]
            assert keff_95hi <= self.REF_TARGET_RANGES[i][1]

    def _get_outputs(self):
        # Get the HDF5 file and obtain the control group displacement in time
        h5_group_names = ["case_1/operation_1/step_1/",
                          "case_2/operation_1/step_1/",
                          "case_3/operation_1/step_1/",
                          "case_4/operation_2/step_1/",
                          "case_5/operation_1/step_1/",
                          "case_6/operation_1/step_1/"]

        displacements, keffs, keff_stddevs = [], [], []
        with h5py.File("results.h5", "r") as h5:
            for group_name in h5_group_names:
                grp = h5[group_name + "control_groups/bank_1/"]
                displacements.append(float(grp.attrs["displacement"]))

                keffs.append(h5[group_name].attrs["keff"])
                keff_stddevs.append(h5[group_name].attrs["keff_stddev"])

        initial_positions = []
        with open("adder.log", "r") as log:
            found = False
            for line in log:
                if ' as initial guess' in line:
                    found = True
                elif found and ('Iteration 1' in line):
                    initial_positions.append(float(line.split('Position')[-1]))
                    found = False

        return displacements, initial_positions, keffs, keff_stddevs


def test_mcnp_surf_search():
    output_text_files = []
    test = GeomSweepHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()


