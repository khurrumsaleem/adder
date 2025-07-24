import pytest
import os
import h5py
import numpy as np
from tests import mcnp_2x2x2
from tests import default_config as config
from tests.testing_harness import TestHarness
from adder.depletionlibrary import DepletionLibrary, ReactionData, DecayData, \
    YieldData

mcnp_simple_li6 = """Simple Lithium-6 depletion
c
c Define cells
c
1 1 2.0  30 1 -2 11 -12 21 -22 u=0 imp:n=1 vol=1.0 $ Uranium outer shell
2 2 1.3 -30 1 -2               u=0 imp:n=1 vol=3.0 $ Lithium interior
c
c Define the model periphery
99 0 -1:2:-11:12:-21:22           imp:n=0 $ Exterior

c
c Define surfaces
c
*1  px  0.
*2  px 10.
*11 py  0.
*12 py 10.
*21 pz  0.
*22 pz 10.
30  c/z 5. 5. 1.

c
c Define Materials
c
m0 nlib=70c
m1  92235  1.0
m2   3006  1.0
c
c Set run parameters
c
kcode 1000 1.0 5 35
sdef  x=d1 y=d2 z=d3 erg=2
si1    0 10
sp1    0  1
si2    0 10
sp2    0  1
si3    0 10
sp3    0  1
"""

output_text_files = [
    'case_1_op_2_step_1p.inp',
    'case_1_op_2_step_2e.inp',
]

class LithiumHarness(TestHarness):
    def execute_test(self):
        try:
            self._build_inputs()
            self._create_test_lib()
            self._run_adder()
            self._check_libraries()
            self._check_Li6_depletion()
            results = self._get_results()
            self._write_results(results)
            self._compare_results()
        finally:
            self._cleanup()
    
    def _build_inputs(self):
        with open("mcnp.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_simple_li6)

    def _create_test_lib(self):
        # This will be a simple depletion library, modified with respect
        # to TestHarness to include Li6
        depllib = DepletionLibrary("test", np.array([0., 20.]))

        # H3
        h3dk = DecayData(None, "s", 0.)
        depllib.add_isotope("H3", decay=h3dk)

        # He4
        he4dk = DecayData(None, "s", 0.)
        depllib.add_isotope("He4", decay=he4dk)

        # U235
        u235xs = ReactionData()
        u235xs.add_type("fission", "b", [1.0])
        u235dk = DecayData(None, "s", 200.)
        u235yd = YieldData()
        u235yd.add_isotope("I135", 2. * 0.4)
        u235yd.add_isotope("I134", 2. * 0.6)
        depllib.add_isotope("U235", xs=u235xs, decay=u235dk, nfy=u235yd)

        # I135, stable
        i135dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I135", decay=i135dk)

        # I134, stable
        i134dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I134", decay=i134dk)

        # Li6, stable, initialized with (n,a) reaction which should be
        # converted to (n,t) automatically
        li6xs = ReactionData()
        li6xs.add_type("(n,a)", "b", [14.0], 
                       targets=['H3'], yields_=[1.])
        li6dk = DecayData(None, "s", 0.)

        depllib.add_isotope("Li6", xs=li6xs, decay=li6dk)

        depllib.finalize_library()

        depllib.to_hdf5(self._test_lib_name)

        self._create_ce_data()

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""

        # And perform the rest
        super()._compare_results()

    def _check_libraries(self):
        # This test produces a library before and after depletion. The method
        # Checks that the XS values are as expected for Li-6
        
        # First tolerance: set value, seocond tolerance: based on 6 sigfigs 
        # from MCNP output files
        TOL_XS = [5e-16, 1e-5]

        # Expected value before MCNP is based on value set by _create_test_lib
        # while after MCNP is executed is based on the value provided in the 
        # custom ACE file for 3006.70c 
        xs_expected = [14., 42.] 

        for istate in range(2):
            h5_lib = h5py.File(f'state{istate}_lib.h5' , 'r')
            assert 'Li6' in h5_lib.keys() 
            assert 'Li6' in h5_lib['Li6']['isotopic_data'].keys()
            assert ('(n,t)' in 
                    h5_lib['Li6']['isotopic_data']['Li6']['neutron_xs'].keys())
            xs_lib = \
                h5_lib['Li6']['isotopic_data']['Li6']['neutron_xs']['(n,t)']
            assert set(xs_lib['targets'][:]) == set([b'H3', b'He4'])
            assert set(xs_lib['yields'][:])  == set([1.   , 1.    ])
            assert np.abs(xs_lib['xs'][0]/xs_expected[istate]-1.) < TOL_XS[istate]

    def _check_Li6_depletion(self):
        # Checks the results.h5 file for the final composition of material 2 
        # (depleting Li-6) to make sure that it is correctly depleted. It uses
        # fluxes directly from the results.h5 file

        TOL_N = {
            'origen22': 1e-4,
            'cram':     1e-9,
        }
        
        N_0 = 1.3   #   atoms/b-cm, from MCNP input file
        dt  = 50.   #   days, from MCNP input file

        out_h5 = h5py.File('results.h5', 'r')
        Li6 = out_h5['case_1']['operation_2']['step_1']['materials']['Li6']
        flux = Li6['flux']
        ref_density = 1.3 * np.exp(-42.0 * 1e-24 * flux * dt * 86400)
        test_density = out_h5['case_1']['operation_2']['step_1']['materials']\
            ['Li6']['density']

        # Extract final density from MCNP
        with open('case_1_op_2_step_2e.inp') as mcnp_inp:
            for line in mcnp_inp:
                if line.startswith('2 2 '):
                    test_density = float(line.split()[2])

        if 'origen22' in self.input_fname:
            assert np.abs(test_density/ref_density-1.0) < TOL_N['origen22']
        else:   
            assert np.abs(test_density/ref_density-1.0) < TOL_N['cram']

def test_deplete_xs_convention_cram16():   
    test = LithiumHarness(output_text_files, test_lib_name = "test.h5",
                          input_fname="test_cram16.add")
    test.results_true_fname = 'results_true_cram.dat'
    test.results_error_fname = 'results_error_cram16.dat'
    test.execute_test()

def test_deplete_xs_convention_cram48():
    test = LithiumHarness(output_text_files, test_lib_name = "test.h5",
                          input_fname="test_cram48.add")
    test.results_true_fname = 'results_true_cram.dat'
    test.results_error_fname = 'results_error_cram48.dat'
    test.execute_test()

def test_deplete_xs_convention_origen2():
    test = LithiumHarness(output_text_files, test_lib_name = "test.h5",
                          input_fname="test_origen22.add")
    test.results_true_fname = 'results_true_origen22.dat'
    test.results_error_fname = 'results_error_origen22.dat'
    test.execute_test()

