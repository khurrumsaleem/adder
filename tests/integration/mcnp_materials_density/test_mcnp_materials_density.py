import pytest
from adder.data import _ATOMIC_MASS
from adder.constants import AVOGADRO
from tests.testing_harness import TestHarness
import h5py


mcnp_lattice_densities = """simple lattice
1 0 -1 fill 10 (0 0 0 1 0  0 0 0 1 0 -1 0 1) imp:n 1
2 10 1.0 -301 302 -303 304 lat=1 u=10 imp:n=1
     fill -2:2 -2:2 0:0
     10 10 10 10 10
     10  1  2(1)  3 10
     10  2(2)  5  6 10
     10  7  8  9 10
     10 10 10 10 10
11 1 2.0 -10 u=1 imp:n=1 vol = 0.528101
21 2 2.0 -10 u=2 imp:n=1 vol=0.528101
31 3 2.0 -10 u=3 imp:n=1 vol=0.528101
41 4 2.0 -10 u=4 imp:n=1 vol=0.528101
51 5 2.0 -10 u=5 imp:n=1 vol=0.528101
61 6 2.0 -10 u=6 imp:n=1 vol=0.528101
71 7 2.0 -10 u=7 imp:n=1 vol=0.528101
81 8 2.0 -10 u=8 imp:n=1 vol=0.528101
91 like 81 but mat=9 u=9
95 like 11 but u=11 mat=11 rho=2.0
97 13 2.0 -10 u=12 imp:n=1 vol=0.528101
10 10 1.0 +10 u=1 imp:n=1 vol=1.059498275
20 like 10 but u=2
30 10 1.0 +10 u=3 imp:n=1 vol=1.059498275
40 10 1.0 +10 u=4 imp:n=1 vol=1.059498275
50 10 -9.9690819326845300 +10 u=5 imp:n=1 vol=1.059498275
60 10 1.0 +10 u=6 imp:n=1 vol=1.059498275
70  7 1.0 +10 u=7 imp:n=1 vol=1.059498275
80 10 1.0 +10 u=8 imp:n=1 vol=1.059498275
90 10 1.0 +10 u=9 imp:n=1 vol=1.059498275
96 12 1.0 +10 u=11 imp:n=1 vol=1.059498275
961 12 -3.0 +10 u=11 imp:n=1 vol=1.059498275
962 12 7.0 +10 u=11 imp:n=1 vol=1.059498275
98 14 1.0 +10 u=12 imp:n=1 vol=1.059498275
981 14 -3.0 +10 u=12 imp:n=1 vol=1.059498275
982 14 7.0 +10 u=12 imp:n=1 vol=1.059498275
5 0 1 imp:n=0

1 cz 3.15
10 cz 0.41
301 px .63
302 px -.63
303 py .63
304 py -.63

m0 nlib=72c
m1 92235 5.0 92238.70c 95. 8016.70c 100. nlib=71c
m2 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m3 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m4 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m5 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m6 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m7 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m8 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m9 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m10 1001.70c 2. 8016 1. nlib=72c
m11 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m12 1001.70c 2. 8016 1. nlib=72c
mt12 lwtr.10t
m13 92235.70c 5.0 92238.70c 95. 8016.70c 100.
m14 1001.70c 2. 8016 1. nlib=72c
mt14 lwtr.10t
m901 92235.70c  0.6 92238.70c  0.2 8016.70c  0.2 nlib=70c $ Ex-core material
m902 92235.70c  0.6 92238.70c  0.2 8016.70c  0.2 nlib=70c $ Ex-core material
c
c Define coordinate transforms
c tr1: 90 deg about x, with explicit m
tr1 0 0 0 1 0  0 0 0 1 0 -1 0 1
c tr2: 90 deg about y, w/o m
tr2 0 0 0 0 0 -1 0 1 0 1  0 0
c
kcode 100 1.0 10 100 2J 6500
ksrc -.63 -.63 0. -.63 0. 0. -.63 .63 0. &
0. -.63 0.   0. 0. 0.   0. .63 0. &
.63 -.63 0.  .63 0. 0.  .63 .63 0.
mode n
DBCN 1
"""


class DensityHarness(TestHarness):

    def execute_test(self):
        """Run ADDER with the appropriate arguments and check the outputs."""
        try:
            self._create_test_lib()
            self._run_adder()
            results = self._get_results()
            self._write_results(results)
        except:
            pass

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_lattice_densities)


def test_mcnp_materials_density_missing_storage():
    """ Tests error message when density is missing for storage material """ 

    # Run test
    try:
        test = DensityHarness([], 'test.h5', 
            input_fname='test_error_storage.add')
        test._build_inputs()
        test.execute_test()
    
        # Check error message: density of storage material not given in ADDER input
        error_msg  = 'ERROR - Storage material 901 was not assigned a density '
        error_msg += 'in the ADDER input file'
        with open('adder.log', 'r') as log:
            for line in log:
                pass
            assert error_msg in line
    finally:
        test._cleanup()


def test_mcnp_materials_density_missing_supply():
    """ Tests error message when density is missing for storage material """ 

    # Run test
    try:
        test = DensityHarness([], 'test.h5', 
            input_fname='test_error_supply.add')
        test._build_inputs()
        test.execute_test()
    
        # Check the error message for the missing density of a supply material
        error_msg  = 'ERROR - Supply material 902 was not assigned a density '
        error_msg += 'in the ADDER input file'
        with open('adder.log', 'r') as log:
            for line in log:
                pass
            assert error_msg in line
    finally: 
        test._cleanup()


def test_mcnp_materials_density():
    """ Tests that all the densities are assigned correctly """ 

    # Set tolerance for testing
    DENSITY_TOL = 5e-16
    
    # Calculate density of material 14_reg_981[1]
    M_H_1 = _ATOMIC_MASS['h1']
    M_16_O = _ATOMIC_MASS['o16']
    DENSITY_M14_2_MASS = -3.
    DENSITY_M14_2 = AVOGADRO*(-DENSITY_M14_2_MASS)/((M_H_1*2 + M_16_O)/3)*1e-24

    # Reference testing dictionary
    material_densities = {                  # Material Origin           | Modified in ADDER input
        '7':                123.456,        # In-core material 7        | Yes
        '7_reg_70':         123.456,        # In-core material 7 copy   | Yes
        '12[3]':            42.,            # Supply universe 11 copy   | Yes
        '12_reg_961[1]':    42.,            # Supply universe 11 copy   | Yes
        '12_reg_962[1]':    42.,            # Supply universe 11 copy   | Yes
        '14[3]':            1.,             # Supply universe 12 copy   | No
        '14_reg_981[1]':    DENSITY_M14_2,  # Supply universe 12 copy   | No
        '14_reg_982[1]':    7.,             # Supply universe 12 copy   | No
    }

    # Run test
    try:
        output_text_files = ["test.inp", "shuffle.inp"]
        test = DensityHarness(output_text_files, 'test.h5')
        test._build_inputs()
        test.execute_test()
        test._compare_mcnp_files()
    
        # Cycle through results file and assert densities are as expected 
        results = h5py.File('results.h5', 'r')
        case_name, op_name, step_name = 'case_1', 'operation_3', 'step_1'

        for material, ref_density in material_densities.items():
            material_data = results[case_name][op_name][step_name][
                'materials/{}'.format(material)]
            density = float(material_data['density'][0])
            
            assert abs(density/ref_density - 1.) < DENSITY_TOL 

    finally: 
        test._cleanup()
