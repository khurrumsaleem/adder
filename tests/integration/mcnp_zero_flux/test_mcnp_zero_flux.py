import pytest
import os
import h5py
import numpy as np
from tests.testing_harness import TestHarness

inp_file ="""Zero flux
c CELLS
c Fuel Pin #1
101    1  -11.0 -101  imp:n=1 u=1
102    1  -11.0  101 -149  imp:n=1 u=1
150    1  -11.0  149  imp:n=1 u=1
c Fuel Pin #2
201    1  -11.0 -101  imp:n=1 u=2
202    1  -11.0  101 -149  imp:n=1 u=2
250    1  -11.0  149  imp:n=1 u=2
c Fuel Pin #3
301    1  -11.0 -101  imp:n=1 u=3
302    1  -11.0  101 -149  imp:n=1 u=3
350    1  -11.0  149  imp:n=1 u=3
c Fuel Pin #4
401    1  -11.0 -101  imp:n=1 u=4
402    1  -11.0  101 -149  imp:n=1 u=4
450    1  -11.0  149  imp:n=1 u=4
c Fuel Pin #5
501    1  -11.0 -101  imp:n=1 u=5
502    1  -11.0  101 -149  imp:n=1 u=5
550    1  -11.0  149  imp:n=1 u=5
c Fuel pins
1    0  -51  31 -32  imp:n=1 *fill=1 (1) 
2    0  -52  31 -32  imp:n=1 *fill=2 (2)
3    0  -53  31 -32  imp:n=1 *fill=3 (3)
4    0  -54  31 -32  imp:n=1 *fill=4 (4)
6    0  -55  31 -32  imp:n=1 *fill=5 (5)
c Water
1001   2  -1.0   11 -12  21 -22  31 -32  51  52  53  54  55   imp:n=1  $ Water in between pins
1002   2  -1.0   11 -12  21 -22 -31  33                       imp:n=1  $ Water below fuel pins
1003   2  -1.0   11 -12  21 -22  32 -34                       imp:n=1  $ Water above fuel pins
c Outer world
9999 0  -11  12 -21  22 -33  34  imp:n=0

c SURFACES
*11  px  -1.3
*12  px   1.3
*21  py  -1.3
*22  py   201.3
c *22  py   1.3
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
55   5 cz 0.5
61   cz   0.4
c Fuel pin axial division
101   pz  -48.000
102   pz  -46.000
148   pz  46.000
149   pz  48.000

c MATERIALS
m0 nlib=70c
c UO2 fuel
m1 92235.70c -0.0264 92238.70c -0.8536  8016.70c -0.12
c Water
m2  1001.70c -0.1111  8016.70c -0.8889
c TRANSFORMATIONS
tr1  0.65  0.65 0.0
tr2 -0.65  0.65 0.0
tr3 -0.65 -0.65 0.0
tr4  0.65 -0.65 0.0
tr5  0.00 195 0.0
c CALCULATION PARAMETERS
kcode 100 1.0 20 1000
ksrc 
        0.650    0.650  -49.000
        0.650    0.650  -47.000
       -0.650   -0.650   47.000
       -0.650   -0.650   49.000
c flux in cell FE-4.
f4:n 502
"""


class ZeroFluxHarness(TestHarness):
    def _build_inputs(self):
        with open("test_mcnp_zero_flux.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)


def test_zero_flux():
    # This tests the management of zero flux tallied in fuel
    # material, where cross-sections are set to zero.

    # Warning message given when materials with no volumes are shuffled.
    output_text_files = []
    test = ZeroFluxHarness(output_text_files, "test.h5")
    test._build_inputs()
    test._create_test_lib()
    test._run_adder()
    # check info and log messages
    info_message = "flux-weighted cross-sections set to 0"
    warning_message = "Zero flux in material"
    test.log_messages = [('info', info_message, 3)]
    test._check_log_messages()
    test.log_messages = [('warning', warning_message, 3)]
    test._check_log_messages()
    test._cleanup()


