# Parametric optimizable beam

A project focusing on a finite-element method formulation-based straight 3D parametric optimizable Timoshenko beam

Motivated but not limited to usage in education as well as for research purposes

With effort made to be compatible/usable with the interface of [Kratos Multiphisics](https://github.com/KratosMultiphysics)

The project is free under the BSD 3-Clause License

# Features available
## Beam types:
* Bernoulli beam
* Timoshenko beam
* Co-rotational beam (WIP -> linked to geometric nonlinear analysis)

## Boundary conditions
* fixed-fixed
* pinned-pinned
* fixed-pinned
* pinned-fixed
* fixed-free
* free-fixed

## Analysis
* eigenvalue analysis
* dynamic analysis
* static analysis

## Output
* output file of selected degree of freedoms at selected time
* 1D animation of dynamic analysis results for selected time
* mapped 3D animation of dynamic analysis results for selected time

## Tests
Test scripts are in *source/test_scripts* and can be executed from [run_all_tests.py](https://github.com/mpentek/ParOptBeam/blob/1297a2ab907b66a8bdd3eb5f59a0cb202b55049b/run_all_tests.py) which discovers and runs all of them. If you wish to run individual tests, you'll have to navigate a terminal to the project's source directory, and run them as modules like so:
```
python -m source.test_scripts.test_beam_eigenvalue_analytic
```
Each test script uses python's [`unittest`](https://docs.python.org/3/library/unittest.html) framework, so you can pass any argument that `unittest` recognises.

Some tests store their references in text files (stored in *source/test_scripts/reference_output*) and compare their results against them. These text files can be regenerated by passing `--generate-references` to the test script you wish to update references for.

If you wish to run a specific test case using a debugger, you'll have to write a script that imports and executes the desired case, and run that in the debugger. An example of such a script:
```py
import source.test_scripts.test_beam_eigenvalue_analytic
from source.test_utils.test_case import TestMain
case = source.test_scripts.test_beam_eigenvalue_analytic.BeamEigenvalueAnalyticalTest
TestMain()
```
