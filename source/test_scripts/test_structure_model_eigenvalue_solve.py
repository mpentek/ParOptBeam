# --- Internal Imports ---
from source.test_utils.code_structure import TEST_INPUT_DIRECTORY
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain

# --- STL Imports ---
import json


class test_structure_model_eigenvalue_solve (TestCase):

    @staticmethod
    def StraightBeamFactory() -> StraightBeam:
        with open(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingStatic.json", "r") as config_file:
            parameters = json.load(config_file)
        return StraightBeam(parameters["model_parameters"])

    @TestCase.UniqueReferenceDirectory
    def test_values (self):
        beam = self.StraightBeamFactory()
        # example values
        beam.comp_k = [[ 6.72000000e+08,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  4.30133760e+04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -5.37667200e+05],
        [ 0.00000000e+00,  0.00000000e+00,  1.72037376e+05,
        0.00000000e+00,  2.15046720e+06,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.36750769e+05,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  2.15046720e+06,
        0.00000000e+00,  3.58411200e+07,  0.00000000e+00],
        [ 0.00000000e+00, -5.37667200e+05,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  8.96112000e+06]]

        beam.comp_m = [[5.23333333e+03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.83152906e+03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -2.05597332e+04],
        [ 0.00000000e+00,  0.00000000e+00,  5.83183050e+03,
        0.00000000e+00,  2.05603612e+04,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        8.72265833e+01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  2.05603612e+04,
        0.00000000e+00,  9.34802929e+04,  0.00000000e+00],
        [ 0.00000000e+00, -2.05597332e+04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.34593596e+04]]

        # run the tested method
        StraightBeam.eigenvalue_solve(beam)

        self.CompareToReferenceFile(beam.eig_freqs,
                                    self.reference_directory / "eig_freqs.csv",
                                    delta = 1e-12)


if __name__ == "__main__":
    TestMain()
