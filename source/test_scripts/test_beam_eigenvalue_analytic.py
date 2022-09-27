# --- External Imports ---
import numpy

# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

from source.test_utils.code_structure import OUTPUT_DIRECTORY, TEST_REFERENCE_OUTPUT_DIRECTORY
from source.test_utils.test_case import TestCase, TestMain
from source.test_utils.analytical_solutions import EulerBernoulli, TorsionalBeam

# --- STL Imports ---
import math


beam_eigenvalue_analytic_parameters = {
    "model_parameters": {
        "name": "name",
        "domain_size": "3D",
        "system_parameters": {
            "element_params": {
                "type": "Bernoulli",
                "is_nonlinear": False
            },
            "material": {
                "is_nonlinear": False,
                "density": 7850.0,
                "youngs_modulus": 2.10e11,
                "poisson_ratio": 0.3,
                "damping_ratio": 0.0
            },
            "geometry": {
                "length_x": 25,
                "number_of_elements": 100,
                "defined_on_intervals": [{
                    "interval_bounds" : [0.0,"End"],
                    "length_y": [0.20],
                    "length_z": [0.40],
                    "area"    : [0.08],
                    "shear_area_y" : [0.0667],
                    "shear_area_z" : [0.0667],
                    "moment_of_inertia_y" : [0.0010667],
                    "moment_of_inertia_z" : [0.0002667],
                    "torsional_moment_of_inertia" : [0.00007328]}]
            }
        },
        "boundary_conditions": "bc"
    }
}


class BeamEigenvalueAnalyticalTest(TestCase):

    def test_fixed_fixed(self):
        self.runModel("FixedFixedTest", "fixed-fixed")

    def test_free_fixed(self):
        self.runModel("FreeFixedTest", "free-fixed")

    def test_fixed_free(self):
        self.runModel("FixedFreeTest", "fixed-free")

    def test_fixed_pinned(self):
        self.runModel("FixedPinnedTest", "fixed-pinned")

    def test_pinned_fixed(self):
        self.runModel("PinnedFixedTest", "pinned-fixed")

    def test_pinned_pinned(self):
        self.runModel("PinnedPinnedTest", "pinned-pinned")

    def runModel(self, model_name: str, boundary_conditions: str) -> StraightBeam:
        """Run analysis and check values that have analytical solutions for any boundary pair."""
        # ==============================================
        # Set BC and name
        parameters = beam_eigenvalue_analytic_parameters
        parameters['model_parameters']['name']=model_name
        parameters['model_parameters']['boundary_conditions']=boundary_conditions

        # create initial model
        beam_model = StraightBeam(parameters['model_parameters'])
        beam_model.eigenvalue_solve()

        # Mode identification is not checked in this test,
        # so there's no way of knowing which frequency belongs to what mode.
        # => check whether there's a frequency sufficiently close to the
        # analytical eigenfrequency (angular frequencies).
        boundaries = [substring.strip() for substring in boundary_conditions.split("-")]
        youngs_modulus = parameters["model_parameters"]["system_parameters"]["material"]["youngs_modulus"]
        section_density = parameters["model_parameters"]["system_parameters"]["material"]["density"] * parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["area"][0]
        length = parameters["model_parameters"]["system_parameters"]["geometry"]["length_x"]

        base_relative_tolerance_order = -3

        # sway_z
        analytical_beam = EulerBernoulli(stiffness = youngs_modulus,
                                         section_density = section_density,
                                         length = length,
                                         moment_of_inertia = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_z"][0])

        for index, eigenfrequency in enumerate(analytical_beam.GetEigenfrequencies(boundaries,
                                                                                   frequency_seeds = numpy.linspace(1e-3, 1e2, int(1e2)),
                                                                                   rtol = 1e-12)):
            # Assume the accuracy loses an order of magnitude for each consequent eigenfrequency
            relative_tolerance = 10**(base_relative_tolerance_order + index)
            self.AssertInWithTolerance(eigenfrequency, beam_model.eig_values, relative_tolerance = relative_tolerance)

        # sway_y
        analytical_beam = EulerBernoulli(stiffness = youngs_modulus,
                                         section_density = section_density,
                                         length = length,
                                         moment_of_inertia = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_y"][0])

        for index, eigenfrequency in enumerate(analytical_beam.GetEigenfrequencies(boundaries,
                                                                                   frequency_seeds = numpy.linspace(1e-3, 1e2, int(1e2)),
                                                                                   rtol = 1e-12)):
            # Assume the accuracy loses an order of magnitude for each consequent eigenfrequency
            relative_tolerance = 10**(base_relative_tolerance_order + index)
            self.AssertInWithTolerance(eigenfrequency, beam_model.eig_values, relative_tolerance = relative_tolerance)

        # torsion
        # TODO: find analytical solution for other boundary conditions (free-free is not relevant for bending anyway)
        if boundary_conditions == "free-free":
            material_coefficient = math.sqrt(youngs_modulus / 2.0 / (1.0 + parameters["model_parameters"]["system_parameters"]["material"]["poisson_ratio"])
                                            / parameters["model_parameters"]["system_parameters"]["material"]["density"])
            width = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["length_y"][0]
            height = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["length_z"][0]
            geometry_coefficient = math.sqrt(width * (height**3) / 16.0 * (16.0 / 3.0) - 3.36 * height / width * (1.0 - (height / width)**4 / 12.0)
                                            / parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["torsional_moment_of_inertia"][0]) * math.pi / length
            for index, torsional_index in enumerate(range(1, 3)):
                eigenfrequency = torsional_index * material_coefficient * geometry_coefficient

                # Assume the accuracy loses an order of magnitude for each consequent eigenfrequency
                relative_tolerance = 10**(base_relative_tolerance_order + index)
                self.AssertInWithTolerance(eigenfrequency, beam_model.eig_values, relative_tolerance = relative_tolerance)

        return beam_model

    def AssertInWithTolerance(self, value: float, array: list[float], relative_tolerance: float = 1e-16) -> None:
        def GetRelativeError(test: float, reference: float):
            return abs(test) if reference == 0 else abs((reference - test) / reference)
        self.assertAlmostEqual(min(GetRelativeError(value, item) for item in array if not math.isnan(item)),
                                0,
                                delta = relative_tolerance,
                                msg = f"(value: {value})")


class BeamDecomposeEigenmodesAnalyticalTest(TestCase):

    def test_fixed_fixed(self):
        beam_model = self.runModel("FixedFixedTest", "fixed-fixed")


    def test_free_fixed(self):
        beam_model = self.runModel("FreeFixedTest", "free-fixed")

    def test_fixed_free(self):
        beam_model = self.runModel("FixedFreeTest", "fixed-free")

    def test_fixed_pinned(self):
        beam_model = self.runModel("FixedPinnedTest", "fixed-pinned")

    def test_pinned_fixed(self):
        beam_model = self.runModel("PinnedFixedTest", "pinned-fixed")

    def test_pinned_pinned(self):
        beam_model = self.runModel("PinnedPinnedTest", "pinned-pinned")

    def runModel(self, model_name: str, boundary_conditions: str) -> StraightBeam:
        # ==============================================
        # Set BC and name
        parameters = beam_eigenvalue_analytic_parameters
        parameters['model_parameters']['name']=model_name
        parameters['model_parameters']['boundary_conditions']=boundary_conditions

        # create initial model
        beam_model = StraightBeam(parameters['model_parameters'])
        beam_model.decompose_and_quantify_eigenmodes()

        boundaries = [substring.strip() for substring in boundary_conditions.split("-")]
        youngs_modulus = parameters["model_parameters"]["system_parameters"]["material"]["youngs_modulus"]
        section_density = parameters["model_parameters"]["system_parameters"]["material"]["density"] * parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["area"][0]
        length = parameters["model_parameters"]["system_parameters"]["geometry"]["length_x"]

        # sway_z
        with self.subTest("sway_z"):
            analytical_beam = EulerBernoulli(stiffness = youngs_modulus,
                                            section_density = section_density,
                                            length = length,
                                            moment_of_inertia = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_z"][0])
            eigenfrequencies = analytical_beam.GetEigenfrequencies(boundaries,
                                                                   frequency_seeds = numpy.linspace(1e-3, 1e2, int(1e2)),
                                                                   rtol = 1e-12)

            self.assertIn("sway_z", beam_model.mode_identification_results)
            for mode_index, reference_frequency in enumerate(eigenfrequencies):
                mode_info = beam_model.mode_identification_results["sway_z"][mode_index]
                radial_frequency = 2 * math.pi * beam_model.eig_freqs[mode_info["mode_id"] - 1]
                tolerance = reference_frequency * 10**(self.__base_relative_tolerance_order + mode_index)

                message_on_fail = f"\ntest eigenfrequencies: {beam_model.eig_values[:len(eigenfrequencies)]}\nreference eigenfrequencies: {eigenfrequencies}"

                self.assertAlmostEqual(radial_frequency, reference_frequency, delta = tolerance, msg = message_on_fail)

                # Effective modal mass and modal participation factor
                effective_modal_mass, modal_participation_factor = analytical_beam.GetModalProperties(reference_frequency, boundaries)

                tolerance = modal_participation_factor * 10**(self.__base_relative_tolerance_order + mode_index)
                self.assertAlmostEqual(mode_info["rel_participation"], modal_participation_factor, delta = tolerance, msg = message_on_fail)

                tolerance = effective_modal_mass * 10**(self.__base_relative_tolerance_order + mode_index)
                self.assertAlmostEqual(mode_info["eff_modal_mass"], effective_modal_mass, delta = tolerance, msg = message_on_fail)

        # sway_y
        with self.subTest("sway_y"):
            analytical_beam = EulerBernoulli(stiffness = youngs_modulus,
                                            section_density = section_density,
                                            length = length,
                                            moment_of_inertia = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_y"][0])
            eigenfrequencies = analytical_beam.GetEigenfrequencies(boundaries,
                                                                   frequency_seeds = numpy.linspace(1e-3, 1e2, int(1e2)),
                                                                   rtol = 1e-12)

            self.assertIn("sway_y", beam_model.mode_identification_results)
            for mode_index, reference_frequency in enumerate(eigenfrequencies):
                mode_info = beam_model.mode_identification_results["sway_y"][mode_index]
                radial_frequency = 2 * math.pi * beam_model.eig_freqs[mode_info["mode_id"] - 1]
                tolerance = reference_frequency * 10**(self.__base_relative_tolerance_order + mode_index)
                self.assertAlmostEqual(radial_frequency, reference_frequency, delta = tolerance, msg = f"\ntest: {beam_model.eig_values[:len(eigenfrequencies)]}\nreference: {eigenfrequencies}")

                message_on_fail = f"\ntest eigenfrequencies: {beam_model.eig_values[:len(eigenfrequencies)]}\nreference eigenfrequencies: {eigenfrequencies}"

                # Effective modal mass and modal participation factor
                effective_modal_mass, modal_participation_factor = analytical_beam.GetModalProperties(reference_frequency, boundaries)

                tolerance = modal_participation_factor * 10**(self.__base_relative_tolerance_order + mode_index)
                self.assertAlmostEqual(mode_info["rel_participation"], modal_participation_factor, delta = tolerance, msg = message_on_fail)

                tolerance = effective_modal_mass * 10**(self.__base_relative_tolerance_order + mode_index)
                self.assertAlmostEqual(mode_info["eff_modal_mass"], effective_modal_mass, delta = tolerance, msg = message_on_fail)

        # torsional
        with self.subTest("torsion"):
            if boundary_conditions == "free-free":
                analytical_beam = TorsionalBeam(stiffness = youngs_modulus,
                                                poisson_ratio = parameters["model_parameters"]["system_parameters"]["material"]["poisson_ratio"],
                                                density = parameters["model_parameters"]["system_parameters"]["material"]["density"],
                                                length = length,
                                                moment_of_inertia_y = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_y"][0],
                                                moment_of_inertia_z = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["moment_of_inertia_z"][0],
                                                torsional_moment_of_inertia = parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"][0]["torsional_moment_of_inertia"][0])

                self.assertIn("torsional", beam_model.mode_identification_results)
                for mode_index, reference_frequency in enumerate(analytical_beam.GetEigenfrequencies(boundaries, 3)): # first 3 torsional modes
                    mode_info = beam_model.mode_identification_results["torsional"][mode_index]
                    radial_frequency = 2 * math.pi * beam_model.eig_freqs[mode_info["mode_id"] - 1]
                    tolerance = reference_frequency * 10**(self.__base_relative_tolerance_order + mode_index)
                    self.assertAlmostEqual(radial_frequency, reference_frequency, delta = tolerance)

                    # Effective modal mass and modal participation factor
                    effective_modal_mass, modal_participation_factor = analytical_beam.GetModalProperties(reference_frequency, boundaries)

                    tolerance = modal_participation_factor * 10**(self.__base_relative_tolerance_order + mode_index)
                    self.assertAlmostEqual(mode_info["rel_participation"], modal_participation_factor, delta = tolerance)

                    tolerance = effective_modal_mass * 10**(self.__base_relative_tolerance_order + mode_index)
                    self.assertAlmostEqual(mode_info["eff_modal_mass"], effective_modal_mass, delta = tolerance)

        return beam_model

    @property
    def __base_relative_tolerance_order(self):
        return -3


if __name__ == "__main__":
    TestMain()
