# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

from source.test_utils.test_case import TestCase
from source.test_utils.code_structure import OUTPUT_DIRECTORY, TEST_ANALYTICAL_REFERENCE_RESULTS_DIRECTORY
from source.test_utils.parsed_results import ParsedResults

# --- STL Imports ---
import unittest


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


    @property
    def parameters(self):
        return {
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
                        "number_of_elements": 10,
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
            },
            "analyses_parameters":{
                "global_output_folder" : "some/path",
                "model_properties": {
                    "write": False,
                    "plot": False
                },
                "report_options": {
                    "combine_plots_into_pdf" : False,
                    "display_plots_on_screen" : False,
                    "use_skin_model" : False
                },
                "runs": [{
                        "type": "eigenvalue_analysis",
                        "settings": {
                            "normalization": "mass_normalized"},
                        "input":{},
                        "output":{
                            "eigenmode_summary": {
                                "write" : False, 
                                "plot" : False},
                            "eigenmode_identification": {
                                "write" : True, 
                                "plot" : False},
                            "selected_eigenmode": {
                                "plot_mode": [], 
                                "write_mode": [],
                                "animate_mode": [],
                                "animate_skin_model": []},
                            "selected_eigenmode_range": {
                                "help": "maximum 4 modes per range at a time",
                                "considered_ranges": [[1,2]], 
                                "plot_range": [False, False], 
                                "write_range": [False, False]}
                            }
                    
                    }]
            }
        }


    def runModel(self, model_name: str, boundary_conditions: str):
        # ==============================================
        # Set BC and name
        parameters = self.parameters
        parameters['model_parameters']['name']=model_name
        parameters['model_parameters']['boundary_conditions']=boundary_conditions

        # create initial model
        beam_model = StraightBeam(parameters['model_parameters'])

        # ==============================================
        # Analysis wrapper

        analyses_controller = AnalysisController(
            beam_model, parameters['analyses_parameters'])
        analyses_controller.solve()
        analyses_controller.postprocess()

        # ==============================================
        # test results against available analytical solutions
        abs_tol = 1e-1

        # Parse result and reference files
        result_file_path = OUTPUT_DIRECTORY / model_name / "eigenvalue_analysis_eigenmode_identification.dat"
        reference_file_path = TEST_ANALYTICAL_REFERENCE_RESULTS_DIRECTORY / (model_name + ".txt")
        
        self.assertTrue(result_file_path.is_file())
        self.assertTrue(reference_file_path.is_file())

        result = ParsedResults(result_file_path).AsDictionary()
        reference = ParsedResults(reference_file_path).AsDictionary()

        # Check result data
        self.assertTrue("TypeCounter" in result)
        self.assertTrue("Eigenfrequency [Hz]" in result)
        self.assertTrue("Type" in result)

        # Check reference data
        self.assertTrue("TypeCounter" in reference)
        self.assertTrue("Eigenfrequency [Hz]" in reference)
        self.assertTrue("Type" in reference)

        # Checks
        for ref_type_counter, ref_frequency, ref_typ in zip(reference["TypeCounter"], reference["Eigenfrequency [Hz]"], reference["Type"]):
            for type_counter, frequency, typ in zip(result["TypeCounter"], result["Eigenfrequency [Hz]"], result["Type"]):
                if type_counter == ref_type_counter and typ == ref_typ:
                    self.assertAlmostEqual(
                        frequency,
                        ref_frequency,
                        delta = abs_tol
                    )


if __name__ == "__main__":
    unittest.main()