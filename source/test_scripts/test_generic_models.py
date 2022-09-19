# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.model.optimizable_structure_model import OptimizableStraightBeam
from source.analysis.analysis_controller import AnalysisController
from source.test_utils.code_structure import ROOT_DIRECTORY, OUTPUT_DIRECTORY, TEST_INPUT_DIRECTORY
from source.test_utils.parsed_results import ParsedResults
from source.test_utils.test_case import TestCase, TestMain

# --- STL Imports ---
import pathlib
import json
import os


class TestGenericModels(TestCase):

    def setUp(self):
        self.cwd = pathlib.Path(os.getcwd()).absolute()
        os.chdir(ROOT_DIRECTORY)


    def tearDown(self):
        os.chdir(self.cwd)


    @TestCase.UniqueReferenceDirectory
    def test_static(self):
        self.RunModel(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingStatic.json")


    @TestCase.UniqueReferenceDirectory
    def test_dynamic(self):
        self.RunModel(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingDynamic.json")


    @TestCase.UniqueReferenceDirectory
    def test_eigen(self):
        self.RunModel(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingEigen.json")


    def RunModel(self, parameters_path: pathlib.Path):
        # Read and format parameters
        with open(parameters_path, 'r') as file:
            parameters = json.loads(file.read())

        for run in parameters["analyses_parameters"]["runs"]:
            if "input" in run and "file_path" in run["input"]:
                run["input"]["file_path"] = str(TEST_INPUT_DIRECTORY / run["input"]["file_path"])

        # Setup
        model_base = StraightBeam(parameters["model_parameters"])
        model = OptimizableStraightBeam(model_base, parameters["optimization_parameters"]["adapt_for_target_values"]).model

        # Run
        controller = AnalysisController(model, parameters["analyses_parameters"])
        controller.solve()
        controller.postprocess()

        # Target directories
        output_directory = OUTPUT_DIRECTORY / parameters["analyses_parameters"]["global_output_folder"]
        reference_directory = self.reference_directory / parameters["analyses_parameters"]["global_output_folder"]

        # Check output directory contents
        for reference_file in reference_directory.glob("*"):
            output_file = output_directory / reference_file.name
            self.assertTrue(output_file.is_file())

        # Check output files
        # TODO set delta
        for output_file in output_directory.glob("*"):
            reference_file = reference_directory / output_file.name
            if output_file.suffix in ParsedResults.GetExtensionParserMap():
                self.CompareFiles(output_file, reference_file)


if __name__ == "__main__":
    TestMain()