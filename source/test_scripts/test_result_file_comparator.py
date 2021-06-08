# --- Internal Imports ---
from source.test_utils.code_structure import TEST_SCRIPTS_DIRECTORY
from source.test_utils.result_file_comparator import ResultFileComparator

# --- STL Imports ---
import unittest


class TestResultFileComparator(unittest.TestCase):

    def test_DefaultAnalysisWithIdenticalFiles(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_test_file.txt"
        reference_file_path = test_file_path

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            testCase = self
        ).Compare()


    def test_DefaultAnalysis(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_test_file.txt"
        reference_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_reference_file.txt"

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            testCase = self,
            settings = {"absolute_tolerance" : 1e-3}
        ).Compare()


    def test_EigenmodeIdentificationAnalysisWithIdenticalFiles(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_test_file.txt"
        reference_file_path = test_file_path

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            testCase = self,
            settings = {
                "analysis_type" : "eigenmode_identification"
            }
        ).Compare()


    def test_EigenmodeIdentificationAnalysis(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_test_file.txt"
        reference_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_reference_file.txt"

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            testCase = self,
            settings = {
                "analysis_type" : "eigenmode_identification",
                "absolute_tolerance" : 1e-3
            }
        ).Compare()




if __name__ == "__main__":
    unittest.main()