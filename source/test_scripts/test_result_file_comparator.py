# --- Internal Imports ---
from source.test_utils.code_structure import TEST_SCRIPTS_DIRECTORY
from source.test_utils.result_comparator import ResultFileComparator
from source.test_utils.test_case import TestCase, TestMain


class TestResultFileComparator(TestCase):

    def test_DefaultAnalysisWithIdenticalFiles(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_test_file.dat"
        reference_file_path = test_file_path

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            test_case = self
        ).Compare()


    def test_DefaultAnalysis(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_test_file.dat"
        reference_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "default_reference_file.dat"

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            test_case = self,
            delta=1e-3
        ).Compare()


    def test_EigenmodeIdentificationAnalysisWithIdenticalFiles(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_test_file.dat"
        reference_file_path = test_file_path

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            test_case = self,
            analysis_type = "eigenmode_identification"
        ).Compare()


    def test_EigenmodeIdentificationAnalysis(self):
        test_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_test_file.dat"
        reference_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / "eigenmode_identification_reference_file.dat"

        ResultFileComparator(
            test_file_path,
            reference_file_path,
            test_case = self,
            analysis_type="eigenmode_identification",
            delta=1e-3
        ).Compare()




if __name__ == "__main__":
    TestMain()