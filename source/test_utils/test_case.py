# --- STL Imports ---
import unittest
import pathlib
import sys

# --- External Imports ---
import numpy

# --- Internal Imports ---
from .parsed_results import ParsedResults
from .result_comparator import ResultComparator


class TestCase(unittest.TestCase):
    """unittest.TestCase with some additional functionality."""
    generate_references = False


    def CompareToReferenceFile(self, test, reference_file_path: pathlib.Path, **kwargs):
        """Generates 'ParsedResults' objects from input data and file path"""
        if not isinstance(reference_file_path, pathlib.Path):
            reference_file_path = pathlib.Path(reference_file_path)

        Parser = ParsedResults.GetParserForFileType(reference_file_path.suffix)
        test_parser = Parser.FromData(data=test)
        reference_parser = Parser(reference_file_path)

        self._CompareResults(test_parser, reference_parser, **kwargs)


    def CompareFiles(self, test_file_path: pathlib.Path, reference_file_path: pathlib.Path, **kwargs):
        if not isinstance(test_file_path, pathlib.Path):
            test_file_path = pathlib.Path(test_file_path)

        if not isinstance(reference_file_path, pathlib.Path):
            reference_file_path = pathlib.Path(reference_file_path)

        Parser = ParsedResults.GetParserForFileType(reference_file_path.suffix)
        test_parser = Parser(test_file_path)
        reference_parser = Parser(reference_file_path)

        self._CompareResults(test_parser, reference_parser, **kwargs)


    def _CompareResults(self, test: ParsedResults, reference: ParsedResults, **kwargs):
        if self.generate_references:
            # Create directory
            reference.file_path.absolute().parent.mkdir(exist_ok=True, parents=True)

            # Write to file
            if reference.file_path.is_file():
                print("Overwriting {}".format(reference.file_path))
            with open(reference.file_path, 'w') as file:
                file.write(str(test))
        else:
            ResultComparator(test, reference, test_case=self, **kwargs).CompareContents()


    def assertArrayAlmostEqual(self, array, reference, **kwargs):
        for item, item_reference in zip(array, reference):
            self.assertAlmostEqual(item, item_reference, **kwargs)


    def assertMatrixAlmostEqual(self, matrix: numpy.ndarray, reference: numpy.ndarray, **kwargs):
        for row, row_reference in zip(matrix, reference):
            if isinstance(row, (list, tuple, numpy.ndarray)):
                self.assertArrayAlmostEqual(row, row_reference, **kwargs)
            else:
                self.assertAlmostEqual(row, row_reference, **kwargs)


def TestMain():
    if "--generate-references" in sys.argv:
        sys.argv.remove("--generate-references")
        TestCase.generate_references = True

    unittest.main()