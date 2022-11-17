# --- STL Imports ---
import unittest
import pathlib
import sys
import shutil
import functools

# --- External Imports ---
import numpy

# --- Internal Imports ---
from .parsed_results import ParsedResults
from .result_comparator import ResultComparator
from .code_structure import TEST_SCRIPTS_DIRECTORY, TEST_REFERENCE_OUTPUT_DIRECTORY


class TestCase(unittest.TestCase):
    """unittest.TestCase with some additional functionality."""
    generate_references = False


    def CompareToReferenceFile(self, test, reference_file_path: pathlib.Path, file_type="", **kwargs):
        """
        Compare input data to referefence results in a file.

        If 'self.generate_references' is True, the reference file will be
        overwritten with the test data.
        """
        if not isinstance(reference_file_path, pathlib.Path):
            reference_file_path = pathlib.Path(reference_file_path)

        self._CheckFiles(reference_file_path)

        if file_type:
            Parser = ParsedResults.GetParserForFileType(file_type)
        else:
            Parser = ParsedResults.GetParserForFileType(reference_file_path.suffix)

        test_parser = Parser.FromData(data=test)
        reference_parser = Parser(reference_file_path)

        self._CompareResults(test_parser, reference_parser, **kwargs)


    def CompareFiles(self, test_file_path: pathlib.Path, reference_file_path: pathlib.Path, file_type="", **kwargs):
        if not isinstance(test_file_path, pathlib.Path):
            test_file_path = pathlib.Path(test_file_path)

        if not isinstance(reference_file_path, pathlib.Path):
            reference_file_path = pathlib.Path(reference_file_path)

        self._CheckFiles(test_file_path, reference_file_path)

        if file_type:
            Parser = ParsedResults.GetParserForFileType(file_type)
        else:
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
            ResultComparator(test, reference, test_case=self, **kwargs).Compare()


    @staticmethod
    def UniqueReferenceDirectory(function):
        """A class method decorator that adds a unique 'reference_directory' to the class in the member function's scope."""
        @functools.wraps(function)
        def WrappedFunction(this, *args, **kwargs):
            tmp = getattr(this, "reference_directory") if hasattr(this, "reference_directory") else None
            this.reference_directory = TEST_REFERENCE_OUTPUT_DIRECTORY / type(this).__name__ / function.__name__
            output = function(this, *args, **kwargs)
            this.reference_directory = tmp
            return output
        return WrappedFunction


    def _CheckFiles(self, *paths):
        """Check whether input paths point to existing files, or create empty ones if 'self.generate_references' is set."""
        for path in paths:
            if not isinstance(path, pathlib.Path):
                path = pathlib.Path(path)

            if not path.is_file(): # Path does not exist

                if path.is_dir():
                    raise FileExistsError("{} is a directory".format(path))

                if self.generate_references: # Copy default file
                    extension = path.suffix
                    if extension in ParsedResults.GetExtensionParserMap():
                        default_file_path = TEST_SCRIPTS_DIRECTORY / "misc" / ("default" + extension)
                        path.absolute().parent.mkdir(exist_ok=True, parents=True)
                        shutil.copyfile(default_file_path, path)
                    else:
                        raise ValueError("Unsupported file type: {}".format(extension))

                else: # file not found and it shouldn't be created
                    raise FileNotFoundError(str(path))


    def assertArrayAlmostEqual(self, array, reference, **kwargs):
        for item, item_reference in zip(array, reference):
            self.assertAlmostEqual(item, item_reference, **kwargs)


    def assertMatrixAlmostEqual(self, matrix: numpy.ndarray, reference: numpy.ndarray, **kwargs):
        for row, row_reference in zip(matrix, reference):
            if isinstance(row, (list, tuple, numpy.ndarray)):
                self.assertArrayAlmostEqual(row, row_reference, **kwargs)
            else:
                self.assertAlmostEqual(row, row_reference, **kwargs)


    def assertDictAlmostEqual(self, map: dict, reference: dict, **kwargs):
        # Check keys
        for key in map:
            self.assertIn(key, reference, f"Key '{key}' in test dict is not in the reference dict")

        for key in reference:
            self.assertIn(key, map, f"Key '{key}' in the reference dict is not in the test dict")

        # Check values
        msg = ""
        if msg in kwargs:
            msg = kwargs.get("msg", "")
            del kwargs["msg"]
        for key in map:
            self.assertAlmostEqual(map[key], reference[key], msg=f"Failure for key '{key}': {msg}", **kwargs)


def TestMain():
    if "--generate-references" in sys.argv:
        sys.argv.remove("--generate-references")
        TestCase.generate_references = True

    unittest.main()
