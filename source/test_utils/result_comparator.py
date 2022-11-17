# --- STL Imports ---
import unittest
import pathlib
from unittest import TestCase

# --- External Imports ---
import numpy

# --- Internal Imports ---
from .parsed_results import ParsedResults


class ResultComparator:

    def __init__(self, test_results: ParsedResults, reference_results: ParsedResults, test_case=None, **settings):
        if not all(isinstance(item, ParsedResults) for item in (test_results, reference_results)):
            raise TypeError("Expecting 'ParsedResults' objects, but got {}".format((type(test_results), type(reference_results))))

        self.test_results = test_results
        self.reference_results = reference_results

        # Settings
        settings = self.ValidateAndAddDefaults(settings.copy())
        self.delta = settings["delta"]
        self.allow_additional_data = settings["allow_additional_data"]
        self.analysis_type = settings["analysis_type"]

        if test_case == None:
            self.test_case = TestCase()
        else:
            self.test_case = test_case


    def Compare(self):
        # Check whether the requested analysis type has a corresponding result comparison method
        if not (self.analysis_type in self.analysis_check_map):
            raise ValueError("'{}' is not a valid analysis type, options are: {}".format(self.analysis_type, list(self.analysis_check_map.keys)))

        # Perform the comparison
        self.analysis_check_map[self.analysis_type]()


    def CompareContents(self, match_entry_lengths=True):
        reference_keys = list(self.reference_results.keys())
        test_keys = list(self.test_results.keys())

        for key in reference_keys:
            if not (key in test_keys):
                raise ValueError("'{}' is in the reference but not the test file!".format(key))
            else:
                if match_entry_lengths:
                    if len(self.reference_results[key]) != len(self.test_results[key]):
                        raise ValueError("result array size mismatch for '{}': {} != {}".format(
                            key, len(self.reference_results[key]), len(self.test_results[key])))

        if not (self.allow_additional_data):
            for key in test_keys:
                if (not key in reference_keys):
                    raise KeyError("The test file contains '{}' while the reference does not".format(key))


    def CompareDefaultAnalysisResults(self):
        """Directly compares each component of every data column to its corresponding reference"""
        self.CompareContents()
        for key, reference_values in self.reference_results.items():
            test_values = self.test_results[key]

            if len(test_values) != len(reference_values):
                raise ValueError("result array size mismatch for '{}'".format(key))

            for index, (reference, test) in enumerate(zip(reference_values, test_values)):
                self._CheckValues(test, reference, key, index)


    def CompareEigenmodeIdentificationResults(self):
        """Compare eigenmodes defined by their 'TypeCounter' and 'Type'"""
        self.CompareContents(match_entry_lengths=False)

        # Require 'TypeCounter' and 'Type' in both files
        for results, results_name in zip([self.reference_results, self.test_results], ["reference", "test"]):
            if not ("TypeCounter" in results.keys()):
                raise KeyError("'TypeCounter' is required for 'eigenmode_identification' analyses, but the {} file does not have it".format(results_name))

            if not ("Type" in results.keys()):
                raise KeyError("'Type' is required for 'eigenmode_identification' analyses, but the {} file does not have it".format(results_name))

        for reference_index, (reference_type, reference_type_counter) in enumerate(zip(self.reference_results["Type"], self.reference_results["TypeCounter"])):
            for test_index in range(len(self.test_results["Type"])):
                if self.test_results["Type"][test_index] == reference_type and self.test_results["TypeCounter"][test_index] == reference_type_counter:
                    # Found entries with matching 'Type' and 'TypeCounter'
                    # -> compare all data fields in reference to the corresponding ones in test
                    for reference_key, reference_value in self.reference_results.items():
                        self._CheckValues(
                            self.test_results[reference_key][test_index],
                            reference_value[reference_index],
                            reference_key,
                            test_index
                        )


    def _CheckValues(self, test, reference, key: str, index: int):
        """Based on their types, redirects the test and reference values to a function written to check their equality"""
        ValueType = type(test)

        if ValueType != type(reference):
            raise TypeError("test type '{}' is not identical to reference type '{}'".format(ValueType, type(reference)))

        if issubclass(ValueType, (float, numpy.float64, numpy.float32)):
            self._CheckFloats(test, reference, key, index)

        elif issubclass(ValueType, (int)):
            self._CheckInts(test, reference, key, index)

        elif issubclass(ValueType, (str)):
            self._CheckStrings(test, reference, key, index)

        else:
            raise TypeError("no checks implemented for type '{}'".format(ValueType))


    def _CheckFloats(self, test: float, reference: float, key: str, index: int):
        self.test_case.assertAlmostEqual(
            test,
            reference,
            msg = "('{key}' at index {index})".format(
                key = key,
                index = index
            ),
            delta=self.delta
        )


    def _CheckInts(self, test: int, reference: int, key: str, index: int):
        self.test_case.assertEqual(
            test,
            reference,
            "('{key}' at index {index})".format(
                key = key,
                index = index
            )
        )


    def _CheckStrings(self, test: str, reference: str, key: str, index: int):
        self.test_case.assertTrue(
            test == reference,
            "('{key}' at index {index})".format(
                key = key,
                index = index
            )
        )


    @property
    def analysis_check_map(self):
        return {
            "default" : self.CompareDefaultAnalysisResults,
            "eigenmode_identification" : self.CompareEigenmodeIdentificationResults
        }


    @staticmethod
    def GetDefaultSettings():
        return {
            "delta" : 1e-16,
            "analysis_type" : "default",
            "allow_additional_data" : True
        }


    @staticmethod
    def ValidateAndAddDefaults(settings: dict):
        """Check whether all keys in the arguments are valid, and append it with keys are missing"""
        default_settings = ResultFileComparator.GetDefaultSettings()

        for key in settings.keys():
            if not (key in default_settings):
                raise KeyError("{} is not a valid setting to be specified".format(key))

        for key, value in default_settings.items():
            if not (key in settings):
                settings[key] = value

        return settings


    @staticmethod
    def _CheckPath(path: pathlib.Path) -> pathlib.Path:
        if not isinstance(path, pathlib.Path):
            if isinstance(path, str):
                path = pathlib.Path(path)
            else:
                raise TypeError("Invalid path type: {}".format(type(path)))

        if not path.is_file():
            raise FileNotFoundError(str(path))

        return path


class ResultFileComparator(ResultComparator):

    def __init__(self, testFilePath: pathlib.Path, referenceFilePath: pathlib.Path, *args, **kwargs):
        # Check input types
        testilePath = self._CheckPath(testFilePath)
        referenceFilePath = self._CheckPath(referenceFilePath)

        # Check file types
        extension = str(testFilePath.suffix).lower()
        if extension != str(referenceFilePath.suffix).lower():
            raise RuntimeError("File type mismatch: {} and {}".format(testFilePath, referenceFilePath))

        # Parse
        Parser = ParsedResults.GetParserForFileType(extension)
        test_results = Parser(testFilePath)
        reference_results = Parser(referenceFilePath)
        ResultComparator.__init__(self, test_results, reference_results, *args, **kwargs)