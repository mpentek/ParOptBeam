'''
Copyright and license disclosure:
a partial replica of https://github.com/KratosMultiphysics/Kratos/blob/master/kratos/python_scripts/compare_two_files_check_process.py
'''


from .validate_and_assign_defaults import validate_and_assign_defaults
import unittest
from numpy import isclose as t_isclose

# Other imports
import filecmp
import os
import math

class CompareTwoFilesCheck(unittest.TestCase):
    """This process compares files that are written during a simulation
    against reference files.
    Please see the "ExecuteFinalize" functions for details about the
    available file-formats
    """
    def __init__(self, params):

        ## Settings string in json format
        default_parameters = {
            "help"                  : "This process checks that two files are the same. This can be used in order to create tests, where a given solution is expected",
            "reference_file_name"   : "",
            "output_file_name"      : "",
            "remove_output_file"    : True,
            "tolerance"             : 1e-6,
            "relative_tolerance"    : 1e-9
        }

        # backwards compatibility
        if "decimal_places" in params:
            if "tolerance" in params or "relative_tolerance" in params:
                raise Exception('Conflicting settings specified, please remove "decimal_places"')
            decimal_places = params["decimal_places"]
            warning =  'W-A-R-N-I-N-G: You have specified "decimal_places", '
            warning += 'which is deprecated and will be removed soon.\n'
            warning += 'Please specify "tolerance" and "relative_tolerance" instead!'
            print(warning)
            tol = 0.1**decimal_places
            params["tolerance"] = tol


        ## Overwrite the default settings with user-provided parameters
        validate_and_assign_defaults(default_parameters, params)

        self.reference_file_name = params["reference_file_name"]
        self.output_file_name = params["output_file_name"]

        self.remove_output_file = params["remove_output_file"]

        # TODO: perhaps enhance for various types of comparison
        # self.comparison_type = params["comparison_type"]

        self.tol = params["tolerance"]
        self.reltol = params["relative_tolerance"]

        self.info_msg = "".join([  "\n[%s]: Failed with following parameters:\n" % self.__class__.__name__,
                                    str(params)
                                ])

    def execute(self):
        self.__CompareDatFile()

        # TODO: perhaps enhance with various types of comparison
        # elif (self.comparison_type == "dat_file"):
        #     self.__CompareDatFile()
        # elif (self.comparison_type == "csv_file"):
        #     self.__CompareCSVFile()
        # elif (self.comparison_type == "dat_file_variables_time_history"):
        #     self.__CompareDatFileVariablesTimeHistory()

        if self.remove_output_file:
            if os.path.exists(self.output_file_name):
                os.remove(self.output_file_name)

    def __GetFileLines(self):
        """This function reads the reference and the output file
        It returns the lines read from both files and also compares
        if they contain the same numer of lines
        """
        # check if files are valid
        if not os.path.isfile(self.reference_file_name):
            err_msg  = 'The specified reference file name "'
            err_msg += self.reference_file_name
            err_msg += '" is not valid!'
            raise Exception(err_msg)
        if not os.path.isfile(self.output_file_name):
            err_msg  = 'The specified output file name "'
            err_msg += self.output_file_name
            err_msg += '" is not valid!'
            raise Exception(err_msg)

        # "readlines" adds a newline at the end of the line,
        # which will be removed with rstrip afterwards
        with open(self.reference_file_name,'r') as ref_file:
            lines_ref = ref_file.readlines()
        with open(self.output_file_name,'r') as out_file:
            lines_out = out_file.readlines()

        # removing trailing newline AND whitespaces than can mess with the comparison
        lines_ref = [line.rstrip() for line in lines_ref]
        lines_out = [line.rstrip() for line in lines_out]

        num_lines_ref = len(lines_ref)
        num_lines_out = len(lines_out)

        err_msg  = "Files have different number of lines!"
        err_msg += "\nNum Lines Reference File: " + str(num_lines_ref)
        err_msg += "\nNum Lines Output File: " + str(num_lines_out)
        self.assertTrue(num_lines_ref == num_lines_out, msg=err_msg + self.info_msg)

        return lines_ref, lines_out

    def __CompareDatFile(self):
        """This function compares files with tabular data.
        => *.dat
        Lines starting with "#" are comments and therefore compared for equality
        Other lines are compared to be almost equal with the specified tolerance
        """
        lines_ref, lines_out = self.__GetFileLines()

        # assert headers are the same
        lines_ref, lines_out = self.__CompareDatFileComments(lines_ref, lines_out)

        # assert values are equal up to given tolerance
        self.__CompareDelimittedFileResults(lines_ref, lines_out, None)

    def __CompareCSVFile(self):
        """This function compares files with tabular data.
        => *.csv
        Lines starting with "#" are comments and therefore compared for equality
        Other lines are compared to be almost equal with the specified tolerance
        """
        lines_ref, lines_out = self.__GetFileLines()

        # assert headers are the same
        lines_ref, lines_out = self.__CompareDatFileComments(lines_ref, lines_out)

        # assert values are equal up to given tolerance
        self.__CompareDelimittedFileResults(lines_ref, lines_out, ",")

    def __CompareDatFileVariablesTimeHistory(self):
        """This function compares files with tabular data.
        => *.dat
        Lines starting with "#" are comments and therefore compared for equality
        Other lines are compared to be almost equal with the specified tolerance
        If the comparison fails, it prints the location of failure with details
        The expected format is the one written by the PointOutputProcess:

        # some basic file information
        # time var_name_1 var_name_2
        0.1 1.2345 2.852
        0.2 0.889 -89.444
        .
        .
        .
        """
        lines_ref, lines_out = self.__GetFileLines()

        # extracting the names of output variables eg: time, VELOCITY_X, VELOCITY_Y, VELOCITY_Z
        variable_names = lines_ref[1].split()[2:]

        # assert headers are the same
        lines_ref, lines_out = self.__CompareDatFileComments(lines_ref, lines_out)

        # assert values are equal up to given tolerance
        self.__CompareDatFileResultsWithLocation(lines_ref, lines_out, variable_names)

    def __CompareDatFileComments(self, lines_ref, lines_out):
        """This function compares the comments of files with tabular data
        The lines starting with "#" are being compared
        These lines are removed from the list of lines
        """
        for line_ref, line_out in zip(lines_ref, lines_out):
            if line_ref.lstrip()[0] == '#' or line_out.lstrip()[0] == '#':
                self.assertTrue(line_ref == line_out, msg = self.info_msg)

        lines_ref = [line for line in lines_ref if not(line.lstrip()[0] == '#')]
        lines_out = [line for line in lines_out if not(line.lstrip()[0] == '#')]

        return lines_ref, lines_out

    def __CompareDelimittedFileResults(self, lines_ref, lines_out, delimiter):
        """This function compares the data of files with tabular data
        The comment lines were removed beforehand
        """
        for line_ref, line_out in zip(lines_ref, lines_out):
            for v1, v2 in zip(line_ref.split(delimiter), line_out.split(delimiter)):
                self.__CheckCloseValues(float(v1), float(v2))

    def __CompareDatFileResultsWithLocation(self, lines_ref, lines_out, variable_names):
        """This function compares the data of files with tabular data
        It also prints the exact location where data doesnt match each other
        The comment lines were removed beforehand
        """
        for line_ref, line_out in zip(lines_ref, lines_out):
            for i_var, (v1, v2) in enumerate(zip(line_ref.split(), line_out.split())):
                if i_var == 0: # comparing time:
                    additional_info = 'Different time found!'
                    self.__CheckCloseValues(float(v1), float(v2), additional_info)
                    current_time = v1
                else: # comparing variables
                    additional_info  = 'Failed for variable ' + variable_names[i_var-1]
                    additional_info += ' at time: ' + current_time
                    self.__CheckCloseValues(float(v1), float(v2), additional_info)

    def __CheckCloseValues(self, val_a, val_b, additional_info=""):
        isclosethis = t_isclose(val_a, val_b, rtol=self.reltol, atol=self.tol)
        full_msg =  self.info_msg + "\n"
        full_msg += str(val_a) + " != " + str(val_b) + ", rel_tol = "
        full_msg += str(self.reltol) + ", abs_tol = " + str(self.tol)
        if additional_info != "":
            full_msg += "\n" + additional_info
        self.assertTrue(isclosethis, msg=full_msg)