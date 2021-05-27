# --- STL Imports ---
import pathlib


"""
<ParOptBeam root>
    > input
    > output
        > [<analysis name>]
    > source
        > ...
        > test_utils
            > code_structure.py
    > test_scripts
        > analytical_reference_results
        > kratos_reference_results
        > ...
"""

ROOT_DIRECTORY = pathlib.Path(__file__).absolute().parent.parent.parent

SOURCE_DIRECTORY = ROOT_DIRECTORY / "source"

TEST_SCRIPTS_DIRECTORY = SOURCE_DIRECTORY / "test_scripts"

TEST_UTILS_DIRECTORY = SOURCE_DIRECTORY / "test_utils"

TEST_ANALYTICAL_REFERENCE_RESULTS_DIRECTORY = TEST_SCRIPTS_DIRECTORY / "analytical_reference_results"

TEST_KRATOS_REFERENCE_RESULTS_DIRECTORY = TEST_SCRIPTS_DIRECTORY / "kratos_reference_results"

INPUT_DIRECTORY = ROOT_DIRECTORY / "input"

OUTPUT_DIRECTORY = ROOT_DIRECTORY / "output"