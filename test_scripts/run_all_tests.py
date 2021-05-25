# --- Internal Imports ---
from source.test_utils.code_structure import ROOT_DIRECTORY, TEST_SCRIPTS_DIRECTORY, OUTPUT_DIRECTORY

# --- STL Imports ---
import unittest
import contextlib
import sys


class UnbufferedStream(object):
    """A stream buffer that flushes after every call to write"""

    def __init__(self, stream):
        self._stream = stream


    def write(self, content: str):
        self._stream.write(content)
        self._stream.flush()


    def writelines(self, lines):
        self._stream.writelines(lines)
        self._stream.flush()


    def __getattr__(self, attribute):
        return getattr(self._stream, attribute)


class StreamMultiplex(object):

    def __init__(self, *streams):
        self._streams = list(streams)


    def write(self, content):
        for stream in self._streams:
            stream.write(content)


    def writelines(self, contents):
        for stream in self._streams:
            stream.writeline(contents)


    def __getattr__(self, attribute):
        # This should return an error
        return getattr(self._streams[0], attribute)



if __name__ == "__main__":
    test_suite = unittest.TestLoader().discover(
        start_dir = TEST_SCRIPTS_DIRECTORY,
        pattern = "test_*.py",
        top_level_dir = ROOT_DIRECTORY
    )

    test_output_file_name = OUTPUT_DIRECTORY / "test_output.log"
    test_results_file_name = OUTPUT_DIRECTORY / "test_results.log"
    stderr = sys.stderr

    with open(test_output_file_name, 'w') as output_file, open(test_results_file_name, 'w') as results_file:

        stream = UnbufferedStream(output_file)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            
            unittest.TextTestRunner(
                stream = UnbufferedStream(StreamMultiplex(stderr, results_file)),
                verbosity = 2
            ).run(test_suite)