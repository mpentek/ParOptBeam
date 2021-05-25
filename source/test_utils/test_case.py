# --- STL Imports ---
import unittest


class TestCase(unittest.TestCase):
    """Class derived from unittest.TestCase that can be extended later if necessary"""
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)