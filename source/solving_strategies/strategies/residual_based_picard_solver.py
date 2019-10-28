# ===============================================================================
"""
        Derived classes from ResidualBasedSolver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================
import numpy as np

from source.solving_strategies.strategies.residual_based_solver import ResidualBasedSolver
from source.auxiliary.global_definitions import *


class ResidualBasedPicardSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_increment(self, ru):
        du = np.linalg.solve(self.K, ru)
        return du

    def calculate_residual(self, f_ext):
        pass

