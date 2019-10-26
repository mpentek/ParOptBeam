# ===============================================================================
"""
        Derived classes from ResidualBasedSolver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

import numpy as np

from source.solving_strategies.strategies.rasidual_based_solver import ResidualBasedSolver


class ResidualBasedPicardSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_residual(self, u1):
        v1 = self.scheme.predict_velocity(u1)
        a1 = self.scheme.predict_acceleration(v1)
        f = self.force[:, self.step]

        self.ru = f - (np.dot(self.M, a1) + np.dot(self.B, v1) + np.dot(self.K, u1))
        print(self.ru)

    def calculate_increment(self):
        pass

    def solve_single_step(self):
        pass
