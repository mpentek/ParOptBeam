# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

from source.solving_strategies.strategies.rasidual_based_solver import ResidualBasedSolver


class ResidualBasedNewtonRaphsonSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt, comp_model, initial_conditions, force):
        super().__init__(array_time, time_integration_scheme, dt, comp_model, initial_conditions, force)

    def calculate_residual(self):
        passs

    def calculate_increment(self):
        pass

    def solve_single_step(self):
        pass
