# ===============================================================================
"""
        Derived classes from ResidualBasedSolver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================
import numpy as np

from source.solving_strategies.strategies.rasidual_based_solver import ResidualBasedSolver
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
        q = np.zeros(self.structure_model.n_nodes * DOFS_PER_NODE[self.structure_model.domain_size])

        for e in self.structure_model.elements:
            start_index = DOFS_PER_NODE[e.domain_size] * e.index
            end_index = start_index + e.ElementSize

            q[start_index:
              DOFS_PER_NODE[e.domain_size] * e.index +
              DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL] += e.qe

        q = self.structure_model.apply_bc_by_reduction(q, 'column_vector')
        ru = f_ext - q
        return ru

