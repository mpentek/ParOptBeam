# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================
import numpy as np

from source.solving_strategies.strategies.residual_based_solver import ResidualBasedSolver
from source.auxiliary.global_definitions import *

# TODO: take these values as user input
# stopping criteria
TOL = 1.e-5
# maximum iteration
MAX_IT = 10


class ResidualBasedNewtonRaphsonSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def update_total(self, new_displacement):
        # updating displacement in the element
        for e in self.structure_model.elements:
            i_start = DOFS_PER_NODE[e.domain_size] * e.index
            i_end = DOFS_PER_NODE[e.domain_size] * e.index + DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL
            new_displacement_e = new_displacement[i_start: i_end]
            e.update_total(new_displacement_e)

    def update_incremental(self, dp):
        # updating displacement in the element
        for e in self.structure_model.elements:
            i_start = DOFS_PER_NODE[e.domain_size] * e.index
            i_end = DOFS_PER_NODE[e.domain_size] * e.index + DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL
            dp_e = dp[i_start: i_end]
            e.update_incremental(dp_e)

    def get_displacement_from_element(self):
        u = np.zeros(self.structure_model.n_nodes * DOFS_PER_NODE[self.structure_model.domain_size])

        for e in self.structure_model.elements:
            i_start = DOFS_PER_NODE[e.domain_size] * e.index
            i_end = DOFS_PER_NODE[e.domain_size] * e.index + DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL
            u[i_start:i_end] += e.current_deformation

        u = self.structure_model.apply_bc_by_reduction(u, 'column_vector')
        return u

    def solve_single_step(self):
        f_ext = self.force[:, self.step]
        # predict displacement at time step n with external force f_ext
        self.scheme.solve_single_step(f_ext)

        nr_it = 0
        # update displacement in element
        new_displacement = self.scheme.get_displacement()
        new_displacement = self.structure_model.recuperate_bc_by_extension(new_displacement, 'column_vector')
        self.update_total(new_displacement)
        # update stiffness matrix before updating residual
        self.K = self.structure_model.update_stiffness_matrix()
        # update residual
        r = self.calculate_residual(f_ext)

        while abs(np.max(r)) > TOL and nr_it < MAX_IT:
            nr_it += 1
            print("Nonlinear iteration: ", str(nr_it))
            print("ru = {:.2e}".format(abs(np.max(r))))
            dp = self.calculate_increment(r)
            dp = self.structure_model.recuperate_bc_by_extension(dp, 'column_vector')

            # updating displacement in the element
            self.update_incremental(dp)
            self.K = self.structure_model.update_stiffness_matrix()
            r = self.calculate_residual(f_ext)

        u_new = self.get_displacement_from_element()
        self.scheme.update_displacement(u_new)
        # updating K, B, M in the scheme
        self.update_comp_model()

    def calculate_increment(self, r):
        dp = np.linalg.solve(self.K, r)
        return dp

    def calculate_residual(self, f_ext):
        q = np.zeros(self.structure_model.n_nodes * DOFS_PER_NODE[self.structure_model.domain_size])

        # TODO: calculate the residual in the element, because otherwise the boundary condtion wouldn't apply
        for e in self.structure_model.elements:
            start_index = DOFS_PER_NODE[e.domain_size] * e.index
            end_index = DOFS_PER_NODE[e.domain_size] * e.index + DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL

            q[start_index:end_index] += e.nodal_force_global

        q = self.structure_model.apply_bc_by_reduction(q, 'column_vector')
        ru = f_ext - q
        return ru