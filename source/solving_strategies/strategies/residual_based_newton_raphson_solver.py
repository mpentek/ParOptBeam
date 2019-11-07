import numpy as np

from source.solving_strategies.strategies.rasidual_based_solver import ResidualBasedSolver
from source.auxiliary.global_definitions import *

# TODO: take these values as user input
# stopping criteria
TOL = 1e-4
# maximum iteration
MAX_IT = 10


class ResidualBasedNewtonRaphsonSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def update_incremental(self, dp):
        # updating displacement in the element
        for e in self.structure_model.elements:
            i_start = DOFS_PER_NODE[e.domain_size] * e.index
            i_end = DOFS_PER_NODE[e.domain_size] * e.index + DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL
            dp_e = dp[i_start: i_end]
            e.update_incremental(dp_e)

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

        while abs(np.linalg.norm(r)) > TOL and nr_it < MAX_IT:
            nr_it += 1
            print("Nonlinear iteration: ", str(nr_it))
            print("r = {:.2e}".format(abs(np.linalg.norm(r))))
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

    def _compute_reaction(self):
        f_ext = self.force[:, self.step]
        reaction = f_ext - self.get_internal_force_from_element()
        return reaction

    def calculate_residual(self, f_ext):
        q = self.get_internal_force_from_element()
        r = f_ext - q
        return r
