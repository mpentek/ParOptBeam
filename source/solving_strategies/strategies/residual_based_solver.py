# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

from source.solving_strategies.strategies.solver import Solver
from source.auxiliary.global_definitions import *
import numpy as np

# TODO: take these values as user input
# stopping criteria
TOL = 1.e-12
# maximum iteration
MAX_IT = 10


class ResidualBasedSolver(Solver):

    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_residual(self, q):
        pass

    def calculate_increment(self, ru):
        pass

    def update_displacement(self, new_displacement):
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
        self.update_displacement(new_displacement)
        # update residual
        ru = self.calculate_residual(f_ext)

        while abs(np.max(ru)) > TOL and nr_it < MAX_IT:
            nr_it += 1
            print("Nonlinear iteration: ", str(nr_it))
            print("ru = {:.2e}".format(abs(np.max(ru))))
            self.K = self.structure_model.update_stiffness_matrix()
            du = self.calculate_increment(ru)
            du = self.structure_model.recuperate_bc_by_extension(du, 'column_vector')

            # updating displacement in the element
            self.update_incremental(du)
            ru = self.calculate_residual(f_ext)

        u_new = self.get_displacement_from_element()
        self.scheme.update_displacement(u_new)

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
        # for i in range(0, 10):
            self.step = i
            current_time = self.array_time[i]
            print("time: {0:.2f}".format(current_time))

            self.solve_single_step()

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()

            # update results
            self.scheme.update()
