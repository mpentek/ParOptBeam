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

    def calculate_residual(self, un1):
        pass

    def calculate_increment(self):
        pass

    def solve_single_step(self):
        self.scheme.solve_single_step(self.force[:, self.step])

        nr_it = 0
        ru = 1.0
        while abs(np.max(ru)) > TOL and nr_it < MAX_IT:
            u1 = self.scheme.get_displacement()
            ru = self.calculate_residual(u1)
            du = self.scheme.calculate_increment(ru)
            self.scheme.apply_increment_and_update(du)
            print("Nonlinear iteration: ", str(nr_it))
            print("ru = {:.2e}".format(abs(np.max(ru))))
            nr_it += 1

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
            self.step = i
            current_time = self.array_time[i]
            print("time: {0:.2f}".format(current_time))

            self.solve_single_step()

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()

            # TODO: remove this once the test in test_schemes works without structure model
            if self.structure_model is not None:
                self.dynamic_reaction[:, i] = self._compute_reaction(
                    self.displacement[:, i],
                    self.velocity[:, i],
                    self.acceleration[:, i])

                # updating deformation and reaction in the element
                for e in self.structure_model.elements:
                    e.update_nodal_information(
                        self.displacement[DOFS_PER_NODE[e.domain_size] * e.index:
                                          DOFS_PER_NODE[e.domain_size] * e.index +
                                          DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL, i],
                        self.dynamic_reaction[
                        DOFS_PER_NODE[e.domain_size] * e.index:
                        DOFS_PER_NODE[e.domain_size] * e.index +
                        DOFS_PER_NODE[e.domain_size] * NODES_PER_LEVEL, i])

            # update results
            self.scheme.update()
