# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

from source.solving_strategies.strategies.solver import Solver
from source.auxiliary.global_definitions import *


class ResidualBasedSolver(Solver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

        # displacement residual
        self.ru = 0.0

    def calculate_residual(self):
        pass

    def calculate_increment(self):
        pass

    def solve_single_step(self):
        pass

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
        # for i in range(0, 10):
            self.step = i
            current_time = self.array_time[i]
            print("time: ", "{0:.2f}".format(current_time))
            self.scheme.solve_single_step(self.force[:, i])

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()
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
            self.scheme.update_structure_time_step()
