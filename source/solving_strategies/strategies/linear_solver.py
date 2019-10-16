# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

from source.solving_strategies.strategies.solver import Solver


class LinearSolver(Solver):
    def __init__(self, array_time, time_integration_scheme, dt, comp_model, initial_conditions, force):

        super().__init__(array_time, time_integration_scheme, dt, comp_model, initial_conditions, force)

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
            current_time = self.array_time[i]
            print("time: ", "{0:.2f}".format(current_time))
            self.scheme.solve_single_step(self.force[:, i])

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()

            # update results
            self.scheme.update_structure_time_step()
