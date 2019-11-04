# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================

from source.solving_strategies.strategies.solver import Solver


class ResidualBasedSolver(Solver):

    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_residual(self, q):
        pass

    def calculate_increment(self, ru):
        pass

    def solve_single_step(self):
        pass

    def update_comp_model(self):
        self.structure_model.calculate_global_matrices()
        self.M = self.structure_model.comp_m
        self.B = self.structure_model.comp_b
        self.K = self.structure_model.comp_k
        self.scheme.update_comp_model([self.M, self.B, self.K])

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

            # update results
            self.scheme.update()
