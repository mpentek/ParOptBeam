from source.solving_strategies.strategies.rasidual_based_solver import ResidualBasedSolver


class ResidualBasedPicardSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_residual(self):
        # self.ru = f - (M * a_n1 + C * v_n1 + self.K(u_n1) * u_n1)
        pass

    def calculate_increment(self):
        pass

    def solve_single_step(self):
        pass
