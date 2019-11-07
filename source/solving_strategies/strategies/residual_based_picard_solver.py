from source.solving_strategies.schemes.forward_euler1_scheme import ForwardEuler1
from source.solving_strategies.schemes.backward_euler1_scheme import BackwardEuler1
from source.solving_strategies.strategies.residual_based_solver import ResidualBasedSolver
import numpy as np


# TODO: take these values as user input
# stopping criteria
TOL = 1.e-5
# maximum iteration
MAX_IT = 15


class ResidualBasedPicardSolver(ResidualBasedSolver):
    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def solve_single_step(self):
        # predict displacement at time step n with external force f_ext
        f_ext = self.force[:, self.step]
        self.scheme.solve_single_step(f_ext)
        u1 = self.scheme.get_displacement()

        nr_it = 0
        ru = self.calculate_residual(u1, f_ext)

        while abs(np.linalg.norm(ru)) > TOL and nr_it < MAX_IT:
            print("Nonlinear iteration: ", str(nr_it))
            print("ru = {:.2e}".format(abs(np.linalg.norm(ru))))
            u1 = self.scheme.get_displacement()
            ru = self.calculate_residual(u1, f_ext)
            du = self.calculate_increment(ru)
            self.scheme.u1 += du
            nr_it += 1

        u_new = self.scheme.get_displacement()
        u_new = self.structure_model.recuperate_bc_by_extension(u_new, 'column_vector')
        self.update_total(u_new)
        # updating K, B, M in the scheme
        self.update_comp_model()

    def calculate_increment(self, ru):
        du = np.zeros(ru.shape)

        if isinstance(self.scheme, ForwardEuler1):
            LHS = self.M
            RHS = ru * self.dt ** 2
            du = np.linalg.solve(LHS, RHS)
        if isinstance(self.scheme, BackwardEuler1):
            LHS = (self.B * self.dt + self.K * self.dt ** 2 + self.M)
            RHS = ru * self.dt ** 2
            du = np.linalg.solve(LHS, RHS)
        return du

    def calculate_residual(self, u1, f_ext):
        v1 = self.scheme.predict_velocity(u1)
        a1 = self.scheme.predict_acceleration(v1)

        ru = f_ext - (np.dot(self.M, a1) + np.dot(self.B, v1) + np.dot(self.K, u1))
        return ru

