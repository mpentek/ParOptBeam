import numpy as np

from source.solving_strategies.schemes.time_integration_scheme import TimeIntegrationScheme


class Euler12(TimeIntegrationScheme):
    """
    Euler 1st and 2nd order 

    """

    def __init__(self, dt, comp_model, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        super().__init__(dt, comp_model, initial_conditions)

        # initial values for time integration
        self.un2 = self.u0
        self.un1 = self.u0
        self.vn2 = self.v0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing Euler 1st and 2nd order method integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def predict_velocity(self, u1):
        v1 = (u1 - self. un2) / 2 / self.dt
        return v1

    def predict_acceleration(self, v1):
        a1 = (v1 - self. vn2) / 2 / self.dt
        return a1

    def solve_single_step(self, f1):
        # LHS needs to be updated in case of non-linear elements
        LHS = self.M + np.dot(self.B, self.dt / 2)

        if self.M.ndim == 2:
            # system: in matrix form
            RHS = f1 * self.dt ** 2 + \
                np.dot(self.un1, (2 * self.M - self.K * self.dt ** 2))
            RHS += np.dot(self.un2, (-self.M + self.B * self.dt/2))

            # main solve
            self.u1 = np.linalg.solve(LHS, RHS)

        elif self.M.ndim == 1:
            # system: in vector (from diagonal matrix) or scalar form
            RHS = f1 * self.dt ** 2 + \
                self.un1 * (2 * self.M - self.K * self.dt ** 2)
            RHS += self.un2 * (-self.M + self.B * self.dt/2)

            # main solve
            self.u1 = RHS/LHS

        else:
            raise Exception('Dimension of system parameters is Euler12 is wrong')

        # updates self.v1,a1
        self.v1 = self.predict_velocity(self.u1)
        self.a1 = self.predict_acceleration(self.v1)

    def update(self):
        # update self.un2 un1
        self.un2 = self.un1
        self.un1 = self.u1
        self.vn2 = self.vn1
        self.vn1 = self.v1
