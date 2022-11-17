import numpy as np

from source.solving_strategies.schemes.time_integration_scheme import TimeIntegrationScheme


class BackwardEuler1(TimeIntegrationScheme):
    """
    (Implicit) Backward Euler 1st order approximation
    Also known as BDF 1

    """

    def __init__(self, dt, comp_model, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        super().__init__(dt, comp_model, initial_conditions)

        # initial values for time integration
        self.un1 = self.u0
        self.un2 = self.u0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing (Implicit) Backward Euler 1 st order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def predict_velocity(self, u1):
        v1 = (u1 - self.un1) / self.dt
        return v1

    def predict_acceleration(self, v1):
        a1 = (v1 - self.vn1) / self.dt
        return a1

    def solve_single_step(self, f1):
        # LHS needs to be updated in case of non-linear elements
        LHS = self.M + self.B * self.dt + self.K * self.dt ** 2

        if self.M.ndim == 2:
            # system: in matrix form
            RHS = self.dt * np.dot(self.B, self.un1) + 2 * np.dot(self.M, self.un1)
            RHS += - np.dot(self.M, self.un2) + self.dt ** 2 * f1

            # main solve
            self.u1 = np.linalg.solve(LHS, RHS)

        elif self.M.ndim == 1:
            # system: in vector (from diagonal matrix) or scalar form
            RHS = self.dt * self.B * self.un1 + 2 * self.M * self.un1
            RHS += - self.M * self.un2 + self.dt ** 2 * f1

            # main solve
            self.u1 = RHS/LHS

        else:
            raise Exception('Dimension of system parameters is BackwardEuler1 is wrong')

        # updates self.v1,a1
        self.v1 = self.predict_velocity(self.u1)
        self.a1 = self.predict_acceleration(self.v1)

    def update(self):
        # update self.un2 un1
        self.un2 = self.un1
        self.un1 = self.u1
        self.vn1 = self.v1
        self.an1 = self.a1
