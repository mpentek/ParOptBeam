import numpy as np

from source.solving_strategies.schemes.time_integration_scheme import TimeIntegrationScheme


class ForwardEuler1(TimeIntegrationScheme):
    """
    (Explicit) Forward Euler 1st order approximation 

    """

    def __init__(self, dt, comp_model, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        super().__init__(dt, comp_model, initial_conditions)

        # initial values for time integration
        self.un1 = self.u0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing (Explicit) Forward Euler 1 st order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_single_step(self, f1):

        # calculates self.un0,vn0,an0
        self.u1 = self.un1 + self.dt * self.vn1
        self.v1 = self.vn1 + self.dt * self.an1

        LHS = self.M
        RHS = f1 - np.dot(self.B, self.v1) - np.dot(self.K, self.u1)
        self.a1 = np.linalg.solve(LHS, RHS)

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un1 = self.u1
        self.vn1 = self.v1
        self.an1 = self.a1
