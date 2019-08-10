import numpy as np

from source.scheme.time_integration_scheme import TimeIntegrationScheme


class BackwardEuler1(TimeIntegrationScheme):
    """
    (Implicit) Backward Euler 1st order approximation

    """

    def __init__(self, dt, comp_model, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt

        # mass, damping and spring stiffness
        self.M = comp_model[0]
        self.B = comp_model[1]
        self.K = comp_model[2]

        # structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]

        # initial values for time integration
        self.un1 = self.u0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing (Implicit) Backward Euler 1 st order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):

        # calculates self.un0,vn0,an0
        LHS = self.M + np.dot(self.B, self.dt) + np.dot(self.K, self.dt ** 2)
        RHS = f1 - np.dot(self.B, self.vn1) - np.dot(self.K,
                                                     self.un1) - np.dot(self.K, self.vn1 * self.dt)
        self.an0 = np.linalg.solve(LHS, RHS)
        self.vn0 = self.vn1 + self.dt * self.an0
        self.un0 = self.un1 + self.dt * self.vn0

        # update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un1 = self.un0
        self.vn1 = self.vn0
        self.an1 = self.an0
