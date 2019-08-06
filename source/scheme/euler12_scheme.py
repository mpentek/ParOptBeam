import numpy as np

from source.scheme.time_integration_scheme import TimeIntegrationScheme


class Euler12(TimeIntegrationScheme):
    """
    Euler 1st and 2nd order 

    """

    def __init__(self, dt, structure, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt

        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)

        # structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]

        # initial values for time integration
        self.un2 = self.u0
        self.un1 = self.u0 - self.v0 * self.dt + self.a0 * (self.dt ** 2 / 2)
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing Euler 1st and 2nd order method integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):
        LHS = self.M + np.dot(self.B, self.dt / 2)
        RHS = f1 * self.dt ** 2 + \
            np.dot(self.un1, (2 * self.M - self.K * self.dt ** 2))
        RHS += np.dot(self.un2, (-self.M + self.B * self.dt/2))

        # calculates self.un0,vn0,an0
        self.un0 = np.linalg.solve(LHS, RHS)
        self.vn0 = (self.un0 - self. un2) / 2 / self.dt
        self.an0 = (self.un0 - 2 * self.un1 + self.un2) / self.dt ** 2

        # update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un2 = self.un1
        self.un1 = self.un0
