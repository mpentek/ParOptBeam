import numpy as np

from source.scheme.time_integration_scheme import TimeIntegrationScheme


class BDF2(TimeIntegrationScheme):
    """
    BDF2 2nd order

    """

    def __init__(self, dt, comp_model, initial_conditions):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt

        # bdf2 scheme coefficients
        self.bdf0 = 1.5 / self.dt
        self.bdf1 = -2. / self.dt
        self.bdf2 = 0.5 / self.dt

        # mass, damping and spring stiffness
        self.M = comp_model[0]
        self.B = comp_model[1]
        self.K = comp_model[2]

        self.LHS = self.bdf0 * self.B + self.K + self.bdf0 * self.bdf0 * self.M

        # structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]

        # initial values for time integration
        self.an4 = self.a0
        self.vn4 = self.v0
        self.un4 = self.u0
        self.an3 = - np.dot(self.B, self.vn4) - np.dot(self.K, self.un4) + self.an4
        self.vn3 = self.vn4 + self.an3 * self.dt
        self.un3 = self.un4 + self.vn3 * self.dt
        self.an2 = - np.dot(self.B, self.vn3) - np.dot(self.K, self.un3) + self.an3
        self.vn2 = self.vn3 + self.an2 * self.dt
        self.un2 = self.un3 + self.vn2 * self.dt
        self.an1 = - np.dot(self.B, self.vn2) - np.dot(self.K, self.un2) + self.an2
        self.vn1 = self.vn2 + self.an1 * self.dt
        self.un1 = self.un2 + self.vn1 * self.dt

        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing BDF2 2nd order method integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):
        RHS = - np.dot(self.B, self.bdf1 * self.un1) - np.dot(self.B, self.bdf2 * self.un2)
        RHS += - 2 * self.bdf0 * self.bdf1 * np.dot(self.M, self.un1)
        RHS += - 2 * self.bdf0 * self.bdf2 * np.dot(self.M, self.un2)
        RHS += -     self.bdf1 * self.bdf1 * np.dot(self.M, self.un2)
        RHS += - 2 * self.bdf1 * self.bdf2 * np.dot(self.M, self.un3)
        RHS += -     self.bdf2 * self.bdf2 * np.dot(self.M, self.un4) + f1

        # calculates self.un0,vn0,an0
        self.u1 = np.linalg.solve(self.LHS, RHS)
        self.v1 = self.bdf0 * self.u1 + self.bdf1 * self.un1 + self.bdf2 * self.un2
        self.a1 = self.bdf0 * self.v1 + self.bdf1 * self.vn1 + self.bdf2 * self.vn2

    def update_structure_time_step(self):
        # update self.un3 un2 un1 vn2 vn1
        self.un4 = self.un3
        self.un3 = self.un2
        self.un2 = self.un1
        self.un1 = self.u1
        self.vn2 = self.vn1
        self.vn1 = self.v1