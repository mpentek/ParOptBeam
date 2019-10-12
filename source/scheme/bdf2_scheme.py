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
        self.bdf0 = 3 * 0.5 / self.dt
        self.bdf1 = -4 * 0.5 / self.dt
        self.bdf2 = 1 * 0.5 / self.dt

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
        self.an3 = self.a0
        self.vn3 = self.v0
        self.un3 = self.u0
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

        LHS = np.dot(self.B, self.bdf0) + self.K + np.dot(self.M, self.bdf0 * self.bdf0)
        RHS = - np.dot(self.B, self.bdf1 * self.un1) - np.dot(self.B, np.dot(self.bdf2, self.un2))
        RHS += - 2 * np.dot(self.M, np.dot(self.bdf0 * self.bdf1, self.un1))
        RHS += - 2 * np.dot(self.M, np.dot(self.bdf0 * self.bdf2, self.un2))
        RHS += - np.dot(self.M, self.bdf1 * self.bdf1 * self.un2)
        RHS += - 2 * np.dot(self.M, self.bdf1 * self.bdf2 * self.un2)
        RHS += - np.dot(self.M, np.dot(self.bdf2 * self.bdf2, self.un3)) + f1

        # calculates self.un0,vn0,an0
        self.un0 = np.linalg.solve(LHS, RHS)
        self.vn0 = self.bdf0 * self.un0 + self.bdf1 * self.un1 + self.bdf2 * self.un2
        self.an0 = self.bdf0 * self.vn0 + self.bdf1 * self.vn1 + self.bdf2 * self.vn2

        # update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0
        # print(self.u1)

    def update_structure_time_step(self):
        # update self.un3 un2 un1 vn2 vn1
        self.un3 = self.un2
        self.un2 = self.un1
        self.un1 = self.un0
        self.vn2 = self.vn1
        self.vn1 = self.vn0


def test():
    M = [[1, 0], [0, 1]]
    B = [[1, 0], [0, 1]]
    K = [[1, 0], [0, 1]]
    u0 = [0.0, 1.0]
    v0 = [0.0, 0.0]
    a0 = [0.0, 0.0]
    dt = 0.1
    f1 = [0.0, 1.0]
    solver = BDF2(dt, [M, B, K], [u0, v0, a0])
    solver.solve_structure(f1)


if __name__ == "__main__":
    test()


