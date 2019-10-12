import numpy as np

from source.scheme.time_integration_scheme import TimeIntegrationScheme


class Euler12(TimeIntegrationScheme):
    """
    Euler 1st and 2nd order 

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


def test():
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[0.0, 0.0], [0.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    u0 = np.array([0.0, 1.0])
    v0 = np.array([0.0, 0.0])
    a0 = np.array([0.0, 0.0])
    dt = 0.1
    f1 = np.array([0.0, 0.0])
    displacement = np.empty([2, 100])
    solver = Euler12(dt, [M, B, K], [u0, v0, a0])

    for i in range(1, 100):

        solver.solve_structure(f1)

        # appending results to the list
        displacement[:, i] = solver.get_displacement()

        # update results
        solver.update_structure_time_step()

    print(displacement[:, -1])
