import numpy as np

from source.scheme.time_integration_scheme import TimeIntegrationScheme


class RungeKutta4(TimeIntegrationScheme):
    """
    (Explicit) Runge Kutta 4th order approximation

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

        # inverse matrix
        self.InvM = np.linalg.inv(self.M)

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
        print("Printing Runge Kutta 4th order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):

        # calculates self.un0,vn0,an0

        k0 = self.dt * self.vn1
        l0 = self.dt * np.dot(self.InvM, (-np.dot(self.B, self.vn1) - np.dot(self.K, self.un1) + f1))
        k1 = self.dt * (0.5*l0 + self.vn1)
        l1 = self.dt * np.dot(self.InvM, (-np.dot(self.B, (0.5*l0 + self.vn1)) - np.dot(self.K, (0.5*k0 + self.un1)) + f1))
        k2 = self.dt * (0.5*l1 + self.vn1)
        l2 = self.dt * np.dot(self.InvM, (-np.dot(self.B, (0.5*l1 + self.vn1)) - np.dot(self.K, (0.5*k1 + self.un1)) + f1))
        k3 = self.dt * (l2 + self.vn1)
        l3 = self.dt * np.dot(self.InvM, (-np.dot(self.B, (l2 + self.vn1)) - np.dot(self.K, (k2 + self.un1)) + f1))

        self.un0 = self.un1 + (k0 + 2*(k1 + k2) + k3)/6.0
        self.vn0 = self.vn1 + (l0 + 2*(l1 + l2) + l3)/6.0
        self.an0 = (l0 + 2*(l1 + l2) + l3)/6.0/self.dt

        # update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un1 = self.un0
        self.vn1 = self.vn0
        self.an1 = self.an0


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
    solver = RungeKutta4(dt, [M, B, K], [u0, v0, a0])

    for i in range(1, 100):

        solver.solve_structure(f1)

        # appending results to the list
        displacement[:, i] = solver.get_displacement()

        # update results
        solver.update_structure_time_step()

    print(displacement[:, -1])
