from source.time_integration_scheme import TimeIntegrationScheme


class GeneralizedAlphaScheme(TimeIntegrationScheme):

    def __init__(self, dt, structure, initial_conditions,  p_inf=0.16):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt

        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)

        # generalized alpha parameters (to ensure unconditional stability, 2nd order accuracy)
        self.alphaM = (2.0 * p_inf - 1.0) / (p_inf + 1.0)
        self.alphaF = p_inf / (p_inf + 1.0)
        self.beta = 0.25 * (1 - self.alphaM + self.alphaF)**2
        self.gamma = 0.5 - self.alphaM + self.alphaF

        # coefficients for LHS
        self.a1h = (1.0 - self.alphaM) / (self.beta * self.dt**2)
        self.a2h = (1.0 - self.alphaF) * self.gamma / (self.beta * self.dt)
        self.a3h = 1.0 - self.alphaF

        # coefficients for mass
        self.a1m = self.a1h
        self.a2m = self.a1h * self.dt
        self.a3m = (1.0 - self.alphaM - 2.0 * self.beta) / (2.0 * self.beta)

        # coefficients for damping
        self.a1b = (1.0 - self.alphaF) * self.gamma / (self.beta * self.dt)
        self.a2b = (1.0 - self.alphaF) * self.gamma / self.beta - 1.0
        self.a3b = (1.0 - self.alphaF) * \
            (0.5 * self.gamma / self.beta - 1.0) * self.dt

        # coefficient for stiffness
        self.a1k = -1.0 * self.alphaF

        # coefficients for velocity update
        self.a1v = self.gamma / (self.beta * self.dt)
        self.a2v = 1.0 - self.gamma / self.beta
        self.a3v = (1.0 - self.gamma / (2 * self.beta)) * self.dt

        # coefficients for acceleration update
        self.a1a = self.a1v / (self.dt * self.gamma)
        self.a2a = -1.0 / (self.beta * self.dt)
        self.a3a = 1.0 - 1.0 / (2.0 * self.beta)

        # structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]
        # initial displacement, velocity and acceleration
        self.u1 = self.u0
        self.v1 = self.v0
        self.a1 = self.a0

        # force from a previous time step (initial force)
        self.f0 = np.dot(self.M, self.a0) + np.dot(self.B,
                                                   self.v0) + np.dot(self.K, self.u0)
        self.f1 = np.dot(self.M, self.a1) + np.dot(self.B,
                                                   self.v1) + np.dot(self.K, self.u1)

        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing Generalized Alpha Method integration scheme setup:")
        print("dt: ", self.dt)
        print("alphaM: ", self.alphaF)
        print("alphaF: ", self.alphaM)
        print("gamma: ", self.gamma)
        print("beta: ", self.beta)
        print(" ")

    def solve_structure(self, f1):

        F = (1.0 - self.alphaF) * f1 + self.alphaF * self.f0

        LHS = self.a1h * self.M + self.a2h * self.B + self.a3h * self.K
        RHS = np.dot(self.M, (self.a1m * self.u0 +
                              self.a2m * self.v0 + self.a3m * self.a0))
        RHS += np.dot(self.B, (self.a1b * self.u0 +
                               self.a2b * self.v0 + self.a3b * self.a0))
        RHS += np.dot(self.a1k * self.K, self.u0) + F

        # update self.f1
        self.f1 = f1

        # updates self.u1,v1,a1
        self.u1 = np.linalg.solve(LHS, RHS)
        self.v1 = self.a1v * (self.u1 - self.u0) + \
            self.a2v * self.v0 + self.a3v * self.a0
        self.a1 = self.a1a * (self.u1 - self.u0) + \
            self.a2a * self.v0 + self.a3a * self.a0

    def update_structure_time_step(self):
        # update displacement, velocity and acceleration
        self.u0 = self.u1
        self.v0 = self.v1
        self.a0 = self.a1

        # update the force
        self.f0 = self.f1
