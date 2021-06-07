import numpy as np

from source.solving_strategies.schemes.time_integration_scheme import TimeIntegrationScheme


class GeneralizedAlphaScheme(TimeIntegrationScheme):

    def __init__(self, dt, comp_model, initial_conditions, p_inf=0.16):
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        super().__init__(dt, comp_model, initial_conditions)

        # generalized alpha parameters (to ensure unconditional stability, 2nd order accuracy)
        self.alphaM = (2.0 * p_inf - 1.0) / (p_inf + 1.0)
        #print("alphaM: ", self.alphaM)

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
        if self.M.ndim == 2:
            #print('System: in matrix form')
            self.f0 = np.dot(self.M, self.a0) + np.dot(self.B,
                                                    self.v0) + np.dot(self.K, self.u0)
            self.f1 = np.dot(self.M, self.a1) + np.dot(self.B,
                                                    self.v1) + np.dot(self.K, self.u1)
        
        elif self.M.ndim == 1:
            print('System: in vector (from diagonal matrix) or scalar form')
            self.f0 = self.M * self.a0 + self.B * self.v0 + self.K * self.u0
            self.f1 = self.M * self.a1 + self.B * self.v1 + self.K * self.u1

        else:
            raise Exception('Dimension of system parameters is GeneralizedAlphaScheme is wrong')
        
        #self._print_time_integration_setup()

    def _print_time_integration_setup(self):
        print("Printing Generalized Alpha Method integration scheme setup:")
        print("dt: ", self.dt)
        print("alphaM: ", self.alphaM)
        print("alphaF: ", self.alphaF)
        print("gamma: ", self.gamma)
        print("beta: ", self.beta)
        print(" ")

    def predict_velocity(self, u1):
        v1 = self.a1v * (u1 - self.un1) + self.a2v * \
            self.vn1 + self.a3v * self.an1
        return v1

    # TODO: GenAlpha needs to take u1 as input, which is not general
    def predict_acceleration(self, v1):
        a1 = self.a1a * (self.u1 - self.un1) + self.a2a * \
            self.vn1 + self.a3a * self.an1
        return a1

    def solve_single_step(self, f1):
        # LHS needs to be updated in case of non-linear elements
        LHS = self.a1h * self.M + self.a2h * self.B + self.a3h * self.K

        F = (1.0 - self.alphaF) * f1 + self.alphaF * self.f0

        if self.M.ndim == 2:
            # system: in matrix form
            RHS = np.dot(self.M, (self.a1m * self.un1 +
                                self.a2m * self.vn1 + self.a3m * self.an1))
            RHS += np.dot(self.B, (self.a1b * self.un1 +
                                self.a2b * self.v0 + self.a3b * self.an1))
            RHS += np.dot(self.a1k * self.K, self.un1) + F

            # main solve
            self.u1 = np.linalg.solve(LHS, RHS)

        elif self.M.ndim == 1:
            # system: in vector (from diagonal matrix) or scalar form
            RHS = self.M * (self.a1m * self.un1 +
                                self.a2m * self.vn1 + self.a3m * self.an1)
            RHS += self.B * (self.a1b * self.un1 +
                                self.a2b * self.v0 + self.a3b * self.an1)
            RHS += (self.a1k * self.K) * self.un1 + F

            # main solve
            self.u1 = RHS/LHS

        else:
            raise Exception('Dimension of system parameters is GeneralizedAlphaScheme is wrong')
        
        # updates self.v1,a1,f1
        self.v1 = self.predict_velocity(self.u1)
        self.a1 = self.predict_acceleration(self.v1)
        self.f1 = f1

    def update(self):
        # update displacement, velocity and acceleration
        self.un1 = self.u1
        self.vn1 = self.v1
        self.an1 = self.a1

        # update the force
        self.f0 = self.f1
