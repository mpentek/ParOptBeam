################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f rewrite 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = - C * v(t) - K / M * u(t)
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import numpy as np
from sympy import *
from time_schemes import analytical_general, euler, bdf1, bdf2, rk4

class SDoF:

    def __init__(self, scheme=None, K=1.0, M=1, C=0.5, f=None, u0=1.0, v0=0.0, dt=0.01):
        self.K = K
        self.M = M
        self.C = C
        self.f = lambda t: 0.2
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        time_scheme = scheme
        self.tend = 20.0
        self.ua, self.va = [],[]


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def update_time(self, t):
        t += self.dt
        return t


    def apply_scheme(self, time_scheme, u, v, t, tstep):
        if (time_scheme== "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            u_n1, v_n1 = euler(self ,t, u[-1], v[-1])

        if (time_scheme == "bdf1"):
            u_n1, v_n1 = bdf1(self, t, u[-1], v[-1])

        if (time_scheme == "bdf2"):
            if (tstep == 1):
                u_n1, v_n1 = bdf1(self, t, u[-1], v[-1])
            else:
                u_n1, v_n1 = bdf2(self, t, u[-1], v[-1], u[-2], v[-2])

        if (time_scheme == "rk4"):
            u_n1, v_n1 = rk4(self, t, u[-1], v[-1])

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)

        u.append(u_n1)
        v.append(v_n1)


    def solve(self, time_scheme):
        u, v = [], []
        t = 0.0
        t_vec = []

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
                self.initialize(self.ua, self.va)
            else:
                self.apply_scheme(time_scheme, u, v, t, tstep)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec, u, v


    def error_estimation(self, t, u, v):
        eu = []
        ev = []
        for i in range (0,len(u)):
            eu.append(u[i] - self.ua[i])
            ev.append(v[i] - self.va[i])
        return eu, ev


