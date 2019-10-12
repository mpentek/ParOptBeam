################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

############# ONE VARIABLE FORMULATION ################

import numpy as np
from sympy import *
from time_schemes import analytical_general, euler_disp_based, euler_vel_based, bdf1_disp_based, bdf1_vel_based, \
    bdf2_disp_based, bdf2_vel_based


# DISPLCAMENT_BASED = True -> displacement based formulation
# VELOCITY_BASED = True -> velocity based formulation

class SDoF:

    def __init__(self, DISPLCAMENT_BASED=None, scheme=None, K=1.0, M=1.0, C=0.1, f=None, u0=1.0, v0=0.0, a0=0.0,
                 dt=0.01):
        self.K = K
        self.M = M
        self.C = C
        self.f = lambda t: 0.0
        self.fstr = "0.0"
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0
        self.dt = dt
        self.tend = 20.0
        self.ua, self.va = [], []
        self.use_disp_form = DISPLCAMENT_BASED
        time_scheme = scheme

    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)

    def predict(self, t, u, v):
        print("predicting")
        a1 = self.f(t) - self.C * v[-1] - self.K * u[-1] + self.a0
        v1 = v[-1] + a1 * self.dt
        u1 = u[-1] + v1 * self.dt
        u.append(u1)
        v.append(v1)

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)

        return u1, v1

    def update_time(self, t):
        t += self.dt
        return t

    def apply_scheme_disp(self, time_scheme, u_disp, v_disp, t, tstep):
        ##### DISPLACEMENT BASED SCHEMES #####
        u_n1 = 0
        v_n1 = 0
        if (time_scheme == "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            u_n1, v_n1 = euler_disp_based(self, t, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf1"):
            u_n1, v_n1 = bdf1_disp_based(self, t, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf2"):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_disp_based(self, t, u_disp[-1], u_disp[-2])
            else:
                u_n1, v_n1 = bdf2_disp_based(self, t, u_disp[-1], u_disp[-2], u_disp[-3], u_disp[-4])

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)

        u_disp.append(u_n1)
        v_disp.append(v_n1)

    def apply_scheme_vel(self, time_scheme, u_vel, v_vel, t, tstep):

        ##### VELOCITY BASED SCHEMES #####
        if (time_scheme == "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            v_n1, u_n1 = euler_vel_based(self, t, v_vel[-1], v_vel[-2], u_vel[-1])

        if (time_scheme == "euler2"):
            if tstep == 2:
                u_n1, v_n1 = self.predict(t, u_vel, v_vel)
                u_vel.pop()
                v_vel.pop()
            else:
                v_n1, u_n1 = euler_vel_based2(self, t, v_vel[-1], v_vel[-2], v_vel[-3], u_vel[-1])

        if (time_scheme == "bdf1"):
            v_n1, u_n1 = bdf1_vel_based(self, t, v_vel[-1], v_vel[-2], u_vel[-1])

        if (time_scheme == "bdf2"):
            if (tstep == 2 or tstep == 3):
                v_n1, u_n1 = bdf1_vel_based(self, t, v_vel[-1], v_vel[-2], u_vel[-1])
            else:
                v_n1, u_n1 = bdf2_vel_based(self, t, v_vel[-1], v_vel[-2], v_vel[-3], u_vel[-1], u_vel[-2])

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)

        u_vel.append(u_n1)
        v_vel.append(v_n1)

    def solve(self, time_scheme):
        u, v = [], []

        t = 0.0
        t_vec = []

        nsteps = int(self.tend / self.dt)

        for tstep in range(0, nsteps):
            print("time step: ", tstep)

            if self.use_disp_form == True:
                if (tstep == 0):
                    self.initialize(u, v)
                    self.initialize(self.ua, self.va)
                elif (tstep == 1):
                    self.predict(t, u, v)
                else:
                    self.apply_scheme_disp(time_scheme, u, v, t, tstep)

            else:
                if (tstep == 0):
                    self.initialize(u, v)
                    self.initialize(self.ua, self.va)
                elif (tstep == 1):
                    self.predict(t, u, v)
                else:
                    self.apply_scheme_vel(time_scheme, u, v, t, tstep)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec, u, v

    def error_estimation(self, t, u, v):
        eu = []
        ev = []
        for i in range(0, len(u)):
            eu.append(u[i] - self.ua[i])
            ev.append(v[i] - self.va[i])
        return eu, ev
