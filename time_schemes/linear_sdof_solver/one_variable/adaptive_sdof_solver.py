################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import sys
from itertools import cycle
from time_schemes import *

init_printing(use_unicode=True)
cycol = cycle('rbgcmk')
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

class SDoF:

    def __init__(self, K=1.0, M=1.0, C=0.1, f=None, u0=1.0, v0=0.0, a0=0.0, dt=0.01):
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
        self.ta, self.ua, self.va, self.ua, self.va = self.solve("analytical")
        ax1.plot(self.ta, self.ua, label="analytical", c = 'k')
        ax3.plot(self.ta, self.va, label="analytical", c = 'k')


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def predict(self, t, u, v):
        a1 = self.f(t) - self.C * v[-1] - self.K * u[-1] + self.a0
        v1 = v[-1] + a1 *self.dt
        u1 = u[-1] + v1 * self.dt
        u.append(u1)
        v.append(v1)
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
            u_n1, v_n1 = euler_disp_based(self ,t, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf1"):
            u_n1, v_n1 = bdf1_disp_based(self, t, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf2"):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_disp_based(self, t, u_disp[-1], u_disp[-2])
            else:
                u_n1, v_n1 = bdf2_disp_based(self, t, u_disp[-1], u_disp[-2], u_disp[-3], u_disp[-4])

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

        u_vel.append(u_n1)
        v_vel.append(v_n1)


    def solve(self, time_scheme):
        u_disp, v_disp = [], []
        u_vel, v_vel = [], []

        t = 0.0
        t_vec = []

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u_disp, v_disp)
                self.initialize(u_vel, v_vel)
            elif (tstep == 1):
                self.predict(t, u_disp, v_disp)
                self.predict(t, u_vel, v_vel)
            else:
                self.apply_scheme_disp(time_scheme, u_disp, v_disp, t, tstep)
                self.apply_scheme_vel(time_scheme, u_vel, v_vel, t, tstep)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec,  u_disp, v_disp, u_vel, v_vel


    def update_dt(self, t, dt_old):
        if self.eta < self.eta_min: # when the change is small, large time step
            self.dt = self.rho * self.dt
        elif self.eta > self.eta_max: # when the change is large, small time step
            self.dt = self.sigma * self.dt # 0 < sigma < 1

        if self.dt > self.max_dt:
            dt = self.max_dt
        elif self.dt < self.min_dt:
            dt = self.min_dt
        else:
            dt = self.dt

        print("Current dt is: " + str(dt))
        return dt


    def compute_eta(self,vn1,vn):
        # eta is the monitoring function for the choice of time step size
        # see Denner 2.2.1
        self.eta = abs(vn1 - vn)/(abs(vn) + self.epsilon)
