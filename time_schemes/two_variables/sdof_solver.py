################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f rewrite 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = - C * v(t) - K / M * u(t)
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import sys
from itertools import cycle
from time_schemes import analytical_general, euler, bdf1, bdf2, rk4

init_printing(use_unicode=True)
cycol = cycle('bgrcmk')
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

class SDoF:

    def __init__(self, scheme=None, K=1.0, M=1, C=0.1, f=None, u0=1.0, v0=0.0, dt=0.01):
        self.K = K
        self.M = M
        self.C = C
        self.f = lambda t: 0.2
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        time_scheme = scheme
        self.tend = 20.0
        self.ta, self.ua, self.va = self.solve("analytical")
        ax1.plot(self.ta, self.ua, label="analytical", c = 'k')
        ax3.plot(self.ta, self.va, label="analytical", c = 'k')


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
            else:
                self.apply_scheme(time_scheme, u, v, t, tstep)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec, u, v


    def error_estimation(self, t, ua, u):
        e =[]
        for i in range (0,len(ua)):
            e.append(u[i] - ua[i])
        return e


    def plot_result(self, time_scheme):
        global fig1, fig2, cycol

        color = next(cycol)

        t, u, v = self.solve(time_scheme)
        eu = self.error_estimation(t, self.ua, u)
        ev = self.error_estimation(t, self.va, v)


        ax1.set_title('Mu\'\'(t) + Cu\'(t) + Ku(t) = f(t), for u(0) = 1, v(0) = 0 \n DISPLACEMENT PLOT')
        ax1.set_xlabel("t")
        ax1.set_ylabel("u(t)")
        ax1.grid('on')
        ax1.plot(t, u, label=time_scheme + ' two variables', c = color)
        ax1.legend()

        ax2.set_title('Error Estimation \n DISPLACEMENT')
        ax2.set_xlabel("t")
        ax2.set_ylabel("e(t)")
        ax2.grid('on')
        ax2.plot(t, eu, label=time_scheme + ' two variables', c = color)
        ax2.legend()

        ax3.set_title('Mu\'\'(t) + Cu\'(t) + Ku(t) = f(t), for u(0) = 1, v(0) = 0 \n VELOCITY PLOT')
        ax3.set_xlabel("t")
        ax3.set_ylabel("v(t)")
        ax3.grid('on')
        ax3.plot(t, v, label=time_scheme + ' two variables', c = color)
        ax3.legend()

        ax4.set_title('Error Estimation \n VELOCITY')
        ax4.set_xlabel("t")
        ax4.set_ylabel("e(t)")
        ax4.grid('on')
        ax4.plot(t, ev, label=time_scheme + ' two variables', c = color)
        ax4.legend()


# Check number of command line arguments
if len(sys.argv) < 2:
	print ("Usage: python sdof_solver.py <scheme> <scheme>")
	sys.exit(1)

if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        my_sdof = SDoF()
        # Get command line arguments
        scheme = sys.argv[i]
        my_sdof.plot_result(scheme)
    plt.show()


