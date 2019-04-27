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
from adaptive_time_step_schemes import analytical_general, euler, bdf1, bdf2, rk4

class SDoF:

    def __init__(self, scheme=None, K=1.0, M=1, C=0.1, f=None, u0=1.0, v0=0.0, dt=0.01):
        self.K = K
        self.M = M
        self.C = C
        self.f = lambda t: 0.2
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.max_dt = 10 * self.dt
        self.min_dt = 0.5 * self.dt
        time_scheme = scheme
        self.tend = 20.0
        self.ta, self.ua, self.va = self.solve("analytical")


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def update_time(self, t, dt):
        t += dt
        return t
    

    def update_dt(self, t, dt_old):
        dt = dt_old * 1.001

        if dt > self.max_dt:
            dt = self.max_dt
        print("Current dt is: " + str(dt))
        return dt


    def apply_scheme(self, time_scheme, u, v, t, dt, old_dt, tstep):
        if (time_scheme== "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            u_n1, v_n1 = euler(self ,t, dt, u[-1], v[-1])

        if (time_scheme == "bdf1"):
            u_n1, v_n1 = bdf1(self, t, dt, u[-1], v[-1])

        if (time_scheme == "bdf2"):
            if (tstep == 1):
                u_n1, v_n1 = bdf1(self, t, dt, u[-1], v[-1])
            else:
                u_n1, v_n1 = bdf2(self, t, dt, old_dt, u[-1], v[-1], u[-2], v[-2])

        if (time_scheme == "rk4"):
            u_n1, v_n1 = rk4(self, t, u[-1], v[-1])

        u.append(u_n1)
        v.append(v_n1)


    def solve(self, time_scheme):
        u, v = [], []
        t = 0.0
        dt = []
        t_vec = []
        tstep = 0
        delta_time = self.dt
        old_delta_time = self.dt

        while t < self.tend:
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
            else:
                self.apply_scheme(time_scheme, u, v, t, delta_time, old_delta_time, tstep)

            t_vec.append(t)
            dt.append(delta_time)
            tstep += 1
            t = self.update_time(t, delta_time)
            old_delta_time = delta_time
            delta_time = self.update_dt(t, dt[-1])


        return t_vec, u, v


    def error_estimation(self, t, ua, u):
        e =[]
        for i in range (0,len(ua)):
            e.append(u[i] - ua[i])
        return e


if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        # Check number of command line arguments
        if len(sys.argv) < 2:
            print ("Usage: python sdof_solver.py <scheme> <scheme>")
            sys.exit(1)
        else:
            my_sdof = SDoF()
            # Get command line arguments
            scheme = sys.argv[i]
            my_sdof.plot_result(scheme)
    plt.show()


