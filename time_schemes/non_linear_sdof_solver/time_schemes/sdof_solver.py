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
from math import *
from sympy import *
import cmath
import sys
from itertools import cycle

init_printing(use_unicode=True)
cycol = cycle('bgrcmk')
fig = plt.figure()

class SDoF:

    def __init__(self, scheme=None, K=1.0, M=1, C=0.2, f=None, u0=1.0, v0=0.0, dt=0.01):
        self.K = K
        self.M = M
        self.C = C
        self.f = lambda t: 0.2
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.scheme = scheme
        self.tend = 80.0
        self.plot_result()


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def update_time(self, t):
        t += self.dt
        return t


    def analytical(self, t):
        # !!! This analytical solution only works for constant f !!!
        if ( (self.C * self.C - 4 * self.K * self.M) > 0):
            Delta = sqrt( self.C * self.C - 4 * self.K * self.M )
        else:
            Delta = cmath.sqrt( self.C * self.C - 4 * self.K * self.M )

        A1 = - ( - self.C * self.f(t) + self.u0 * self.C * self.K + 2 * self.v0 * self.K * self.M \
                + self.f(t) * Delta - self.u0 * self.K * Delta ) / (2 * self.K * Delta)
        A2 = - ( self.C * self.f(t) - self.u0 * self.C * self.K - 2 * self.v0 * self.K * self.M \
                + self.f(t) * Delta - self.u0 * self.K * Delta ) /( 2 * self.K * Delta)

        # source: Wolfram Alpha
        u_new = A1 * cmath.exp( 0.5 * t * (- Delta / self.M  - self.C / self.M )) + \
                A2 * cmath.exp( 0.5 * t * ( Delta / self.M  - self.C / self.M )) + self.f(t) / self.K

        return u_new


    def euler(self, t, un, vn):

        u_new = self.dt * vn + un
        v_new = ( self.M * vn - self.dt * (self.C * vn + self.K * un - self.f(t)) ) / self.M

        return u_new, v_new


    def bdf1(self, t, un, vn):

        C = self.C
        M = self.M
        K = self.K
        f = self.f(t)
        dt = self.dt

        u_new = (dt*(M*vn + dt*f) + un*(C*dt + M))/(C*dt + K*dt**2 + M)
        v_new = (-K*dt*un + M*vn + dt*f)/(C*dt + K*dt**2 + M)

        return u_new, v_new


    def bdf2(self, t, un, vn, unm1, vnm1):

        C = self.C
        M = self.M
        K = self.K
        f = self.f(t)
        dt = self.dt

        u_new = (2.0*dt*(4.0*M*vn - M*vnm1 + 2.0*dt*f) + (4.0*un - unm1)*(2.0*C*dt + 3.0*M))/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)
        v_new = (-8.0*K*dt*un + 2.0*K*dt*unm1 + 12.0*M*vn - 3.0*M*vnm1 + 6.0*dt*f)/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)

        return u_new, v_new


    def rk4(self, t, un, vn):

        C = self.C
        M = self.M
        K = self.K
        f = self.f(t)
        dt = self.dt

        k0 =  dt*vn
        l0 =  dt*(-C*vn - K*un + f)/M
        k1 =  dt*(0.5*l0 + vn)
        l1 =  dt*(-C*(0.5*l0 + vn) - K*(0.5*k0 + un) + f)/M
        k2 =  dt*(0.5*l1 + vn)
        l2 =  dt*(-C*(0.5*l1 + vn) - K*(0.5*k1 + un) + f)/M
        k3 =  dt*(l2 + vn)
        l3 =  dt*(-C*(l2 + vn) - K*(k2 + un) + f)/M
        u_new =  k0/6 + k1/3 + k2/3 + k3/6 + un
        v_new =  l0/6 + l1/3 + l2/3 + l3/6 + vn

        return u_new, v_new


    def solve(self):
        u, v = [], []
        t = 0.0

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
            else:
                if (self.scheme == "analytical"):
                    u_new = self.analytical(t)
                    v_new = 0.0

                if (self.scheme == "euler"):
                    u_new, v_new = self.euler(t, u[-1], v[-1])

                if (self.scheme == "bdf1"):
                    u_new, v_new = self.bdf1(t, u[-1], v[-1])

                if (self.scheme == "bdf2"):
                    if (tstep == 1):
                        u_new, v_new = self.bdf1(t, u[-1], v[-1])
                    else:
                        u_new, v_new = self.bdf2(t, u[-1], v[-1], u[-2], v[-2])

                if (self.scheme == "rk4"):
                    u_new, v_new = self.rk4(t, u[-1], v[-1])

                print("u_new: ", u_new)

                u.append(u_new)
                v.append(v_new)

            t = self.update_time(t)

        return u


    def solve_averaged(self):
        u_ave, v_ave = [], []
        t = 0.0

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u_ave, v_ave)
            else:
                n = tstep
                A = lambda n : 1 / n
                B = lambda n : (n - 1) / n
                if tstep == 1:
                    u_n = u_ave[-1]
                    v_n = v_ave[-1]

                else:
                    u_n = (u_ave[-1] - B(n) * u_ave[-2]) / A(n)
                    v_n = (v_ave[-1] - B(n) * v_ave[-2]) / A(n)

                if (self.scheme == "analytical"):
                    u_n1 = self.analytical(t)
                    v_n1 = 0.0

                if (self.scheme == "euler"):
                    u_n1, v_n1 = self.euler(t, u_n, v_n)

                if (self.scheme == "bdf1"):
                    u_n1, v_n1 = self.bdf1(t, u_n, v_n)

                if (self.scheme == "bdf2"):
                    if (tstep == 1):
                        u_n1, v_n1 = self.bdf1(t, u_n, v_n)

                    else:
                        if(tstep == 2):
                            u_nm1 = u_ave[-2]
                            v_nm1 = v_ave[-2]
                        else:
                            u_nm1 = (u_ave[-2] - B(n-1) * u_ave[-3]) / A(n-1)
                            v_nm1 = (v_ave[-2] - B(n-1) * v_ave[-3]) / A(n-1)

                        u_n1, v_n1 = self.bdf2(t, u_n, v_n, u_nm1, v_nm1)

                if (self.scheme == "rk4"):
                    u_n1, v_n1 = self.rk4(t, u_n, v_n)

                print("u_n1: ", u_n1)

                u_bar_n1 = A(n+1) * u_n1 + B(n+1) * u_ave[-1]
                v_bar_n1 = A(n+1) * v_n1 + B(n+1) * v_ave[-1]

                u_ave.append(u_bar_n1)
                v_ave.append(v_bar_n1)

            t = self.update_time(t)

        return u_ave


    def average_result(self, u):
        u_ave = []
        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            if tstep == 0:
                u_ave.append(u[tstep])
                u_sum = u[tstep]
            else:
                n = tstep + 1
                u_sum += u[tstep]
                u_bar_n1 = u_sum / n
                u_ave.append(u_bar_n1)

        return u_ave


    def error_estimation(self, t, ua, u):
        e =[]
        for i in range (0,len(ua)-1):
            e.append(ua[i] - ua[i])
        print("Error at time: ", t[i], " = ", e[i])


    def plot_result(self):
        global fig, cycol

        plt.title('Mu\'\'(t) + Cu\'(t) + Ku(t) = f(t), for u(0) = 1, v(0) = 0')
        plt.grid(True)

        nsteps = self.tend/self.dt
        t = np.linspace(0.0, self.tend, nsteps)

        u = self.solve()
        ua = self.average_result(u)
        ua_solved = self.solve_averaged()
        self.error_estimation(t, ua, ua_solved)

        color = next(cycol)
        plt.plot(t, u, label=self.scheme, c = color)
        plt.plot(t, ua, label=self.scheme + ' averaged', c = color, ls='--')
        plt.plot(t, ua_solved, label=self.scheme + ' solve averaged', c = color, ls='None', marker='x',  markersize=5)
        plt.legend()


# Check number of command line arguments
if len(sys.argv) < 2:
	print ("Usage: python sdof_solver.py <scheme> <scheme>")
	sys.exit(1)

if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        # Get command line arguments
        scheme = sys.argv[i]
        my_sdof = SDoF(scheme)
    plt.show()


