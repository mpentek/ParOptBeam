################################################################################################
###   M * u''(t) + C * u'(t) + K(u) * u(t) = f rewrite nonlinear 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = ( f - C * v(t) - K * u(t) ) / M = f(t, u, v) = rhs
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

    def __init__(self, time_scheme=None, numerical_scheme=None, K=1.0, M=1.0, C=0.2, f=None, u0=1.0, v0=0.0, dt=0.01):

        self.K = lambda u: (u**2 + 1)
        self.Ku = lambda u: self.K(u) * u
        self.M = M
        self.C = C
        self.f = lambda t: 0.0
        self.fstr = '0.0'
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.time_scheme = time_scheme
        self.numerical_scheme = numerical_scheme
        self.tend = 20.0
        self.plot_result()


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def calculate_residual(self, time_scheme, t, u_n, u_n1, v_n, v_n1, u_nm1=None, v_nm1=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            ru = v_n - (-u_n + u_n1)/dt
            rv = -C*v_n - self.K(u_n)*u_n - M*(-v_n + v_n1)/dt + f

        if time_scheme == 'bdf1':
            ru = v_n1 - (-u_n + u_n1)/dt

            if self.numerical_scheme == 'Newton Raphson':
                rv = -C*v_n1 - self.K(u_n1)*u_n1 - M*(-v_n + v_n1)/dt + f
            elif self.numerical_scheme == 'Picard':
                rv = -C*v_n1 - self.K(u_n)*u_n1 - M*(-v_n + v_n1)/dt + f

        if time_scheme == 'bdf2':
            ru = v_n1 + 2*u_n/dt - 3*u_n1/(2*dt) - u_nm1/(2*dt)

            if self.numerical_scheme == 'Newton Raphson':
                rv = -C*v_n1 - M*(-2*v_n/dt + 3*v_n1/(2*dt) + v_nm1/(2*dt)) + f - self.K(u_n1)*u_n1
            elif self.numerical_scheme == 'Picard':
                rv = -C*v_n1 - M*(-2*v_n/dt + 3*v_n1/(2*dt) + v_nm1/(2*dt)) + f - self.K(u_n)*u_n1

        if time_scheme == 'rk4':
            ru0 =  v_n
            rv0 =  (-C*v_n + f - u_n*(u_n**2 + 1))/M
            ru1 =  0.5*dt*rv0 + v_n
            rv1 =  (-C*(0.5*dt*rv0 + v_n) + f - (0.5*dt*ru0 + u_n)*((0.5*dt*ru0 + u_n)**2 + 1))/M
            ru2 =  0.5*dt*rv1 + v_n
            rv2 =  (-C*(0.5*dt*rv1 + v_n) + f - (0.5*dt*ru1 + u_n)*((0.5*dt*ru1 + u_n)**2 + 1))/M
            ru3 =  dt*rv2 + v_n
            rv3 =  (-C*(dt*rv2 + v_n) + f - (dt*ru2 + u_n)*((dt*ru2 + u_n)**2 + 1))/M

            ru =  -ru0/6 - ru1/3 - ru2/3 - ru3/6 + (-u_n + u_n1)/dt
            rv =  -rv0/6 - rv1/3 - rv2/3 - rv3/6 + (-v_n + v_n1)/dt
        return ru, rv


    def calculate_increment(self, time_scheme, t, ru, rv, u_n1=None, v_n1=None, u_n=None, v_n=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            du = dt * ru
            dv = dt * rv/M

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                du =  dt*(C*dt*ru + M*ru + dt*rv)/(C*dt + M + 3*dt**2*u_n1**2 + dt**2)
                dv = -dt*(dt*ru*(3*u_n1**2 + 1) - rv)/(C*dt + M + dt**2*(3*u_n1**2 + 1))
            elif self.numerical_scheme == 'Picard':
                du = dt*(C*dt*ru + M*ru + dt*rv)/(C*dt + M + dt**2*u_n**2 + dt**2)
                dv = -dt*(dt*ru*(u_n**2 + 1) - rv)/(C*dt + M + dt**2*(u_n**2 + 1))

        if time_scheme == 'bdf2':
            if self.numerical_scheme == 'Newton Raphson':
                du = 2*dt*(2*C*dt*ru + 3*M*ru + 2*dt*rv)/(6*C*dt + 9*M + 12*dt**2*u_n1**2 + 4*dt**2)
                dv = -2*dt*(2*dt*ru*(3*u_n1**2 + 1) - 3*rv)/(6*C*dt + 9*M + 4*dt**2*(3*u_n1**2 + 1))
            elif self.numerical_scheme == 'Picard':
                du = 2*dt*(2*C*dt*ru + 3*M*ru + 2*dt*rv)/(6*C*dt + 9*M + 4*dt**2*u_n**2 + 4*dt**2)
                dv = -2*dt*(2*dt*ru*(u_n**2 + 1) - 3*rv)/(6*C*dt + 9*M + 4*dt**2*(u_n**2 + 1))

        if time_scheme == 'rk4':
                du =  -dt*ru
                dv =  -dt*rv
        return du, dv


    # Function for the u'(t) = v ODE.
    def g(self, v):
        return v


    def update_time(self, t):
        t += self.dt
        return t


    def euler(self, t, u_n, v_n):
        C = self.C
        M = self.M
        K = self.K(u_n)
        f = self.f(t)
        dt = self.dt

        u_n1 = dt * v_n + u_n
        v_n1 = ( M * v_n - dt * (C * v_n + K * u_n - f) ) / M

        return u_n1, v_n1


    def bdf1(self, t, u_n, v_n):

        C = self.C
        M = self.M
        K = self.K(u_n)
        f = self.f(t)
        dt = self.dt

        u_n1 = (dt*(M*v_n + dt*f) + u_n*(C*dt + M))/(C*dt + K*dt**2 + M)
        v_n1 = (-K*dt*u_n + M*v_n + dt*f)/(C*dt + K*dt**2 + M)

        return u_n1, v_n1


    def bdf2(self, t, u_n, v_n, u_nm1, v_nm1):

        C = self.C
        M = self.M
        K = self.K(u_n)
        f = self.f(t)
        dt = self.dt

        u_n1 = (2.0*dt*(4.0*M*v_n - M*v_nm1 + 2.0*dt*f) + (4.0*u_n - u_nm1)*(2.0*C*dt + 3.0*M))/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)
        v_n1 = (-8.0*K*dt*u_n + 2.0*K*dt*u_nm1 + 12.0*M*v_n - 3.0*M*v_nm1 + 6.0*dt*f)/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)

        return u_n1, v_n1


    def rk4(self, t, u_n, v_n):

        C = self.C
        M = self.M
        #K = self.K(u_n)
        f = self.f(t)
        dt = self.dt

        k0 =  dt*v_n
        l0 =  dt*(-C*v_n - self.K(u_n)*u_n + f)/M
        k1 =  dt*(0.5*l0 + v_n)
        l1 =  dt*(-C*(0.5*l0 + v_n) - self.K(0.5*k0 + u_n)*(0.5*k0 + u_n) + f)/M
        k2 =  dt*(0.5*l1 + v_n)
        l2 =  dt*(-C*(0.5*l1 + v_n) - self.K(0.5*k1 + u_n)*(0.5*k1 + u_n) + f)/M
        k3 =  dt*(l2 + v_n)
        l3 =  dt*(-C*(l2 + v_n) - self.K(k2 + u_n)*(k2 + u_n) + f)/M

        u_n1 =  k0/6 + k1/3 + k2/3 + k3/6 + u_n
        v_n1 =  l0/6 + l1/3 + l2/3 + l3/6 + v_n

        return u_n1, v_n1


    def solve(self):
        u, v = [], []
        t = 0.0

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
                ru, rv = 0.0, 0.0
            else:
                u_n1, v_n1 = self.get_first_iteration_step(t, tstep, u, v)

                if (self.time_scheme == 'bdf2' and tstep == 1):
                    ru, rv = self.calculate_residual('bdf1', t, u[-1], u_n1, v[-1], v_n1)
                elif (self.time_scheme == 'bdf2' and tstep > 1):
                    ru, rv = self.calculate_residual(self.time_scheme, t, u[-1], u_n1, v[-1], v_n1, u[-2], v[-2])
                else:
                    ru, rv = self.calculate_residual(self.time_scheme, t, u[-1], u_n1, v[-1], v_n1)

                print("ru: ", ru)
                print("rv: ", rv)

                u_n1, v_n1 = self.iterate_in_one_time_step(rv, ru, u_n1, v_n1, t, tstep, u, v)

                u.append(u_n1)
                v.append(v_n1)

            t = self.update_time(t)

        return u


    def get_first_iteration_step(self, t, tstep, u, v):
        print(self.time_scheme)
        if (self.time_scheme == 'euler'):
            u_n1, v_n1 = self.euler(t, u[-1], v[-1])

        if (self.time_scheme == 'bdf1'):
            u_n1, v_n1 = self.bdf1(t, u[-1], v[-1])

        if (self.time_scheme == 'bdf2'):
            if (tstep == 1):
                u_n1, v_n1 = self.bdf1(t, u[-1], v[-1])
            else:
                u_n1, v_n1 = self.bdf2(t, u[-1], v[-1], u[-2], v[-2])

        if (self.time_scheme == 'rk4'):
            u_n1, v_n1 = self.rk4(t, u[-1], v[-1])

        return u_n1, v_n1


    def iterate_in_one_time_step(self, rv, ru, u_n1, v_n1, t, tstep, u, v):
        it = 0
        while ( abs(ru) >= 1.0e-12 or abs(rv) >= 1.0e-12) and it < 10:
            if (self.time_scheme == 'bdf2' and tstep == 1):
                du, dv = self.calculate_increment('bdf1', t, ru, rv, u_n1)
            else:
                du, dv = self.calculate_increment(self.time_scheme, t, ru, rv, u_n1, v_n1, u[-1], v[-1])

            u_n1 += du
            v_n1 += dv

            if (self.time_scheme == 'bdf2' and tstep == 1):
                ru, rv = self.calculate_residual('bdf1', t, u[-1], u_n1, v[-1], v_n1)
            elif (self.time_scheme == 'bdf2' and tstep > 1):
                ru, rv = self.calculate_residual(self.time_scheme, t, u[-1], u_n1, v[-1], v_n1, u[-2], v[-2])
            else:
                ru, rv = self.calculate_residual(self.time_scheme, t, u[-1], u_n1, v[-1], v_n1)

            print("ru: ", ru)
            print("rv: ", rv)

            it += 1
        print("Number of iteration per step: ", it)

        return u_n1, v_n1


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

                u = [u_n]
                v = [v_n]

                if (self.time_scheme == 'bdf2'):
                    if (tstep == 1):
                        pass
                    else:
                        if(tstep == 2):
                            u_nm1 = u_ave[-2]
                            v_nm1 = v_ave[-2]
                        else:
                            u_nm1 = (u_ave[-2] - B(n-1) * u_ave[-3]) / A(n-1)
                            v_nm1 = (v_ave[-2] - B(n-1) * v_ave[-3]) / A(n-1)

                        u = [u_nm1, u_n]
                        v = [v_nm1, v_n]

                u_n1, v_n1 = self.get_first_iteration_step(t, tstep, u, v)

                if (self.time_scheme == 'bdf2' and tstep == 1):
                    ru, rv = self.calculate_residual('bdf1', t, u_n, u_n1, v_n, v_n1)
                elif (self.time_scheme == 'bdf2' and tstep > 1):
                    ru, rv = self.calculate_residual(self.time_scheme, t, u_n, u_n1, v_n, v_n1, u_nm1, v_nm1)
                else:
                    ru, rv = self.calculate_residual(self.time_scheme, t, u[-1], u_n1, v[-1], v_n1)

                print("ru: ", ru)
                print("rv: ", rv)

                u_n1, v_n1 = self.iterate_in_one_time_step(rv, ru, u_n1, v_n1, t, tstep, u, v)

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


    def plot_result(self):
        global fig, cycol
        plt.title('Mu\'\'(t) + ' + str(self.C) + 'u\'(t) + K(u)u(t) = ' + str(self.fstr) + ', for u(0) = 1, v(0) = 0')
        plt.grid(True)

        nsteps = self.tend/self.dt
        t = np.linspace(0.0, self.tend, nsteps)
        u = self.solve()
        ua = self.average_result(u)
        ua_solved = self.solve_averaged()
        color = next(cycol)
        plt.plot(t, u, label=self.time_scheme + ' with ' + self.numerical_scheme, c = color)
        plt.plot(t, ua, label=self.time_scheme + ' with ' + self.numerical_scheme + ' averaged', c = color, ls='--')
        plt.plot(t, ua_solved, label=self.time_scheme + ' with ' + self.numerical_scheme + ' solve averaged', c = color, ls='None', marker='x',  markersize=5)
        plt.legend()


# Check number of command line arguments
if len(sys.argv) < 3:
	print ("Usage: python sdof_solver.py <numerica scheme for all> <time scheme> <time scheme>...")
	sys.exit(1)

if __name__ == "__main__":

    for i in range(2, len(sys.argv)):
        # Get command line arguments
        numerical_scheme = sys.argv[1]
        time_scheme = sys.argv[i]
        my_sdof = SDoF(time_scheme, numerical_scheme)
    plt.show()


