################################################################################################
###   M * u''(t) + C * u'(t) + K(u) * u(t) = f rewrite nonlinear 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = ( f - C * v(t) - K * u(t) ) / M = f(t, u, v) = rhs
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(dv^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import numpy as np
from math import *
from sympy import *
from time_schemes import euler_vel_based, euler_vel_based, bdf1_vel_based,  bdf1_vel_based, bdf2_vel_based,  bdf2_vel_based


class SDoF:

    def __init__(self, time_scheme=None, numerical_scheme=None, K=1.0, M=1.0, C=0.2, f=None, u0=1.0, v0=0.0, a0=0.0, dt=0.01):

        self.K = lambda u: (u**2 + 1)
        self.Ku = lambda u: self.K(u) * u
        self.M = M
        self.C = C
        self.f = lambda t: 1.0
        self.fstr = '1.0'
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0
        self.dt = dt
        self.time_scheme = time_scheme
        self.numerical_scheme = numerical_scheme
        self.tend = 20.0


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def predict(self, t, u, v):
        print("predicting")
        a1 = self.f(t) - self.C * v[-1] - self.K(u[-1]) * u[-1] + self.a0
        v1 = v[-1] + a1 *self.dt
        u1 = u[-1] + v1 * self.dt
        u.append(u1)
        v.append(v1)
        
        return u1, v1

    def calculate_residual_vel_based(self, time_scheme, t, v_n1, v_n, u_n, v_nm1=None, u_nm1=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            rv =  -C*v_n - M*(-v_n + v_n1)/dt + f - u_n*(u_n**2 + 1)

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                rv =  -C*v_n1 - M*(-v_n + v_n1)/dt + f - (dt*v_n1 + u_n)*((dt*v_n1 + u_n)**2 + 1)
            elif self.numerical_scheme == 'Picard':
                rv =  -C*v_n1 - M*(-v_n + v_n1)/dt + f - (u_n**2 + 1)*(dt*v_n1 + u_n)
        if time_scheme == 'bdf2':
            if self.numerical_scheme == 'Newton Raphson':
                rv =  -C*v_n1 - M*(-2.0*v_n/dt + 1.5*v_n1/dt + 0.5*v_nm1/dt) + f - ((2./3.*dt*v_n1 +  4./3.*u_n - 1./3.*u_nm1)**2 + 1)*(2./3.*dt*v_n1 + 4./3.*u_n - 1./3.*u_nm1)
            elif self.numerical_scheme == 'Picard':
                rv =  -C*v_n1 - M*(-2.0*v_n/dt + 1.5*v_n1/dt + 0.5*v_nm1/dt) + f - (u_n**2 + 1)*(2./3.*dt*v_n1 + 4./3.*u_n - 1./3.*u_nm1)
        return rv


    def calculate_increment_vel_based(self, time_scheme, t, rv, v_n1=None, u_n=None, u_nm1=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            dv =  dt*rv/M

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                dv =  dt*rv/(C*dt + M + 3*dt**2*(dt*v_n1 + u_n)**2 + dt**2)
            elif self.numerical_scheme == 'Picard':
                dv =  dt*rv/(C*dt + M + dt**2*u_n**2 + dt**2)

        if time_scheme == 'bdf2':
            if self.numerical_scheme == 'Newton Raphson':
                dv =  18*dt*rv/(18*C*dt + 27*M + 4*dt**2*(2*dt*v_n1 + 4*u_n - u_nm1)**2 + 12*dt**2)
            elif self.numerical_scheme == 'Picard':
                dv =  18*dt*rv/(18*C*dt + 27*M + 4*dt**2*(2*dt*v_n1 + 4*u_n - u_nm1)**2 + 12*dt**2)
        return dv


    # Function for the u'(t) = v ODE.
    def g(self, v):
        return v


    def update_time(self, t):
        t += self.dt
        return t


    def solve(self, time_scheme):
        u, v = [], []
        t_vec = []
        t = 0.0

        nsteps = int(self.tend/self.dt)

        for tstep in range(0,nsteps):
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
                rv = 0.0
            elif (tstep == 1):
                self.predict(t, u, v)
            else:
                u_n1, v_n1 = self.get_first_iteration_step_vel_based(t, tstep, v, u)

                if (time_scheme == 'bdf2' and tstep <= 3):
                    rv = self.calculate_residual_vel_based('bdf1', t, v_n1, v[-1], u[-1])
                elif (time_scheme == 'bdf2' and tstep > 3):
                    rv = self.calculate_residual_vel_based(time_scheme, t, v_n1, v[-1], u[-1], v[-2], u[-2])
                else:
                    rv = self.calculate_residual_vel_based(time_scheme, t, v_n1, v[-1], u[-1])

                print("rv: ", rv)

                u_n1, v_n1 = self.iterate_in_one_time_step_vel_based(rv, u_n1, t, tstep, v, u)

                u.append(u_n1)
                v.append(v_n1)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec, u, v


    def get_first_iteration_step_vel_based(self, t, tstep, v, u):
        print(self.time_scheme)
        if (self.time_scheme == 'euler'):
            u_n1, v_n1 =  euler_vel_based(self, t, v[-1],v[-2],u[-1])

        if (self.time_scheme == 'bdf1'):
            u_n1, v_n1 = bdf1_vel_based(self, t, v[-1],v[-2],u[-1])

        if (self.time_scheme == 'bdf2'):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_vel_based(self, t, v[-1],v[-2],u[-1])
            else:
                u_n1, v_n1 = bdf2_vel_based(self, t, v[-1],v[-2],v[-3],u[-1],u[-2])

        return u_n1, v_n1


    def iterate_in_one_time_step_vel_based(self, rv, v_n1, t, tstep, v, u):
        it = 0
        while ( abs(rv) >= 1.0e-12) and it < 10:
            if (self.time_scheme == 'bdf2' and tstep == 1):
                dv = self.calculate_increment_vel_based('bdf1', t, rv, v_n1, u[-1])
            else:
                dv = self.calculate_increment_vel_based(self.time_scheme, t, rv, v_n1, u[-1], u[-2])

            v_n1 += dv

            if (self.time_scheme == 'bdf2' and tstep <= 3):
                rv = self.calculate_residual_vel_based('bdf1', t, v_n1, v[-1], u[-1])
            elif (self.time_scheme == 'bdf2' and tstep > 3):
                rv = self.calculate_residual_vel_based(self.time_scheme, t, v_n1, v[-1], u[-1], v[-2], u[-2])
            else:
                rv = self.calculate_residual_vel_based(self.time_scheme, t, v_n1, v[-1], u[-1])

            print("rv: ", rv)

            it += 1
        print("Number of iteration per step: ", it)

        if self.time_scheme == 'euler':
            u_n1 = u[-1] + v[-1] * self.dt
        elif self.time_scheme == 'bdf1':
            u_n1 = u[-1] + v[-1] * self.dt
        elif self.time_scheme == 'bdf2':
            u_n1 = 4/3 * u[-1] - 1/3 * u[-2] + 2/3 * self.dt * v_n1

        return u_n1, v_n1



