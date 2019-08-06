################################################################################################
###   M * u''(t) + C * u'(t) + K(u) * u(t) = f rewrite nonlinear 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = ( f - C * v(t) - K * u(t) ) / M = f(t, u, v) = rhs
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import numpy as np
from math import *
from sympy import *
from time_schemes import euler_disp_based, euler_vel_based, bdf1_disp_based,  bdf1_vel_based, bdf2_disp_based,  bdf2_vel_based


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

    def calculate_residual_disp_based(self, time_scheme, t, u_n1, u_n, u_nm1=None, u_nm2=None, u_nm3=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            ru =  -C*(u_n - u_nm1)/dt - M*((-u_n + u_n1)/dt - (u_n - u_nm1)/dt)/dt + f - u_nm1*self.K(u_n)

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                ru =  -C*(-u_n + u_n1)/dt - M*((-u_n + u_n1)/dt - (u_n - u_nm1)/dt)/dt + f - u_n1*self.K(u_n1)
            elif self.numerical_scheme == 'Picard':
                ru =  -C*(-u_n + u_n1)/dt - M*((-u_n + u_n1)/dt - (u_n - u_nm1)/dt)/dt + f - u_n1*self.K(u_n)

        if time_scheme == 'bdf2':
            if self.numerical_scheme == 'Newton Raphson':
                ru =  -C*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt) - M*(1.5*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt)/dt - 2.0*(1.5*u_n/dt - 2.0*u_nm1/dt + 0.5*u_nm2/dt)/dt + 0.5*(1.5*u_nm1/dt - 2.0*u_nm2/dt + 0.5*u_nm3/dt)/dt) + f - u_n1*self.K(u_n1)
            elif self.numerical_scheme == 'Picard':
                ru =  -C*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt) - M*(1.5*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt)/dt - 2.0*(1.5*u_n/dt - 2.0*u_nm1/dt + 0.5*u_nm2/dt)/dt + 0.5*(1.5*u_nm1/dt - 2.0*u_nm2/dt + 0.5*u_nm3/dt)/dt) + f - u_n1*self.K(u_n)
        return ru


    def calculate_increment_disp_based(self, time_scheme, t, ru, u_n1=None, u_n=None):
        C = self.C
        M = self.M
        f = self.f(t)
        dt = self.dt

        if time_scheme == 'euler':
            du = dt * ru

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                du =  dt**2*ru/(C*dt + M + 3*dt**2*u_n1**2 + dt**2)
            elif self.numerical_scheme == 'Picard':
                du =  dt**2*ru/(C*dt + M + dt**2*u_n**2 + dt**2)

        if time_scheme == 'bdf2':
            if self.numerical_scheme == 'Newton Raphson':
                du =  4.0*dt**2*ru/(9.0*M + 12.0*dt**2*u_n1**2 + 4.0*dt**2)
            elif self.numerical_scheme == 'Picard':
                du =  4.0*dt**2*ru/(9.0*M + 4.0*dt**2*u_n**2 + 4.0*dt**2)
        return du


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
                ru = 0.0
            elif (tstep == 1):
                self.predict(t, u, v)
            else:
                u_n1, v_n1 = self.get_first_iteration_step_disp_based(t, tstep, u)

                if (time_scheme == 'bdf2' and tstep <= 3):
                    ru = self.calculate_residual_disp_based('bdf1', t, u_n1, u[-1], u[-2])
                elif (time_scheme == 'bdf2' and tstep > 3):
                    ru = self.calculate_residual_disp_based(time_scheme, t, u_n1, u[-1], u[-2], u[-3], u[-4])
                else:
                    ru = self.calculate_residual_disp_based(time_scheme, t, u_n1, u[-1], u[-2])

                print("ru: ", ru)

                u_n1, v_n1 = self.iterate_in_one_time_step_disp_based(ru, u_n1, t, tstep, u)

                u.append(u_n1)
                v.append(v_n1)

            t_vec.append(t)
            t = self.update_time(t)

        return t_vec, u, v


    def get_first_iteration_step_disp_based(self, t, tstep, u):
        print(self.time_scheme)
        if (self.time_scheme == 'euler'):
            u_n1, v_n1 =  euler_disp_based(self, t, u[-1],u[-2])

        if (self.time_scheme == 'bdf1'):
            u_n1, v_n1 = bdf1_disp_based(self, t, u[-1],u[-2])

        if (self.time_scheme == 'bdf2'):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_disp_based(self, t, u[-1], u[-2])
            else:
                u_n1, v_n1 = bdf2_disp_based(self, t, u[-1], u[-2], u[-3], u[-4])

        return u_n1, v_n1


    def iterate_in_one_time_step_disp_based(self, ru, u_n1, t, tstep, u):
        it = 0
        while ( abs(ru) >= 1.0e-12) and it < 10:
            if (self.time_scheme == 'bdf2' and tstep == 1):
                du = self.calculate_increment_disp_based('bdf1', t, ru, u_n1)
            else:
                du = self.calculate_increment_disp_based(self.time_scheme, t, ru, u_n1, u[-1])

            u_n1 += du

            if (self.time_scheme == 'bdf2' and tstep <= 3):
                ru = self.calculate_residual_disp_based('bdf1', t, u_n1, u[-1], u[-2])
            elif (self.time_scheme == 'bdf2' and tstep > 3):
                ru = self.calculate_residual_disp_based(self.time_scheme, t, u_n1, u[-1], u[-2], u[-3], u[-4])
            else:
                ru = self.calculate_residual_disp_based(self.time_scheme, t, u_n1, u[-1], u[-2])

            print("ru: ", ru)

            it += 1
        print("Number of iteration per step: ", it)

        if self.time_scheme == 'euler':
            v_n1 = ( u_n1 - u[-1] ) / self.dt
        elif self.time_scheme == 'bdf1':
            v_n1 = ( u_n1 - u[-1] ) / self.dt
        elif self.time_scheme == 'bdf2':
            v_n1 = 3 * 0.5/self.dt * u_n1 + -4 * 0.5/self.dt * u[-1] + 1 * 0.5/self.dt * u[-2]

        return u_n1, v_n1



