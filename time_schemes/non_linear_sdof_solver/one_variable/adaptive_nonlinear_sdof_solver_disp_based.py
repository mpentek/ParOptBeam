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
from adaptive_time_schemes import euler_disp_based, bdf1_disp_based, bdf2_disp_based


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
        self.eta_min = 1e-3
        self.eta_max = 1e-1
        self.eta = self.eta_min # minitoring function
        self.max_dt = 10 * self.dt
        self.min_dt = 0.8 * self.dt
        self.rho = 1.1 # amplification factor (should be smaller than 1.91, otherwise stability problems)
        self.sigma = 0.95 # reduction factor
        self.tend = 20.0
        self.epsilon = 1e-6
        self.ua, self.va = [],[]


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def predict(self, t, dt, u, v):
        print("predicting")
        a1 = self.f(t) - self.C * v[-1] - self.K(u[-1]) * u[-1] + self.a0
        v1 = v[-1] + a1 * dt
        u1 = u[-1] + v1 * dt
        u.append(u1)
        v.append(v1)
        
        return u1, v1


    def calculate_residual_disp_based(self, time_scheme, t, dt, dt_old, u_n1, u_n, u_nm1=None, u_nm2=None, u_nm3=None):
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
            Rho = dt_old / dt
            TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)
            bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
            bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
            bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)
       
            if self.numerical_scheme == 'Newton Raphson':
                ru =  -C*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt) - M*(1.5*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt)/dt - 2.0*(1.5*u_n/dt - 2.0*u_nm1/dt + 0.5*u_nm2/dt)/dt + 0.5*(1.5*u_nm1/dt - 2.0*u_nm2/dt + 0.5*u_nm3/dt)/dt) + f - u_n1*self.K(u_n1)
            elif self.numerical_scheme == 'Picard':
                ru =  -C*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt) - M*(1.5*(-2.0*u_n/dt + 1.5*u_n1/dt + 0.5*u_nm1/dt)/dt - 2.0*(1.5*u_n/dt - 2.0*u_nm1/dt + 0.5*u_nm2/dt)/dt + 0.5*(1.5*u_nm1/dt - 2.0*u_nm2/dt + 0.5*u_nm3/dt)/dt) + f - u_n1*self.K(u_n)
        return ru


    def calculate_increment_disp_based(self, time_scheme, t, dt, dt_old, ru, u_n1=None, u_n=None):
        C = self.C
        M = self.M
        f = self.f(t)

        if time_scheme == 'euler':
            du = dt * ru

        if time_scheme == 'bdf1':
            if self.numerical_scheme == 'Newton Raphson':
                du =  dt**2*ru/(C*dt + M + 3*dt**2*u_n1**2 + dt**2)
            elif self.numerical_scheme == 'Picard':
                du =  dt**2*ru/(C*dt + M + dt**2*u_n**2 + dt**2)

        if time_scheme == 'bdf2':
            Rho = dt_old / dt
            TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)
            bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
            bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
            bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)
       
            if self.numerical_scheme == 'Newton Raphson':
                du =  4.0*dt**2*ru/(9.0*M + 12.0*dt**2*u_n1**2 + 4.0*dt**2)
            elif self.numerical_scheme == 'Picard':
                du =  4.0*dt**2*ru/(9.0*M + 4.0*dt**2*u_n**2 + 4.0*dt**2)
        return du


    # Function for the u'(t) = v ODE.
    def g(self, v):
        return v


    def update_time(self, t, dt):
        t += dt
        return t
    

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


    def solve(self, time_scheme):
        u, v = [], []
        t = 0.0
        tstep = 0
        dt = []
        t_vec = []
        delta_time = self.dt
        old_delta_time = self.dt

        while t < self.tend:
            print ("time step: ", tstep)
            if (tstep == 0):
                self.initialize(u, v)
                rv = 0.0
            elif (tstep == 1):
                self.predict(t, delta_time, u, v)
            else:
                u_n1, v_n1 = self.get_first_iteration_step_disp_based(t, delta_time, old_delta_time, tstep, u)

                if (time_scheme == 'bdf2' and tstep <= 3):
                    ru = self.calculate_residual_disp_based('bdf1', t, delta_time, old_delta_time, u_n1, u[-1], u[-2])
                elif (time_scheme == 'bdf2' and tstep > 3):
                    ru = self.calculate_residual_disp_based(time_scheme, t, delta_time, old_delta_time, u_n1, u[-1], u[-2], u[-3], u[-4])
                else:
                    ru = self.calculate_residual_disp_based(time_scheme, t, delta_time, old_delta_time, u_n1, u[-1], u[-2])

                print("ru: ", ru)

                u_n1, v_n1 = self.iterate_in_one_time_step_disp_based(rv, u_n1, t, delta_time, old_delta_time, tstep, v, u)
                
                self.compute_eta(u_n1,u[-1])
                
                u.append(u_n1)
                v.append(v_n1)

            t_vec.append(t)
            dt.append(delta_time)
            tstep += 1
            t = self.update_time(t, delta_time)
            old_delta_time = delta_time
            delta_time = self.update_dt(t, dt[-1])

        return t_vec, u, v


    def get_first_iteration_step_disp_based(self, t, dt, old_dt, tstep, u):
        print(self.time_scheme)
        if (self.time_scheme == 'euler'):
            u_n1, v_n1 =  euler_disp_based(self, t, dt, u[-1],u[-2])

        if (self.time_scheme == 'bdf1'):
            u_n1, v_n1 = bdf1_disp_based(self, t, dt, u[-1],u[-2])

        if (self.time_scheme == 'bdf2'):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_disp_based(self, t, dt, u[-1], u[-2])
            else:
                u_n1, v_n1 = bdf2_disp_based(self, t, dt, old_dt, u[-1], u[-2], u[-3], u[-4])

        return u_n1, v_n1


    def iterate_in_one_time_step_disp_based(self, ru, u_n1, t, dt, old_dt, tstep, v, u):
        it = 0
        while ( abs(ru) >= 1.0e-12) and it < 10:
            if (self.time_scheme == 'bdf2' and tstep == 1):
                du = self.calculate_increment_disp_based('bdf1', t, dt, old_dt, ru, u_n1)
            else:
                du = self.calculate_increment_disp_based(self.time_scheme, t, dt, old_dt, ru, u_n1, u[-1])

            u_n1 += du

            if (self.time_scheme == 'bdf2' and tstep <= 3):
                ru = self.calculate_residual_disp_based('bdf1', t, dt, old_dt, u_n1, u[-1], u[-2])
            elif (self.time_scheme == 'bdf2' and tstep > 3):
                ru = self.calculate_residual_disp_based(self.time_scheme, t, dt, old_dt, u_n1, u[-1], u[-2], u[-3], u[-4])
            else:
                ru = self.calculate_residual_disp_based(self.time_scheme, t, dt, old_dt, u_n1, u[-1], u[-2])

            print("ru: ", ru)

            it += 1
        print("Number of iteration per step: ", it)

        if self.time_scheme == 'euler':
            v_n1 = ( u_n1 - u[-1] ) / dt
        elif self.time_scheme == 'bdf1':
            v_n1 = ( u_n1 - u[-1] ) / dt
        elif self.time_scheme == 'bdf2':
            Rho = old_dt / dt
            TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)
            bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
            bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
            bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)

            v_n1 = bdf0 * u_n1 + bdf1 * u[-1] + bdf2 * u[-2]

        return u_n1, v_n1

