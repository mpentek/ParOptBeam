################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

############# ONE VARIABLE FORMULATION ################

import numpy as np
from sympy import *
from adaptive_time_step_schemes import analytical_general, euler_disp_based, euler_vel_based, bdf1_disp_based,  bdf1_vel_based, bdf2_disp_based,  bdf2_vel_based
                                             
# DISPLCAMENT_BASED = True -> displacement based formulation
# VELOCITY_BASED = True -> velocity based formulation

class SDoF:

    def __init__(self, DISPLCAMENT_BASED = None, scheme=None, K=1.0, M=1.0, C=0.1, f=None, u0=1.0, v0=0.0, a0=0.0, dt=0.01):
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
        self.ua, self.va = [],[]
        self.use_disp_form = DISPLCAMENT_BASED
        time_scheme = scheme
        self.eta_min = 1e-3
        self.eta_max = 1e-1
        self.eta = self.eta_min # minitoring function
        self.max_dt = 10 * self.dt
        self.min_dt = 0.8 * self.dt
        self.rho = 1.1 # amplification factor (should be smaller than 1.91, otherwise stability problems)
        self.sigma = 0.95 # reduction factor
        self.epsilon = 1e-6


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


    def predict(self, t, u, v):
        print("predicting")
        a1 = self.f(t) - self.C * v[-1] - self.K * u[-1] + self.a0
        v1 = v[-1] + a1 *self.dt
        u1 = u[-1] + v1 * self.dt
        u.append(u1)
        v.append(v1)

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)
        
        return u1, v1


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


    def apply_scheme_disp(self, time_scheme, u_disp, v_disp, t, tstep, dt, dt_old=None):
        ##### DISPLACEMENT BASED SCHEMES #####
        u_n1 = 0
        v_n1 = 0
        if (time_scheme == "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            u_n1, v_n1 = euler_disp_based(self ,t, dt, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf1"):
            u_n1, v_n1 = bdf1_disp_based(self, t, dt, u_disp[-1], u_disp[-2])

        if (time_scheme == "bdf2"):
            if (tstep == 2 or tstep == 3):
                u_n1, v_n1 = bdf1_disp_based(self, t, dt, u_disp[-1], u_disp[-2])
            else:
                u_n1, v_n1 = bdf2_disp_based(self, t, dt, dt_old, u_disp[-1], u_disp[-2], u_disp[-3], u_disp[-4])

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)
        self.compute_eta(v_n1,v_disp[-1])

        u_disp.append(u_n1)
        v_disp.append(v_n1)


    def apply_scheme_vel(self, time_scheme, u_vel, v_vel, t, tstep, dt, dt_old=None):

        ##### VELOCITY BASED SCHEMES #####
        if (time_scheme == "analytical"):
            u_n1, v_n1 = analytical_general(self, t)

        if (time_scheme == "euler"):
            v_n1, u_n1 = euler_vel_based(self, t, dt, v_vel[-1], v_vel[-2], u_vel[-1])

        if (time_scheme == "euler2"):
            if tstep == 2:
                u_n1, v_n1 = self.predict(t, u_vel, v_vel)
                u_vel.pop()
                v_vel.pop()
            else:
                v_n1, u_n1 = euler_vel_based2(self, t, dt, v_vel[-1], v_vel[-2], v_vel[-3], u_vel[-1])

        if (time_scheme == "bdf1"):
            v_n1, u_n1 = bdf1_vel_based(self, t, dt, v_vel[-1], v_vel[-2], u_vel[-1])

        if (time_scheme == "bdf2"):
            if (tstep == 2 or tstep == 3):
                v_n1, u_n1 = bdf1_vel_based(self, t, dt, v_vel[-1], v_vel[-2], u_vel[-1])
            else:
                v_n1, u_n1 = bdf2_vel_based(self, t, dt, dt_old, v_vel[-1], v_vel[-2], v_vel[-3], u_vel[-1], u_vel[-2])

        ua, va = analytical_general(self, t)
        self.ua.append(ua)
        self.va.append(va)
        self.compute_eta(v_n1,v_vel[-1])

        u_vel.append(u_n1)
        v_vel.append(v_n1)


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

            if self.use_disp_form == True:
                if (tstep == 0):
                    self.initialize(u, v)
                    self.initialize(self.ua, self.va)
                elif (tstep == 1):
                    self.predict(t, u, v)
                else:
                    self.apply_scheme_disp(time_scheme, u, v, t, tstep, delta_time, old_delta_time)

            else:
                if (tstep == 0):
                    self.initialize(u, v)
                    self.initialize(self.ua, self.va)
                elif (tstep == 1):
                    self.predict(t, u, v)
                else:
                    self.apply_scheme_vel(time_scheme, u, v, t, tstep, delta_time, old_delta_time)

            t_vec.append(t)
            dt.append(delta_time)
            tstep += 1
            t = self.update_time(t, delta_time)
            old_delta_time = delta_time
            delta_time = self.update_dt(t, dt[-1])

        return t_vec, u, v


    def error_estimation(self, t, u, v):
        eu = []
        ev = []
        for i in range (0,len(u)):
            eu.append(u[i] - self.ua[i])
            ev.append(v[i] - self.va[i])
        return eu, ev


