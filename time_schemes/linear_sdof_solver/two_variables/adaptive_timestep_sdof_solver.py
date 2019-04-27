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

    def __init__(self, scheme="bdf1",  t_factor=1.0, start_accelerating_procentage=1.0, restart_percentage=1.0, max_dt):
        self.K = 1.0
        self.M = (1/np.pi) * (1/np.pi)
        self.C = 1.0
        self.f = lambda t: 0.2
        self.fstr = '0.2'
        self.u0 = 1.0
        self.v0 = 0.0
        self.dt = 0.01
        self.max_dt = 0.05
        self.scheme = scheme
        self.tend_ref = 5 * float(2 * sqrt(self.K))
        self.nstep = int(self.tend_ref / self.dt)
        #self.min_sample_length = (self.tend_ref / self.dt) * 0.1        

        self.start_accelerating_procentage = start_accelerating_procentage
        self.t_factor = t_factor
        self.tend_solver = self.t_factor * self.tend_ref
        self.restart_percentage = restart_percentage
        self.intial_restart_time = self.restart_percentage*self.tend_solver
        self.restart_time = self.intial_restart_time
        self.plot_result()


    def initialize(self, u, v):
        u.append(self.u0)
        v.append(self.v0)


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

        nsteps = int(self.tend_ref/self.dt)

        for tstep in range(0,nsteps):
            #print ("time step: ", tstep)
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

                #print("u_new: ", u_new)

                u.append(u_new)
                v.append(v_new)

            t = self.update_time(t, self.dt)

        return u


    def update_time(self, t, dt):
        t += dt
        return t


    def update_dt(self, alpha):
        dt = self.dt + alpha
        if dt >= self.max_dt:
            dt = self.max_dt
        #print("Current dt is: " + str(dt))
        return dt


    def update_alpha(self, t, alpha):
        #alpha = (1 - (self.tend_ref - t) / self.tend_ref ) * self.dt
        #if alpha == 0:
        if t < self.tend_solver * self.start_accelerating_procentage:
            alpha = 0.0
        else:
            alpha = (self.max_dt - self.dt) * t / self.tend_ref
        #print("Current alpha is: " + str(alpha))
        return alpha


    def update_sampling_size(self, t, s, alpha):
        if s >= self.min_sample_length:
            s = s - (self.tend_ref - t) / self.tend_ref * 5.0
        else:
            s =  self.min_sample_length
        s = int(s)
        #print("Current sample_length is: " + str(s))
        return s


    def check_restart(self, b_u_ave, b_v_ave, time):
        if abs(time-self.restart_time) < self.dt:
            self.restart_time += self.intial_restart_time
            print("########### RESTART ##########")
            b_u_ave.append(b_u_ave[-1])
            b_v_ave.append(b_v_ave[-1])
        return b_u_ave, b_v_ave


    def solve_averaged(self):
        u_ave, v_ave = [], []
        b_u_ave, b_v_ave = [], []
        time = 0.0
        t = []
        dt = []
        dt.append(self.dt)
        tstep = 0
        alpha = 1.0
        _dt = self.dt

        while time < self.tend_solver:
            # tn+1 = t[-1]
            t.append(time)
            dt.append(_dt)
            b_u_ave, b_v_ave = self.check_restart(b_u_ave, b_v_ave, time)

            #print ("time step: ", tstep)

            if (tstep == 0):
                self.initialize(u_ave, v_ave)
                self.initialize(b_u_ave, b_v_ave)
            else:
                if tstep == 1:
                    u_n = b_u_ave[-1]
                    v_n = b_v_ave[-1]

                else:
                    u_n = (b_u_ave[-1] * t[-2] -  b_u_ave[-2] * t[-3]) / dt[-2]
                    v_n = (b_v_ave[-1] * t[-2] -  b_v_ave[-2] * t[-3]) / dt[-2]

                u_n1, v_n1 = self.solve_one_step(tstep, time, u_n, v_n, b_u_ave, b_v_ave)

                u_bar_n1 = ( dt[-1] * u_n1 + t[-2] * b_u_ave[-1]) / t[-1]
                v_bar_n1 = ( dt[-1] * v_n1 + t[-2] * b_v_ave[-1]) / t[-1]

                u_ave.append(u_bar_n1)
                v_ave.append(v_bar_n1)
                b_u_ave.append(u_bar_n1)
                b_v_ave.append(v_bar_n1)

            tstep += 1
            alpha = self.update_alpha(time, alpha)
            _dt = self.update_dt(alpha)
            time = self.update_time(time, _dt)

        return t, u_ave


    def solve_one_step(self, tstep, time, u_n, v_n, u_ave, v_ave):
        if (self.scheme == "analytical"):
            u_n1 = self.analytical(time)
            v_n1 = 0.0

        if (self.scheme == "euler"):
            u_n1, v_n1 = self.euler(time, u_n, v_n)

        if (self.scheme == "bdf1"):
            u_n1, v_n1 = self.bdf1(time, u_n, v_n)

        if (self.scheme == "bdf2"):
            if (tstep == 1):
                u_n1, v_n1 = self.bdf1(time, u_n, v_n)
            else:
                if(tstep == 2):
                    u_nm1 = u_ave[-2]
                    v_nm1 = v_ave[-2]
                else:
                    u_nm1 = (u_ave[-2] - B * u_ave[-3]) / A
                    v_nm1 = (v_ave[-2] - B * v_ave[-3]) / A

                u_n1, v_n1 = self.bdf2(time, u_n, v_n, u_nm1, v_nm1)

        if (self.scheme == "rk4"):
            u_n1, v_n1 = self.rk4(time, u_n, v_n)

        #print("u_n1: ", u_n1)

        return u_n1, v_n1


    def average_result(self, u):
        u_ave = []
        nsteps = int(self.tend_ref/self.dt)

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


    def error_estimation(self, ua, u_solved):
        e = abs(ua[-1] - u_solved[-1])
        print("Error in the end = ", e)


    def plot_result(self):
        global fig, cycol

        plt.title(str(self.M) + 'u\'\'(t) + ' + str(self.C) + 'u\'(t) + ' + str(self.K) + 'u(t) = ' + self.fstr + ', for u(0) = 1, v(0) = 0')
        plt.grid(True)

        nsteps = int(self.tend_ref/self.dt)
        t_ref = np.linspace(0.0, self.tend_ref, nsteps)

        u = self.solve()
        ua = self.average_result(u)
        time, ua_solved = self.solve_averaged()
        self.error_estimation(u, ua_solved)

        plt.plot(t_ref, u, label=self.scheme, c = 'k')
        plt.plot(t_ref, ua, label=self.scheme + ' averaged', c = 'b')
        plt.plot(time, ua_solved, label=self.scheme + ' solve averaged', c = 'r',  markersize=5)
        #print("SOLVER TAKES:          " + str(len(t_ref)) + " TIME STEPS with U_FINAL = " + str(u[-1]))
        #print("                                       and UA_FINAL = " + str(ua[-1]))
        print("AVERAGED SOLVER TAKES: " + str(len(time)) + " TIME STEPS with UA_FINAL = " + str(ua_solved[-1]))
        plt.legend()



if __name__ == "__main__":
    for i in range(1, 11):
        #t_factor=1.0, start_accelerating_procentage=1.0, restart_percentage=1.0
        print("time factor: " + str(i))
        my_sdof = SDoF("bdf1", i, 0.1, 1.0)
    plt.show()


