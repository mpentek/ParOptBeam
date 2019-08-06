import cmath
import math
import numpy as np


def euler(SDoF, t, dt, un, vn):
    C = SDoF.C
    M = SDoF.M
    K = SDoF.K(un)
    f = SDoF.f(t)

    u_n1 = dt * vn + un 
    v_n1 = ( M * vn - dt * (C * vn + K * un - f) ) / M 
    return u_n1, v_n1


def bdf1(SDoF, t, dt, un, vn):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K(un)
    f = SDoF.f(t)

    u_n1 = (dt*(M*vn + dt*f) + un*(C*dt + M))/(C*dt + K*dt**2 + M)
    v_n1 = (-K*dt*un + M*vn + dt*f)/(C*dt + K*dt**2 + M)

    return u_n1, v_n1


def bdf2(SDoF, t, dt, old_dt, un, vn, unm1, vnm1):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K(un)
    f = SDoF.f(t)

    Rho = old_dt / dt
    TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)    
    bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
    bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
    bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)    
    
    u_n1 =  (-M*bdf1*vn - M*bdf2*vnm1 + f - (C + M*bdf0)*(bdf1*un + bdf2*unm1))/(K + bdf0*(C + M*bdf0))
    v_n1 =  (K*bdf1*un + K*bdf2*unm1 - M*bdf0*bdf1*vn - M*bdf0*bdf2*vnm1 + bdf0*f)/(C*bdf0 + K + M*bdf0**2)

    return u_n1, v_n1
