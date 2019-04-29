import cmath
import math
import numpy as np


def euler(SDoF, t, un, vn):

    u_n1 = SDoF.dt * vn + un
    v_n1 = ( SDoF.M * vn - SDoF.dt * (SDoF.C * vn + SDoF.K * un - SDoF.f(t)) ) / SDoF.M

    return u_n1, v_n1


def bdf1(SDoF, t, un, vn):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K(un)
    f = SDoF.f(t)
    dt = SDoF.dt

    u_n1 = (dt*(M*vn + dt*f) + un*(C*dt + M))/(C*dt + K*dt**2 + M)
    v_n1 = (-K*dt*un + M*vn + dt*f)/(C*dt + K*dt**2 + M)

    return u_n1, v_n1


def bdf2(SDoF, t, un, vn, unm1, vnm1):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K(un)
    f = SDoF.f(t)
    dt = SDoF.dt

    u_n1 = (2.0*dt*(4.0*M*vn - M*vnm1 + 2.0*dt*f) + (4.0*un - unm1)*(2.0*C*dt + 3.0*M))/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)
    v_n1 = (-8.0*K*dt*un + 2.0*K*dt*unm1 + 12.0*M*vn - 3.0*M*vnm1 + 6.0*dt*f)/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)

    return u_n1, v_n1
