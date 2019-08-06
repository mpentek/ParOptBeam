################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import cmath
import math
import numpy as np


##### DISPLACEMENT BASED SCHEMES #####
def euler_disp_based(SDoF, t, dt, un, unm1):
	K = SDoF.K(un)
	C = SDoF.C
	f = SDoF.f(t)

	u_n1 = (-C*dt*un + C*dt*unm1 - K*dt**2*unm1 + SDoF.M*(2*un - unm1) + dt**2*f)/SDoF.M
	v_n = ( u_n1 - un ) / dt

	return u_n1, v_n


def bdf1_disp_based(SDoF, t, dt, un, unm1):
	K = SDoF.K(un)
	C = SDoF.C
	f = SDoF.f(t)

	u_n1 = (C*dt*un + 2*SDoF.M*un - SDoF.M*unm1 + dt**2*f)/(C*dt + K*dt**2 + SDoF.M)
	v_n1 = ( u_n1 - un ) / dt

	return u_n1, v_n1


def bdf2_disp_based(SDoF, t, dt, old_dt, un, unm1, unm2, unm3):
    K = SDoF.K(un)
    C = SDoF.C
    f = SDoF.f(t)

	# bdf coefficients for variable time steps
    Rho = old_dt / dt
    TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)    
    bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
    bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
    bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)    

    un1 =  (-C*bdf1*un - C*bdf2*unm1 - 2*SDoF.M*bdf0*bdf1*un - 2*SDoF.M*bdf0*bdf2*unm1 - SDoF.M*bdf1**2*unm1 - 2*SDoF.M*bdf1*bdf2*unm2 - SDoF.M*bdf2**2*unm3 + f)/(C*bdf0 + K + SDoF.M*bdf0**2)
    vn1 = bdf0 * un1 + bdf1 * un + bdf2 * unm1

    return un1, vn1


##### VELOCITY BASED SCHEMES #####
def euler_vel_based(SDoF, t, dt, vn, vnm1, un):
	K = SDoF.K(un)
	C = SDoF.C
	f = SDoF.f(t)

	#v_n1 =  (-C*dt*vn - K*dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + dt*f - dt*(-C*vnm2 + f))/SDoF.M
	v_n1 =  (-C*dt*vn - K*dt*un + SDoF.M*vn + dt*f)/SDoF.M
	u_n1 = un + vn * dt

	return v_n1, u_n1


def euler_vel_based2(SDoF, t, dt, vn, vnm1, vnm2, un):
	K = SDoF.K(un)
	C = SDoF.C
	f = SDoF.f(t)

	v_n1 =  (-C*dt*vn - K*dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + dt*f - dt*(-C*vnm2 + f))/SDoF.M
	#v_n1 =  (-C*dt*vn - K*dt*un + SDoF.M*vn + dt*f)/SDoF.M
	u_n1 = un + vn * dt

	return v_n1, u_n1


def bdf1_vel_based(SDoF, t, dt, vn, vnm1, un):
	K = SDoF.K(un)
	C = SDoF.C
	f = SDoF.f(t)

	v_n1 =  (SDoF.M*vn - SDoF.M*(-vn + vnm1) + dt*f - dt*(-C*vn + f))/(C*dt + K*dt**2 + SDoF.M)
	u_n1 = un + v_n1 * dt

	return v_n1, u_n1


def bdf2_vel_based(SDoF, t, dt, old_dt, vn, vnm1, vnm2, un, unm1):
    K = SDoF.K(un)
    C = SDoF.C
    f = SDoF.f(t)

	# bdf coefficients for variable time steps
    Rho = old_dt / dt
    TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)    
    bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
    bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
    bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)    

    v_n1 =  (K*bdf1*un + K*bdf2*unm1 - SDoF.M*bdf0*bdf1*vn - SDoF.M*bdf0*bdf2*vnm1 + f*bdf0)/(C*bdf0 + K + SDoF.M*bdf0**2)
    u_n1 = (v_n1 - bdf1 * un - bdf2 * unm1) / bdf0
    return v_n1, u_n1