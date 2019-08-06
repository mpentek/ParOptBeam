import cmath
import math
import numpy as np

##### DISPLACEMENT BASED SCHEMES #####
def euler_disp_based(SDoF, t, un, unm1):

	u_n1 = (-SDoF.C*SDoF.dt*un + SDoF.C*SDoF.dt*unm1 - SDoF.K(un)*SDoF.dt**2*unm1 + SDoF.M*(2*un - unm1) + SDoF.dt**2*SDoF.f(t))/SDoF.M
	v_n = ( u_n1 - un ) / SDoF.dt

	return u_n1, v_n


def bdf1_disp_based(SDoF, t, un, unm1):

	u_n1 = (SDoF.C*SDoF.dt*un + 2*SDoF.M*un - SDoF.M*unm1 + SDoF.dt**2*SDoF.f(t))/(SDoF.C*SDoF.dt + SDoF.K(un)*SDoF.dt**2 + SDoF.M)
	v_n1 = ( u_n1 - un ) / SDoF.dt

	return u_n1, v_n1


def bdf2_disp_based(SDoF, t, un, unm1, unm2, unm3):

	c0 =  3 * 0.5/SDoF.dt
	c1 =  -4 * 0.5/SDoF.dt
	c2 =  1 * 0.5/SDoF.dt
	u_n1 = (8*SDoF.C*SDoF.dt*un - 2*SDoF.C*SDoF.dt*unm1 + 24*SDoF.M*un - 22*SDoF.M*unm1 + 8*SDoF.M*unm2 - SDoF.M*unm3 + 4*SDoF.dt**2*SDoF.f(t))/(6*SDoF.C*SDoF.dt + 4*SDoF.K(un)*SDoF.dt**2 + 9*SDoF.M)
	v_n1 = c0 * u_n1 + c1 * un + c2 * unm1

	return u_n1, v_n1


##### VELOCITY BASED SCHEMES #####
def euler_vel_based(SDoF, t, vn, vnm1, un):

	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K(un)*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K(un)*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def euler_vel_based2(SDoF, t, vn, vnm1, vnm2, un):

	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K(un)*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K(un)*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def bdf1_vel_based(SDoF, t, vn, vnm1, un):

	v_n1 =  (SDoF.M*vn - SDoF.M*(-vn + vnm1) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vn + SDoF.f(t)))/(SDoF.C*SDoF.dt + SDoF.K(un)*SDoF.dt**2 + SDoF.M)
	u_n1 = un + v_n1 * SDoF.dt

	return v_n1, u_n1


def bdf2_vel_based(SDoF, t, vn, vnm1, vnm2, un, unm1):

	v_n1 =  (-8*SDoF.K(un)*SDoF.dt*un + 2*SDoF.K(un)*SDoF.dt*unm1 + 12*SDoF.M*vn - 3*SDoF.M*vnm1 + 6*SDoF.dt*SDoF.f(t))/(6*SDoF.C*SDoF.dt + 4*SDoF.K(un)*SDoF.dt**2 + 9*SDoF.M)
	u_n1 = 4/3 * un - 1/3 * unm1 + 2/3 * SDoF.dt * v_n1

	return v_n1, u_n1