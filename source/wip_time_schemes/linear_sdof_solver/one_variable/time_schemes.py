################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import cmath
import math
import numpy as np

def analytical_general(SDoF, t):
    # !!! This analytical solution only works for constant f !!!
	if ( (SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M) > 0):
		Delta = math.sqrt( SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M )
	else:
		Delta = cmath.sqrt( SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M )

	A1 = - ( - SDoF.C * SDoF.f(t) + SDoF.u0 * SDoF.C * SDoF.K + 2 * SDoF.v0 * SDoF.K * SDoF.M \
            + SDoF.f(t) * Delta - SDoF.u0 * SDoF.K * Delta ) / (2 * SDoF.K * Delta)
	A2 = - ( SDoF.C * SDoF.f(t) - SDoF.u0 * SDoF.C * SDoF.K - 2 * SDoF.v0 * SDoF.K * SDoF.M \
            + SDoF.f(t) * Delta - SDoF.u0 * SDoF.K * Delta ) /( 2 * SDoF.K * Delta)
	# v0 = 1 / ( 2 * SDoF.M) * ( A2 * ( Delta - A3 ) - A1 * ( Delta + A3 ) )
	A3 = ( A2 * Delta - A1 * Delta  - 2 * SDoF.v0 * SDoF.M ) / ( A1 + A2 )

    # source: Wolfram Alpha
	u_n1 = A1 * cmath.exp( 0.5 * t * (- Delta / SDoF.M  - SDoF.C / SDoF.M )) + \
           A2 * cmath.exp( 0.5 * t * ( Delta / SDoF.M  - SDoF.C / SDoF.M )) + SDoF.f(t) / SDoF.K

	v_n1 = 1 / ( 2 * SDoF.M) * cmath.exp( -  ( t *  ( Delta + A3 ) ) / ( 2 * SDoF.M ) ) * \
		   ( A2 * ( Delta - A3 ) * cmath.exp( t * Delta / SDoF.M ) - A1 * ( Delta + A3 ) )

	return u_n1, v_n1


# Analytical solutions for perticular cases
def analytical(SDoF, t):
	wn = math.sqrt( SDoF.K / SDoF.M )
	xi = SDoF.C / (2 * math.sqrt ( SDoF.K * SDoF.M) )
	#print('xi is', xi)

	A = math.sqrt( SDoF.u0 * SDoF.u0 + (SDoF.v0 / wn) * (SDoF.v0 / wn) )
	phi = np.arctan( SDoF.v0 / (SDoF.u0 * wn) )

	c_crit = 2 * SDoF.M * wn
	D = SDoF.C / c_crit

	if (xi < 1): wd = wn * math.sqrt ( 1 - xi * xi )
	if (xi > 1): D1 = math.sqrt ( D * D - 1 )

	# Analytical solutions for perticular cases
	if (SDoF.fstr == "0.0"):
		# Simple harmonic oscillator without damping for u(0) = 1, v(0) = 0
		if (SDoF.C == 0):
				u_n1 = A * math.cos ( wn*t + phi)
		# Harmonic oscillator with damping for u(0) = 1, v(0) = 0
		else:
			# Underdamped
			if (xi < 1):
				u_n1 = A * math.exp ( - D * wn * t ) * math.cos ( wd * t - phi)
			# Critically overdamped
			# SWE Lecture 3 Structual Anaylsis
			elif (xi > 1):
				u_n1 = ((SDoF.u0 * wn * (D + D1) + SDoF.v0) / (2 * wn * D1 )) * math.exp( (-D + D1) * wn * t ) - \
						((SDoF.u0 * wn * (D - D1) + SDoF.v0) / (2 * wn * D1 )) * math.exp( (-D - D1 )* wn * t )
			# Critically damped
			else:
				u_n1 = A * math.exp ( wn * t ) * ( SDoF.u0 + ( SDoF.v0 + wn * SDoF.u0 ) * t )
	else:
		u_n1 = analytical_general(SDoF, t)

	return u_n1

##### DISPLACEMENT BASED SCHEMES #####
def euler_disp_based(SDoF, t, un, unm1):

	u_n1 = (-SDoF.C*SDoF.dt*un + SDoF.C*SDoF.dt*unm1 - SDoF.K*SDoF.dt**2*unm1 + SDoF.M*(2*un - unm1) + SDoF.dt**2*SDoF.f(t))/SDoF.M
	v_n = ( u_n1 - un ) / SDoF.dt

	return u_n1, v_n


def bdf1_disp_based(SDoF, t, un, unm1):

	u_n1 = (SDoF.C*SDoF.dt*un + 2*SDoF.M*un - SDoF.M*unm1 + SDoF.dt**2*SDoF.f(t))/(SDoF.C*SDoF.dt + SDoF.K*SDoF.dt**2 + SDoF.M)
	v_n1 = ( u_n1 - un ) / SDoF.dt

	return u_n1, v_n1


def bdf2_disp_based(SDoF, t, un, unm1, unm2, unm3):

	c0 =  3 * 0.5/SDoF.dt
	c1 =  -4 * 0.5/SDoF.dt
	c2 =  1 * 0.5/SDoF.dt
	u_n1 = (8*SDoF.C*SDoF.dt*un - 2*SDoF.C*SDoF.dt*unm1 + 24*SDoF.M*un - 22*SDoF.M*unm1 + 8*SDoF.M*unm2 - SDoF.M*unm3 + 4*SDoF.dt**2*SDoF.f(t))/(6*SDoF.C*SDoF.dt + 4*SDoF.K*SDoF.dt**2 + 9*SDoF.M)
	v_n1 = c0 * u_n1 + c1 * un + c2 * unm1

	return u_n1, v_n1


##### VELOCITY BASED SCHEMES #####
def euler_vel_based(SDoF, t, vn, vnm1, un):

	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def euler_vel_based2(SDoF, t, vn, vnm1, vnm2, un):

	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def bdf1_vel_based(SDoF, t, vn, vnm1, un):

	v_n1 =  (SDoF.M*vn - SDoF.M*(-vn + vnm1) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vn + SDoF.f(t)))/(SDoF.C*SDoF.dt + SDoF.K*SDoF.dt**2 + SDoF.M)
	u_n1 = un + v_n1 * SDoF.dt

	return v_n1, u_n1


def bdf2_vel_based(SDoF, t, vn, vnm1, vnm2, un, unm1):

	v_n1 =  (-8*SDoF.K*SDoF.dt*un + 2*SDoF.K*SDoF.dt*unm1 + 12*SDoF.M*vn - 3*SDoF.M*vnm1 + 6*SDoF.dt*SDoF.f(t))/(6*SDoF.C*SDoF.dt + 4*SDoF.K*SDoF.dt**2 + 9*SDoF.M)
	u_n1 = 4/3 * un - 1/3 * unm1 + 2/3 * SDoF.dt * v_n1

	return v_n1, u_n1