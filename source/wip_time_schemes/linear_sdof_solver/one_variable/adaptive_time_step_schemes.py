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


def bdf1_disp_based(SDoF, t, dt, un, unm1):

	u_n1 = (SDoF.C*SDoF.dt*un + 2*SDoF.M*un - SDoF.M*unm1 + SDoF.dt**2*SDoF.f(t))/(SDoF.C*SDoF.dt + SDoF.K*SDoF.dt**2 + SDoF.M)
	v_n1 = ( u_n1 - un ) / SDoF.dt

	return u_n1, v_n1


def bdf2_disp_based(SDoF, t, dt, old_dt, un, unm1, unm2, unm3):

	# bdf coefficients for variable time steps
    Rho = old_dt / dt
    TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)    
    bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
    bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
    bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)    

    un1 =  (-SDoF.C*bdf1*un - SDoF.C*bdf2*unm1 - 2*SDoF.M*bdf0*bdf1*un - 2*SDoF.M*bdf0*bdf2*unm1 - SDoF.M*bdf1**2*unm1 - 2*SDoF.M*bdf1*bdf2*unm2 - SDoF.M*bdf2**2*unm3 + SDoF.f(t))/(SDoF.C*bdf0 + SDoF.K + SDoF.M*bdf0**2)
    vn1 = bdf0 * un1 + bdf1 * un + bdf2 * unm1

    return un1, vn1


##### VELOCITY BASED SCHEMES #####
def euler_vel_based(SDoF, t, dt, vn, vnm1, un):

	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def euler_vel_based2(SDoF, t, dt, vn, vnm1, vnm2, un):

	v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt**2*vnm1 + SDoF.M*vn - SDoF.M*(-vn + vnm2) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vnm2 + SDoF.f(t)))/SDoF.M
	#v_n1 =  (-SDoF.C*SDoF.dt*vn - SDoF.K*SDoF.dt*un + SDoF.M*vn + SDoF.dt*SDoF.f(t))/SDoF.M
	u_n1 = un + vn * SDoF.dt

	return v_n1, u_n1


def bdf1_vel_based(SDoF, t, dt, vn, vnm1, un):

	v_n1 =  (SDoF.M*vn - SDoF.M*(-vn + vnm1) + SDoF.dt*SDoF.f(t) - SDoF.dt*(-SDoF.C*vn + SDoF.f(t)))/(SDoF.C*SDoF.dt + SDoF.K*SDoF.dt**2 + SDoF.M)
	u_n1 = un + v_n1 * SDoF.dt

	return v_n1, u_n1


def bdf2_vel_based(SDoF, t, dt, old_dt, vn, vnm1, vnm2, un, unm1):

	# bdf coefficients for variable time steps
    Rho = old_dt / dt
    TimeCoeff = 1.0 / (dt * Rho * Rho + dt * Rho)    
    bdf0 = TimeCoeff * (Rho * Rho + 2.0 * Rho) #coefficient for step n+1 (3/2Dt if Dt is constant)
    bdf1 = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0) #coefficient for step n (-4/2Dt if Dt is constant)
    bdf2 = TimeCoeff #coefficient for step n-1 (1/2Dt if Dt is constant)    

    v_n1 =  (SDoF.K*bdf1*un + SDoF.K*bdf2*unm1 - SDoF.M*bdf0*bdf1*vn - SDoF.M*bdf0*bdf2*vnm1 + SDoF.f(t)*bdf0)/(SDoF.C*bdf0 + SDoF.K + SDoF.M*bdf0**2)
    u_n1 = (v_n1 - bdf1 * un - bdf2 * unm1) / bdf0
    return v_n1, u_n1