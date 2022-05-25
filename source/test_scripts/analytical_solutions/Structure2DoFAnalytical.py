# =====================================================================================================================
# Analytical solution of a MDOF system for cos()-excitation
# Compare Structural Dynamics - Tutorial SS14 page 79 ff. and Structural Dynamics - Lecture notes SS 14 page 104 ff.
#
# Author: Dennis Kasper (dennis.kasper@tum.de), Michael Vogl (michael.vogl@tum.de)
# =====================================================================================================================


#Import modules
from cProfile import label
import sys
import numpy as np

class StructureMDoFAnalytical:

    def __init__(self, M, C, K, u0, v0, a0, f, omega, t_start):

        # mass, damping and stiffness matrices
        self.M = M
        self.C = C
        self.K = K

        # initial displacement, velocity and acceleration
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0

        # force amplitude (f(t) = f * cos(Omega*t)
        # only cos() excitation implemented
        self.f = f

        #force angular frequency
        self.omega = omega

        #determine number of DoFs
        self.f_list = self.f.tolist()
        self.nrDoFs = len(self.f_list)

        # start time for homogeneous solution
        self.t_start = t_start


    # SOLVE FOR PARTICULAR AND HOMOGENEOUS SOLUTION
    def solve(self):
        # PARTICULAR SOLUTION
        # dynamic substitute matrix
        self.Ksubst = (-(self.omega**2) * self.M) + (1j * self.omega * self.C) + self.K

        # assemble the RHS, to be sure is 1-D array (otherwise linalg setup as implemeted doesn't work)
        self.RHS = np.zeros(self.nrDoFs)
        for i in np.arange(self.nrDoFs):
            self.RHS[i] = self.f[i,0]

        # solve for the vector u_p (displacement amplitude)
        self.u_p = np.linalg.solve(self.Ksubst, self.RHS)

        # compute v_p (velocity amplitude)
        self.v_p = 1j * self.omega * self.u_p

        # assemble z_p = [u_p, v_p]^T
        self.z_p_amplitude = np.zeros((2 * self.nrDoFs), dtype=complex)
        for i in np.arange(2 * self.nrDoFs):
            if i < self.nrDoFs:
                self.z_p_amplitude[i] = self.u_p[i]
            else:
                self.z_p_amplitude[i] = self.v_p[i - self.nrDoFs]

        # initial z_p at t=t_start for homogeneous solution
        self.z_p0 = self.z_p_amplitude * np.exp(1j * self.omega * self.t_start)

        # HOMOGENEOUS SOLUTION
        # vector of "real" initial conditions z0 (= homogeneous + particular init conditions, z0 = z_h0 + z_p0)
        self.z0 = np.zeros(2 * self.nrDoFs)
        for i in np.arange( 2 * self.nrDoFs):
            if i < self.nrDoFs:
                self.z0[i] = self.u0[i]
            else:
                self.z0[i] = self.v0[i - self.nrDoFs]

        # vector of init conditions for homogeneous solution z_h0 = z0 - z_p0
        self.z_h0 = self.z0 - self.z_p0


        # assemble first order system matrix A  (A*z = dz/dt)
        self.A = np.zeros((2* self.nrDoFs, 2* self.nrDoFs))

        # compute parts of A
        self.invM = np.linalg.inv(self.M)
        self.invMDotK = - np.dot(self.invM, self.K)
        self.invMDotC = - np.dot(self.invM, self.C)

        # set in parts in A
        for i in np.arange(2* self.nrDoFs):
            if i < self.nrDoFs:
                self.A[i, self.nrDoFs + i] = 1
            else:
                for j in np.arange(2 * self.nrDoFs):
                    if j < self.nrDoFs:
                        self.A[i,j] = self.invMDotK[i - self.nrDoFs, j]
                    else:
                        self.A[i,j] = self.invMDotC[i - self.nrDoFs, j - self.nrDoFs]

        # solve for eigenvectors and eigenvalues of A
        [self.eigValue, self.eigVec] = np.linalg.eig(self.A)

        # compute unknowns
        self.unknowns = np.linalg.solve(self.eigVec, self.z_h0)


    # EVALUATE TOTAL SOLUTION AT DISCRETE TIME INSTANCE
    def computeTimeStepResults(self, current_time):
        current_time = current_time + 0j #make it complex

        # homogeneous part
        # z_h(t) = SUM[k=1 to 2n]( unknowns[k] * eigVec[k] * exp(eigValue * current_time)
        self.expLambdaTimesT = np.zeros((2 * self.nrDoFs), dtype=complex)

        for i in np.arange(2 * self.nrDoFs):
            self.expLambdaTimesT[i] = np.exp( self.eigValue[i] * current_time)

        self.unknTimesExpLam = np.multiply( self.unknowns, self.expLambdaTimesT)

        self.z_h = (np.dot( self.eigVec, self.unknTimesExpLam))

        # particular part
        self.z_p = (self.z_p_amplitude * np.exp(1j * self.omega * current_time))

        # total solution
        self.u_complex = self.z_h[0:self.nrDoFs] + self.z_p[0:self.nrDoFs]

        self.v_complex = self.z_h[self.nrDoFs:] + self.z_p[self.nrDoFs:]

        self.a_complex =  ( np.dot( self.invMDotK, self.z_h[0:self.nrDoFs])
                           +np.dot( self.invMDotC, self.z_h[self.nrDoFs:])
                           +self.z_p[self.nrDoFs:] * 1j * self.omega)


    def getDisplacement(self):
        return self.u_complex.real

    def getVelocity(self):
        return self.v_complex.real

    def getAcceleration(self):
        return self.a_complex.real




if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    # the values are from the example in Petersen page 552 Abb 7.35
    # mass matrix

    M = np.zeros((2,2))
    M[0,0] = 500#80.0
    M[1,1] = 10000#8.0

    #damping matrix
    C1 = 3000
    C2 = 300
    C = np.zeros((2,2))
    C[0,0] = C1
    C[0,1] = -C1
    C[1,0] = -C1
    C[1,1] = C1+C2


    #stiffness matrix
    K1 = 8000
    K2 = 200000
    K = np.zeros((2,2))
    K[0,0] = K1
    K[0,1] = -K1
    K[1,0] = -K1
    K[1,1] = K1+K2

    # init conditions
    u_0 = 0.1 * np.ones((2,1))
    v_0 = np.zeros((2,1))
    a_0 = np.zeros((2,1))

    # force vector
    f = np.zeros((2,1))
    f[0,0] = 0.0

    #excitation frequency
    freq = 0.25
    omega = 2*np.pi*freq

    #start time
    t_start = 0.0

    #initialize solver
    solver = StructureMDoFAnalytical(M,C,K,u_0,v_0,a_0,f,omega,t_start)

    #solve for homogeneous and particular solution
    solver.solve()

    #evaluate solution at discrete timesteps
    #end time
    t_end = 10.0
    sample_size = 0.01  # -> step size
    nr_samples = t_end/ sample_size + 1
    array_time = np.linspace(t_start, t_end, int(nr_samples))
    disp_dof1 = np.zeros(len(array_time))
    disp_dof2 = np.zeros(len(array_time))
    vel_dof1 = np.zeros(len(array_time))
    vel_dof2 = np.zeros(len(array_time))

    for i in range(len(array_time)):
        t = array_time[i]

        #evaluate solution at current time
        solver.computeTimeStepResults(t)

        #get results
        disp_dof1[i] = solver.getDisplacement()[0]
        disp_dof2[i] = solver.getDisplacement()[1]
        vel_dof1[i] = solver.getVelocity()[0]
        vel_dof2[i] = solver.getVelocity()[1]


    #plot results
    # f1.suptitle("DoF 1")
    plt.plot(array_time, disp_dof1,      "-r",  lw=1, label='DoF 1')
    plt.plot(array_time, disp_dof2,      "-b",  lw=1, label='DoF 2')
    # ax2.plot(array_time, vel_dof1,       "-b", lw=1.)
    plt.ylabel("Displacement [m]")
    plt.xlabel("Time [s]")
    plt.xlim(min(array_time), max(array_time))
    plt.legend()
    plt.show()



