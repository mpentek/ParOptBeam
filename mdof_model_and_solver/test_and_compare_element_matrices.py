import numpy as np

m = np.array([[10, 11, 12, 13, 14, 15],
              [20, 21, 22, 23, 24, 25],
              [30, 31, 32, 33, 34, 35]]).transpose()
n = np.array([0, 1, 2, 3, 4, 5])

v = np.zeros((m.shape))
print(v)

n = n[:, np.newaxis]

w = n + v
print(n)
print(w)
print(w+m)

###
##
#

#
##
###

t = 45.0 # t: beam thickness (y) [m]
h = 30.0 # h: beam height (z) [m]
rho = 160.0 # density of steel [kg/mˆ3]
E = 2.861e8 # E: Young's modulus of steel [N/mˆ2]
nu = 3/10 # nu: Poisson's ratio

G = E/2/(1+nu) # G: Shear modulus [N/mˆ2]
l = 3.0 # l: beam element length
A = t*h # beam area [mˆ2]
ASy = 5/6*A    
ASz = 5/6*A # effective area of shear
Iy = 1/12*h**3*t    
Iz = 1/12*t**3*h # second moments of area [mˆ4]
Ip = 1/12*t*h*(h**2+t**2) # polar moment of inertia [mˆ4]
It = min([h,t])**3 *max([h,t])/7 # torsion constant [mˆ4]
Py = 12*E*Iz/(G*ASy*l**2) #
Pz = 12*E*Iy/(G*ASz*l**2) #

length = l
m_const = rho * A * l

###
##
#

#
##
###

def get_mass_matrix_ver1():
    '''
    VERSION 1
    
    NOTE: checking out alternative implementation
    according to https://mediatum.ub.tum.de/doc/1072355/file.pdf
    description and implementation seems correct
    NOTE: find out where the formulation for the mass comes from, stiffness seems standard
    '''
    M11 = np.zeros((6,6))
    M11[0][0] = 1/3
    M11[1][1] = 13/35 + 6*Iz/(5*A*l**2)
    M11[2][2] = 13/35 + 6*Iy/(5*A*l**2)
    M11[3][3] = Ip/(3*A)
    M11[4][4] = l**2/105 + 2*Iy/(15*A)
    M11[5][5] = l**2/105 + 2*Iz/(15*A)
    M11[5][1] = 11*l/210 + Iz/(10*A*l)
    M11[1][5] = M11[5][1]
    M11[4][2] =-11*l/210-Iy/(10*A*l)
    M11[2][4] = M11[4][2]

    M22 = -M11 + 2*np.diag(np.diag(M11))

    M21 = np.zeros((6,6))
    M21[0][0] = 1/6
    M21[1][1] = 9/70-6*Iz/(5*A*l**2)
    M21[2][2] = 9/70-6*Iy/(5*A*l**2)
    M21[3][3] = Ip/(6*A)
    M21[4][4] =-l**2/140-Iy/(30*A)
    M21[5][5] =-l**2/140-Iz/(30*A)
    M21[5][1] =-13*l/420 + Iz/(10*A*l)
    M21[1][5] =-M21[5][1]
    M21[4][2] = 13*l/420-Iy/(10*A*l)
    M21[2][4] =-M21[4][2]

    # mass values for one level
    length = l
    m_const = rho * A * length

    m_el = np.zeros((2*6,2*6))
    # upper left
    m_el[0:6,0:6] += m_const * M11
    # lower left
    m_el[6:12,0:6] += m_const * M21
    # upper right
    m_el[0:6,6:12] += m_const * np.transpose(M21)
    # lower right
    m_el[6:12,6:12] += m_const * M22

    return m_el

def get_mass_matrix_ver2():
    '''
    VERSION 2

    NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
    seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
    implemented mass matrices similar to the stiffness one
    '''
    # define component-wise to have enable better control for various optimization parameters

    # axial inertia - along axis x - here marked as x
    m_x = m_const / 6.0 
    m_x_11 = 2.
    m_x_12 = 1.
    m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                [m_x_12, m_x_11]])
    # torsion inertia - around axis x - here marked as alpha - a
    m_a = m_const * Ip/A / 6.0 
    m_a_11 = 2
    m_a_12 = 1
    m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                [m_a_12, m_a_11]])
    # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
    m_yg = m_const / 420        
    #
    m_yg_11 = 156.
    m_yg_12 = 22. * length
    m_yg_13 = 54.
    m_yg_14 = -13. * length
    #
    m_yg_22 = 4 * length **2
    m_yg_23 = - m_yg_14
    m_yg_24 = -3 * length **2
    #
    m_yg_33 = m_yg_11
    m_yg_34 = -m_yg_12
    #
    m_yg_44 = m_yg_22
    #
    m_el_yg = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

    # bending - inertia along axis z, rotations around axis y - here marked as beta - b
    m_zb = m_const / 420
    #
    m_zb_11 = 156.
    m_zb_12 = -22. * length
    m_zb_13 = 54.
    m_zb_14 = 13. * length
    #
    m_zb_22 = 4. * length **2
    m_zb_23 = -m_zb_14
    m_zb_24 = -3 * length **2
    #
    m_zb_33 = m_zb_11
    m_zb_34 = - m_zb_12 
    #
    m_zb_44 = m_zb_22
    #
    m_el_zb = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])

    # assemble all components
    m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0.,                 m_el_x[0][1], 0., 0., 0., 0., 0.],
                        [0., m_el_yg[0][0], 0., 0., 0., m_el_yg[0][1],     0., m_el_yg[0][2], 0., 0., 0., m_el_yg[0][3]],
                        [0., 0., m_el_zb[0][0], 0., m_el_zb[0][1], 0.,     0., 0., m_el_zb[0][2], 0., m_el_zb[0][3], 0.],
                        [0., 0., 0., m_el_a[0][0], 0., 0.,                 0., 0., 0., m_el_a[0][1], 0., 0.],
                        [0., 0., m_el_zb[0][1], 0., m_el_zb[1][1], 0.,     0., 0., m_el_zb[1][2], 0., m_el_zb[1][3], 0.],
                        [0., m_el_yg[0][1], 0., 0., 0., m_el_yg[1][1],     0., m_el_yg[1][2], 0., 0., 0., m_el_yg[1][3]],
                        
                        [m_el_x[1][0], 0., 0., 0., 0., 0.,                 m_el_x[1][1], 0., 0., 0., 0., 0.],
                        [0., m_el_yg[0][2], 0., 0., 0., m_el_yg[1][2],     0., m_el_yg[2][2], 0., 0., 0., m_el_yg[2][3]],
                        [0., 0., m_el_zb[0][2], 0., m_el_zb[1][2], 0.,     0., 0., m_el_zb[2][2], 0., m_el_zb[2][3], 0.],
                        [0., 0., 0., m_el_a[1][0], 0., 0.,                 0., 0., 0., m_el_a[1][1], 0., 0.],
                        [0., 0., m_el_zb[0][3], 0., m_el_zb[1][3], 0.,     0., 0., m_el_zb[2][3], 0., m_el_zb[3][3], 0.],
                        [0., m_el_yg[0][3], 0., 0., 0., m_el_yg[1][3],     0., m_el_yg[2][3], 0., 0., 0., m_el_yg[3][3]]])

    return m_el

def get_mass_matrix_ver3():
    '''
    VERSION 3
    
    NOTE: from Appendix A - Straight Beam Element Matrices - page 228
    https://link.springer.com/content/pdf/bbm%3A978-3-319-56493-7%2F1.pdf
    
    '''

    # define component-wise to have enable better control for various optimization parameters

    # axial inertia - along axis x - here marked as x
    m_x = m_const / 6.0 
    m_x_11 = 2.
    m_x_12 = 1.
    m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                [m_x_12, m_x_11]])
    # torsion inertia - around axis x - here marked as alpha - a
    m_a = m_const * Ip/A / 6.0 
    m_a_11 = 2
    m_a_12 = 1
    m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                [m_a_12, m_a_11]])

    # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
    # translation
    m_yg = m_const / 210 / (1+Py)**2        
    #
    m_yg_11 = 70*Py**2 + 147*Py + 78
    m_yg_12 = (35*Py**2 + 77*Py + 44) * length / 4
    m_yg_13 = 35*Py**2 + 63*Py + 27
    m_yg_14 = -(35*Py**2 + 63*Py + 26) * length / 4
    #
    m_yg_22 = (7*Py**2 + 14*Py + 8) * length **2 / 4
    m_yg_23 = - m_yg_14 
    m_yg_24 = -(7*Py**2 + 14*Py + 6) * length **2 / 4
    #
    m_yg_33 = m_yg_11
    m_yg_34 = -m_yg_12
    #
    m_yg_44 = m_yg_22
    #
    m_el_yg_trans = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                        [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                        [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                        [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])
    # rotation
    m_yg = rho*Iz / 30 / (1+Py)**2 / length        
    #
    m_yg_11 = 36
    m_yg_12 = -(15*Py-3) * length
    m_yg_13 = -m_yg_11
    m_yg_14 = m_yg_12
    #
    m_yg_22 = (10*Py**2 + 5*Py + 4) * length **2
    m_yg_23 = - m_yg_12
    m_yg_24 = (5*Py**2 - 5*Py -1) * length **2
    #
    m_yg_33 = m_yg_11
    m_yg_34 = - m_yg_12
    #
    m_yg_44 = m_yg_22
    #
    m_el_yg_rot = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                    [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                    [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                    [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

    # sum up translation and rotation
    m_el_yg = m_el_yg_trans + m_el_yg_rot

    # bending - inertia along axis z, rotations around axis y - here marked as beta - b
    # translation
    m_zb = m_const / 210 / (1+Pz)**2        
    #
    m_zb_11 = 70*Pz**2 + 147*Pz + 78
    m_zb_12 = -(35*Pz**2 + 77*Pz + 44) * length / 4
    m_zb_13 = 35*Pz**2 + 63*Pz + 27
    m_zb_14 = (35*Pz**2 + 63*Pz + 26) * length / 4
    #
    m_zb_22 = (7*Pz**2 + 14*Pz + 8) * length **2 / 4
    m_zb_23 = -m_zb_14
    m_zb_24 = -(7*Pz**2 + 14*Pz + 6) * length **2 / 4
    #
    m_zb_33 = m_zb_11
    m_zb_34 = - m_zb_12
    #
    m_zb_44 = m_zb_22
    #
    m_el_zb_trans = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                        [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                        [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                        [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
    # rotation
    m_zb = rho*Iy / 30 / (1+Pz)**2 / length        
    #
    m_zb_11 = 36
    m_zb_12 = (15*Pz-3) * length
    m_zb_13 = -m_zb_11
    m_zb_14 = m_zb_12
    #
    m_zb_22 = (10*Pz**2 + 5*Pz + 4) * length **2
    m_zb_23 = -m_zb_12
    m_zb_24 = (5*Pz**2 - 5*Pz -1) * length ** 2
    #
    m_zb_33 = m_zb_11
    m_zb_34 = -m_zb_12
    #
    m_zb_44 = m_zb_22
    #
    m_el_zb_rot = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                    [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                    [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                    [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])

    # sum up translation and rotation
    m_el_zb = m_el_zb_trans + m_el_zb_rot

    # assemble all components
    m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0.,                 m_el_x[0][1], 0., 0., 0., 0., 0.],
                        [0., m_el_yg[0][0], 0., 0., 0., m_el_yg[0][1],     0., m_el_yg[0][2], 0., 0., 0., m_el_yg[0][3]],
                        [0., 0., m_el_zb[0][0], 0., m_el_zb[0][1], 0.,     0., 0., m_el_zb[0][2], 0., m_el_zb[0][3], 0.],
                        [0., 0., 0., m_el_a[0][0], 0., 0.,                 0., 0., 0., m_el_a[0][1], 0., 0.],
                        [0., 0., m_el_zb[0][1], 0., m_el_zb[1][1], 0.,     0., 0., m_el_zb[1][2], 0., m_el_zb[1][3], 0.],
                        [0., m_el_yg[0][1], 0., 0., 0., m_el_yg[1][1],     0., m_el_yg[1][2], 0., 0., 0., m_el_yg[1][3]],
                        
                        [m_el_x[1][0], 0., 0., 0., 0., 0.,                 m_el_x[1][1], 0., 0., 0., 0., 0.],
                        [0., m_el_yg[0][2], 0., 0., 0., m_el_yg[1][2],     0., m_el_yg[2][2], 0., 0., 0., m_el_yg[2][3]],
                        [0., 0., m_el_zb[0][2], 0., m_el_zb[1][2], 0.,     0., 0., m_el_zb[2][2], 0., m_el_zb[2][3], 0.],
                        [0., 0., 0., m_el_a[1][0], 0., 0.,                 0., 0., 0., m_el_a[1][1], 0., 0.],
                        [0., 0., m_el_zb[0][3], 0., m_el_zb[1][3], 0.,     0., 0., m_el_zb[2][3], 0., m_el_zb[3][3], 0.],
                        [0., m_el_yg[0][3], 0., 0., 0., m_el_yg[1][3],     0., m_el_yg[2][3], 0., 0., 0., m_el_yg[3][3]]])


    return m_el

# mass values for one level
version1_m_el = get_mass_matrix_ver1()
print("\n VERSION1 - m_el")
print(np.array2string(version1_m_el, precision=3, separator=',', suppress_small=True))

version2_m_el = get_mass_matrix_ver2()
print("\n VERSION2 - m_el")
print(np.array2string(version2_m_el, precision=3, separator=',', suppress_small=True))

version3_m_el = get_mass_matrix_ver3()
print("\n VERSION3 - m_el")
print(np.array2string(version3_m_el, precision=3, separator=',', suppress_small=True))

# print("\n NORM - between m_el_ versions:")
# print(np.linalg.norm(version1_m_el - version3_m_el))
# print()
# wait = input("check...")


###
##
#

#
##
###

def get_stiffness_matrix_ver1():
    '''   
    VERSION 1
    
    NOTE: checking out alternative implementation
    according to https://mediatum.ub.tum.de/doc/1072355/file.pdf
    description and implementation seems correct
    NOTE: find out where the formulation for the mass comes from, stiffness seems standard
    '''

    K11 = np.zeros((6,6))
    K11[0][0] = E*A/l
    K11[1][1] = 12*E*Iz/(l**3*(1+Py))
    K11[2][2] = 12*E*Iy/(l**3*(1+Pz))
    K11[3][3] = G*It/l
    K11[4][4] = (4+Pz)*E*Iy/(l*(1+Pz))
    K11[5][5] = (4+Py)*E*Iz/(l*(1+Py))
    K11[1][5] = 6*E*Iz/(l**2*(1+Py))
    K11[5][1] = K11[1][5]
    K11[2][4] =-6*E*Iy/(l**2*(1+Pz))
    K11[4][2] = K11[2][4] 

    K22 = -K11 + 2*np.diag(np.diag(K11))

    K21 = K11 - 2*np.diag(np.diag(K11))
    K21[4][4] = (2-Pz)*E*Iy/(l*(1+Pz))
    K21[5][5] = (2-Py)*E*Iz/(l*(1+Py))
    K21[1][5] =-K21[5][1]
    K21[2][4] =-K21[4][2]

    k_el = np.zeros((2*6,2*6))
    # upper left
    k_el[0:6,0:6] += K11
    # lower left
    k_el[6:12,0:6] += K21
    # upper right
    k_el[0:6,6:12] += np.transpose(K21)
    # lower right
    k_el[6:12,6:12] += K22
    
    return k_el

def get_stiffness_matrix_ver2():
    '''
    VERSION 2
    
    NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
    seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
    implemented mass matrices similar to the stiffness one
    '''

    # stifness values for one level
    # define component-wise to have enable better control for various optimization parameters

    length = l
    # axial stiffness - along axis x - here marked as x
    k_x = E*A/l
    k_x_11 = 1.0
    k_x_12 = -1.0
    k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                [k_x_12, k_x_11]])
    # torsion stiffness - around axis x - here marked as alpha - a
    k_a = G*It/l  # G*K/l
    k_a_11 = 1.0
    k_a_12 = -1.0
    k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                [k_a_12, k_a_11]])
    # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
    beta_yg = Py
    k_yg = E*Iz/(1+beta_yg)/l**3
    #
    k_yg_11 = 12.
    k_yg_12 = 6. * length
    k_yg_13 = -k_yg_11
    k_yg_14 = k_yg_12
    #
    k_yg_22 = (4.+beta_yg) * length **2
    k_yg_23 = -k_yg_12
    k_yg_24 = (2-beta_yg) * length ** 2
    #
    k_yg_33 = k_yg_11
    k_yg_34 = -k_yg_12
    #
    k_yg_44 = k_yg_22
    #
    k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
                                [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
                                [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
                                [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

    # bending - displacement along axis z, rotations around axis y - here marked as beta - b
    beta_zb = Pz
    k_zb = E*Iy/(1+beta_zb)/l**3
    #
    k_zb_11 = 12.
    k_zb_12 = -6. * length
    k_zb_13 = -12.
    k_zb_14 = k_zb_12
    #
    k_zb_22 = (4.+beta_zb) * length **2
    k_zb_23 = -k_zb_12 
    k_zb_24 = (2-beta_zb) * length ** 2
    #
    k_zb_33 = k_zb_11
    k_zb_34 = - k_zb_12
    #
    k_zb_44 = k_zb_22
    #
    k_el_zb = k_zb * np.array([[k_zb_11, k_zb_12, k_zb_13, k_zb_14],
                                [k_zb_12, k_zb_22, k_zb_23, k_zb_24],
                                [k_zb_13, k_zb_23, k_zb_33, k_zb_34],
                                [k_zb_14, k_zb_24, k_zb_34, k_zb_44]])

    # assemble all components
    k_el = np.array([[k_el_x[0][0], 0., 0., 0., 0., 0.,                 k_el_x[0][1], 0., 0., 0., 0., 0.],
                        [0., k_el_yg[0][0], 0., 0., 0., k_el_yg[0][1],     0., k_el_yg[0][2], 0., 0., 0., k_el_yg[0][3]],
                        [0., 0., k_el_zb[0][0], 0., k_el_zb[0][1], 0.,     0., 0., k_el_zb[0][2], 0., k_el_zb[0][3], 0.],
                        [0., 0., 0., k_el_a[0][0], 0., 0.,                 0., 0., 0., k_el_a[0][1], 0., 0.],
                        [0., 0., k_el_zb[0][1], 0., k_el_zb[1][1], 0.,     0., 0., k_el_zb[1][2], 0., k_el_zb[1][3], 0.],
                        [0., k_el_yg[0][1], 0., 0., 0., k_el_yg[1][1],     0., k_el_yg[1][2], 0., 0., 0., k_el_yg[1][3]],
                        
                        [k_el_x[1][0], 0., 0., 0., 0., 0.,                 k_el_x[1][1], 0., 0., 0., 0., 0.],
                        [0., k_el_yg[0][2], 0., 0., 0., k_el_yg[1][2],     0., k_el_yg[2][2], 0., 0., 0., k_el_yg[2][3]],
                        [0., 0., k_el_zb[0][2], 0., k_el_zb[1][2], 0.,     0., 0., k_el_zb[2][2], 0., k_el_zb[2][3], 0.],
                        [0., 0., 0., k_el_a[1][0], 0., 0.,                 0., 0., 0., k_el_a[1][1], 0., 0.],
                        [0., 0., k_el_zb[0][3], 0., k_el_zb[1][3], 0.,     0., 0., k_el_zb[2][3], 0., k_el_zb[3][3], 0.],
                        [0., k_el_yg[0][3], 0., 0., 0., k_el_yg[1][3],     0., k_el_yg[2][3], 0., 0., 0., k_el_yg[3][3]]])

    return k_el


version1_k_el = get_stiffness_matrix_ver1()
print("\n VERSION1 - k_el")
print(np.array2string(version1_k_el, precision=3, separator=',', suppress_small=True))

version2_k_el = get_stiffness_matrix_ver2()
print("\n VERSION2 - k_el")
print(np.array2string(version2_k_el, precision=3, separator=',', suppress_small=True))

# print("\n NORM - k_el versions:")
# print(np.linalg.norm(version1_k_el - version2_k_el))
# print()
# wait = input("check...")