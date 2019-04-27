#===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18 
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
        Analysis type base class and derived classes specific types

Author: mate.pentek@tum.de, anoop.kodakkal@tum.de, catharina.czech@tum.de, peter.kupas@tum.de

      
Note:   UPDATE: The script has been written using publicly available information and 
        data, use accordingly. It has been written and tested with Python 2.7.9.
        Tested and works also with Python 3.4.3 (already see differences in print).
        Module dependencies (-> line 61-74): 
            python
            numpy
            sympy
            matplotlib.pyplot

Created on:  22.11.2017
Last update: 30.11.2017
'''
#===============================================================================
import numpy as np

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*

"""

# =========================mdof system properties==========================  
# mdof from density and area 

rho=5.    # the density of the material 
area=10.    # the area  
target_freq=1.    # the target frequency of the system for which stiffness may be tuned 
target_mode=0    # the target mode of the system for which stiffness may be tuned 
zeta=0.05    # the damping-uniform of the system 
level_height=3.5    # the height of each level - uniform 
num_of_levels=3    # the number of levels 
# model definition 
mdof_model = MDoFShearModel(rho, area, target_freq, target_mode, zeta, level_height, num_of_levels)

Z = np.zeros(num_of_levels) # the heright of the floors
for i in range(1,num_of_levels+1):
    Z[i-1] = level_height * i;


# =========================mdof system properties==========================  
# mdof example from AK chopra
# values from A.K. Chopra - Dynamics of Structures - example 11.1
# Stiffness matrix of the structure
K = 610 * np.array([[ 2, -1,  0],
                    [-1,  2, -1],
                    [ 0, -1,  1]])

# Mass matrix of the structure
M = 1/386 * np.array([[400, 0, 0],
                      [0, 400, 0],
                      [0, 0, 200]])

numberOfFloors = 3
levelHeight = 3.5
Z = np.zeros(numberOfFloors)
for i in range(numberOfFloors):
    Z[i] = levelHeight * (i+1);

B = np.zeros (K.shape) # zero damping
nodal_coordinates = {"x0": np.zeros(len(Z)),
                             "y0": Z,
                             "x": None,
                             "y": None}
mdof_model = Simplified2DCantileverStructure(M, B, K, nodal_coordinates, "mdof_shear_model_AK_Chopra", "MDoFShear")

"""
# =========================mdof system properties==========================  
# mdof from model of a highrise 
from examplefiles import MDoFModelOfHighrise
# stiffness matrix
K = MDoFModelOfHighrise.stiffnessMatrix
# mass matrix
M = MDoFModelOfHighrise.massMatrix
# damping matrix
B = np.zeros (K.shape) # zero damping
# height coordinates
Z = MDoFModelOfHighrise.Z
Z = Z[1:]

nodal_coordinates = {"x0": np.zeros(len(Z)),
                             "y0": Z,
                             "x": None,
                             "y": None}

mdof_model = Simplified2DCantileverStructure(M, B, K, nodal_coordinates, "mdof_shear_model_highrise", "MDoFShear")


#=========================static analysis==========================  
# static force definition 


velocityVector = 1.05 * 28.7 * pow(Z/10,0.2)
force_static = 0.5 * 2.0 * 1.2 * 600 * velocityVector**2

static_analysis = StaticAnalysis(mdof_model)
static_analysis.solve(force_static)
static_analysis.plot_solve_result()

#=========================eigen value analysis ==========================  
eigenvalue_analysis = EigenvalueAnalysis(mdof_model)
eigenvalue_analysis.solve()
eigenvalue_analysis.plot_selected_eigenmode(1)
eigenvalue_analysis.plot_selected_first_n_eigenmodes(3)
eigenvalue_analysis.animate_selected_eigenmode(1)

#=========================dynamic analysis ==========================  

# time parameters 
start_time = 0
end_time = 5
dt = 0.01
array_time = np.arange (start_time,end_time + dt, dt)

# dynamic forces
"""
Choose from "signalSin", "signalRand", "signalConst", "signalSuperposed" or 
for free vibration choose "signalNone" 
"""
    
# external dynamic force acting on the system
freq = 10
force_dynamic = load_type("signalSin", array_time, len(Z), freq, force_static) 

# inital condition 
top_dis = 1 # the displacement at the top of the mdof system for the free vibration case
#u0 = top_dis * np.arange(1/len(Z),1+1/len(Z),1/len(Z))
u0 = np.zeros(len(Z)) # initial displacement
v0 = np.zeros(len(Z))  # initial velocity
a0 = np.zeros(len(Z))  # initial acceleration

"""
Numerical Integration type : Choose from "GenAlpha", "Euler12", "ForwardEuler1", "BackwardEuler1"
"""

dynamic_analysis = DynamicAnalysis(mdof_model, np.array([u0,v0,a0]), force_dynamic, [start_time,end_time], 
                   dt, "Euler12" ) 

dynamic_analysis.solve()
dynamic_analysis.plot_selected_time_step(0.75)
dynamic_analysis.animate_time_history()



