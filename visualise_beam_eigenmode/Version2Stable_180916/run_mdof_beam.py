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
import matplotlib.pyplot as plt

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*

# Create example structure

# Distribution of floor according to their number and floor height
floor_num = 3
floor_height = 4
Z = np.zeros(floor_num + 1)

for i in range(1, floor_num + 1):
    Z[i] = floor_height * i
   
Z = Z[1:]

nodal_coordinates = {"x0": np.zeros(len(Z)),
                             "y0": Z,
                             "x": None,
                             "y": None}

# Geometrical and material characteristics
rho = 26. # kg/m^1
building_length = 45.
building_width = 30.
area = building_length * building_width # area of the building m^2

# Vibration characteristics
target_freq = 0.2
target_mode = 1
zeta = 0.05 # Damping ratio

# Structural layout type
name="MDofBeamModel"

# Create object of the MDoBeamModel
beam_model = MDoFBeamModel(rho, area, target_freq, target_mode, zeta, floor_height, floor_num, name)

#=========================static analysis==========================  
# static force definition 

# Generating wind profile
wind_velocity = 20. # m/s
velocityVector = wind_velocity * pow(Z/Z[-1], 0.2)
wind_force = 0.5 * 1.2 * velocityVector**2 * building_width

# Check wind profile and load distribution
plt.plot(velocityVector, Z)
plt.title('Power law Velocity Profile')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Building Height [m]')
plt.text(min(velocityVector), max(Z), 'Atmosphere stability exponent: 0.2')
plt.grid(True)
plt.show()

# Load distribution on the building, separation of forces and moments
static_force = beam_model._load_distribution(floor_height, floor_num, wind_force)
static_analysis = StaticAnalysis(beam_model)
static_analysis.solve(static_force)
static_analysis.plot_solve_result()

#=========================eigen value analysis ==========================  
eigenvalue_analysis = EigenvalueAnalysis(beam_model)
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
# tip: try frequency values of the structure
freq = 1.253

# MDoFModel more degrees of freedom -> 2*len(Z) 2 degrees of freedom for each floor
force_dynamic = load_type("signalSin", array_time, 2*len(Z), freq, wind_force) 

# inital condition 
top_dis = 1 # the displacement at the top of the mdof system for the free vibration case
#u0 = top_dis * np.arange(1/len(Z),1+1/len(Z),1/len(Z))
u0 = np.zeros(2*len(Z)) # initial displacement
v0 = np.zeros(2*len(Z))  # initial velocity
a0 = np.zeros(2*len(Z))  # initial acceleration

"""
Numerical Integration type : Choose from "GenAlpha", "Euler12", "ForwardEuler1", "BackwardEuler1"
"""

dynamic_analysis = DynamicAnalysis(beam_model, np.array([u0,v0,a0]), force_dynamic, [start_time,end_time], 
                   dt, "GenAlpha" ) 

dynamic_analysis.solve()
dynamic_analysis.plot_selected_time_step(0.75)
dynamic_analysis.animate_time_history()





