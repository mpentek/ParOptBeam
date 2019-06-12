# ===============================================================================
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
# ===============================================================================
import numpy as np
import matplotlib.pyplot as plt

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*


# Create example structure

# Distribution of floor according to their number and floor height
floor_num = 60
floor_height = 3
Z = np.zeros(floor_num + 1)

for i in range(1, floor_num + 1):
    Z[i] = floor_height * i

Z = Z[1:]

nodal_coordinates = {"x0": np.zeros(len(Z)),
                     "y0": Z,
                     "x": None,
                     "y": None}

# Geometrical and material characteristics
rho = 160.
building_length = 45.
building_width = 30.
area = building_length * building_width 

# Vibration characteristics
target_freq = 0.2
target_mode = 1
zeta = 0.05  # Damping ratio

# Structural layout type
# Create object of MDoFMixedModel

# pure bending
# gamma = 1.0
# pure shear
# gamma = 0.0
gamma = 0#.25 # 0.75

mixed_model = MDoFMixed3DModel(
    rho, area, target_freq, target_mode, zeta, floor_height, floor_num, gamma)

#=========================static analysis==========================

#=========================eigen value analysis ==========================
eigenvalue_analysis = EigenvalueAnalysis(mixed_model)
eigenvalue_analysis.solve()
# eigenvalue_analysis.plot_selected_eigenmode(1)
eigenvalue_analysis.plot_selected_first_n_eigenmodes(3)
# eigenvalue_analysis.animate_selected_eigenmode(1)

#=========================dynamic analysis ==========================

