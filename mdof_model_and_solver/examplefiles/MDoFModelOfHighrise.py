#===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
        MDoF system - structural parameters for a sample highrise

Author: mate.pentek@tum.de

        Based upon a collaborative project with A. Michalski, str.ucture GmbH 
        from Stuttgart
                
Description: This is a script containing a proposed build for the stiffness and 
        mass matrix for an MDoF model of a highrise. Data is derived from a full 
        FEM model in Sofistik.
 
Note:   It has been written and tested with Python 2.7.9. Tested and works also with Python 
        3.4.3 (already see differences in print).
        Module dependencies: 
            python
            numpy

Created on:  24.11.2016
Last update: 02.11.2017
'''
#===============================================================================
import numpy as np

#===============================================================================
## number of floors and building height
numberOfFloors = 60

levelHeight = 3.5
Z = np.zeros(numberOfFloors+1)
for i in range(numberOfFloors+1):
    Z[i] = levelHeight * i;

## setting up the mass matrix
massMatrix = np.zeros((numberOfFloors,numberOfFloors))
#precalculated masses - for floors and columns in [kg]
mColumns = 1500000
mFloor   = 750000

[rows,columns] = massMatrix.shape 
# mass-lumping
for i in range(rows):
    if i == 0: #first element
        massMatrix[i,i] = mFloor + 1.0 * mColumns
    elif ((i > 0) & (i <= (rows-2))): #other bottom level
        massMatrix[i,i] = mFloor + 1.0 * mColumns 
    elif i == (rows-1): #top level
        massMatrix[i,i] = mFloor + 0.5 * mColumns
            
print("Total mass check = " + str(np.sum(massMatrix)) + " [kg]")

## setting up the stifness matrix
stiffnessMatrix = np.zeros((numberOfFloors,numberOfFloors))

kX = 1.5e10 # in [N/m]
#kX = 5.0e9 # for lower stiffness
        
[rows,columns] = massMatrix.shape 
for i in range(rows):
    if (i == 0): #first row
        stiffnessMatrix[i,i] = kX * 2
        stiffnessMatrix[i,i+1] = kX * (-1) 
    elif ((i > 0) & (i <= (rows-2))): #intermediate rows
        stiffnessMatrix[i,i-1] = kX * (-1) 
        stiffnessMatrix[i,i] = kX * 2
        stiffnessMatrix[i,i+1] = kX * (-1) 
    elif i == (rows-1): #top row
        stiffnessMatrix[i,i-1] = kX * (-1) 
        stiffnessMatrix[i,i] = kX * 1
   
#===============================================================================    